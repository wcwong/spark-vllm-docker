#!/usr/bin/env python3
from __future__ import annotations

import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patchlib import (  # noqa: E402
    apply_replacements,
    full_line_end,
    full_line_start,
    install_plans,
    locate_vllm_root,
    make_plan,
    node_byte_span,
)

FIX_NAME = "packed-kv-zeroing"
HELPER_NAME = "zero_kv_cache_blocks_inplace"

HELPER_SOURCE = '''def zero_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor | list[torch.Tensor]],
    num_blocks: int,
    block_ids: Sequence[int],
) -> None:
    """Zero complete physical blocks in block-major packed KV storage.

    Packed KV tensors are offset views into a shared allocation. Their block
    stride spans the complete physical block, so zeroing from each view's
    data_ptr() can overlap neighboring components. Deduplicate by underlying
    storage and clear each physical block exactly once.
    """
    if not block_ids:
        return
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")

    storage_tensors: list[torch.Tensor] = []
    seen_storage: set[int] = set()
    for entry in kv_caches:
        tensors = entry if isinstance(entry, (list, tuple)) else (entry,)
        for tensor in tensors:
            storage_ptr = tensor.untyped_storage().data_ptr()
            if storage_ptr in seen_storage:
                continue
            seen_storage.add(storage_ptr)
            storage_tensors.append(tensor)

    if not storage_tensors:
        return

    block_ids_np = np.asarray(block_ids, dtype=np.int64)
    if np.any(block_ids_np < 0) or np.any(block_ids_np >= num_blocks):
        raise IndexError(
            f"KV block id outside [0, {num_blocks}): {block_ids_np.tolist()}"
        )

    device = storage_tensors[0].device
    block_ids_gpu = async_tensor_h2d(block_ids_np, device=device)
    for tensor in storage_tensors:
        if tensor.device != device:
            raise RuntimeError(
                "Packed KV cache storages must be on one device: "
                f"expected {device}, found {tensor.device}"
            )
        storage_bytes = torch.empty(0, dtype=torch.uint8, device=device)
        storage_bytes.set_(tensor.untyped_storage())
        if storage_bytes.numel() % num_blocks != 0:
            raise RuntimeError(
                "Packed KV storage size is not divisible by num_blocks: "
                f"{storage_bytes.numel()} bytes / {num_blocks} blocks"
            )
        physical_blocks = storage_bytes.view(num_blocks, -1)
        physical_blocks.index_fill_(0, block_ids_gpu, 0)


'''


def top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def patch_utils(source: bytes, filename: str) -> tuple[bytes, str]:
    text = source.decode("utf-8")
    tree = ast.parse(text, filename=filename)
    functions = top_level_functions(tree)
    description = "add storage-aware physical-block zeroing helper"

    if HELPER_NAME in functions:
        helper_text = ast.get_source_segment(text, functions[HELPER_NAME]) or ""
        required_fragments = (
            "untyped_storage().data_ptr()",
            "storage_bytes.view(num_blocks, -1)",
            "index_fill_",
        )
        if not all(fragment in helper_text for fragment in required_fragments):
            raise RuntimeError(
                f"Existing {HELPER_NAME} does not match the expected implementation"
            )
        return source, description

    copy_func = functions.get("copy_kv_cache_blocks_inplace")
    if copy_func is None:
        raise RuntimeError("Could not find copy_kv_cache_blocks_inplace insertion point")

    required_imports = {"Iterable", "Sequence", "np", "torch", "async_tensor_h2d"}
    missing = required_imports - imported_names(tree)
    if missing:
        raise RuntimeError(
            "Required imports are missing from vllm.v1.worker.utils: "
            + ", ".join(sorted(missing))
        )

    insert_at = full_line_start(copy_func, source)
    patched = apply_replacements(
        source,
        [(insert_at, insert_at, HELPER_SOURCE.encode("utf-8"))],
    )
    compile(patched.decode("utf-8"), filename, "exec")
    verify_tree = ast.parse(patched.decode("utf-8"), filename=filename)
    if HELPER_NAME not in top_level_functions(verify_tree):
        raise RuntimeError("Helper insertion verification failed")
    return patched, description


def is_scheduler_zero_test(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "new_block_ids_to_zero"
        and isinstance(node.value, ast.Name)
        and node.value.id == "scheduler_output"
    )


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
    return None


def contains_call(node: ast.AST, name: str) -> bool:
    return any(call_name(child) == name for child in ast.walk(node))


class ModelRunnerFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.zero_if_nodes: list[ast.If] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self.function_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        if (
            self.class_stack
            and self.class_stack[-1] == "GPUModelRunner"
            and self.function_stack
            and self.function_stack[-1] == "update_requests"
            and is_scheduler_zero_test(node.test)
        ):
            self.zero_if_nodes.append(node)
        self.generic_visit(node)


def patch_worker_utils_import(
    source: bytes, tree: ast.Module
) -> tuple[bytes, bool]:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "vllm.v1.worker.utils"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one import from vllm.v1.worker.utils; "
            f"found {len(matches)}"
        )
    node = matches[0]
    names = [alias.name for alias in node.names]
    if HELPER_NAME in names:
        return source, False
    names.append(HELPER_NAME)
    preferred_order = [
        "KVBlockZeroer",
        "copy_kv_cache_blocks_inplace",
        HELPER_NAME,
    ]
    ordered = [name for name in preferred_order if name in names]
    ordered.extend(sorted(name for name in names if name not in ordered))
    replacement = (
        "from vllm.v1.worker.utils import (\n"
        + "".join(f"    {name},\n" for name in ordered)
        + ")"
    ).encode("utf-8")
    start, end = node_byte_span(node, source)
    return apply_replacements(source, [(start, end, replacement)]), True


def patch_model_runner(source: bytes, filename: str) -> tuple[bytes, str]:
    description = "route packed KV caches through storage-aware block zeroing"
    text = source.decode("utf-8")
    tree = ast.parse(text, filename=filename)

    finder = ModelRunnerFinder()
    finder.visit(tree)
    if len(finder.zero_if_nodes) != 1:
        raise RuntimeError(
            "Expected one new_block_ids_to_zero block in "
            f"GPUModelRunner.update_requests; found {len(finder.zero_if_nodes)}"
        )
    zero_if = finder.zero_if_nodes[0]
    already_patched = contains_call(zero_if, HELPER_NAME)

    source_with_import, import_changed = patch_worker_utils_import(source, tree)
    if already_patched:
        if import_changed:
            compile(source_with_import.decode("utf-8"), filename, "exec")
            return source_with_import, description
        return source, description

    if not contains_call(zero_if, "zero_block_ids"):
        raise RuntimeError(
            "The original zeroing block does not call kv_block_zeroer.zero_block_ids"
        )

    # Reparse after changing the import so source positions remain correct.
    text2 = source_with_import.decode("utf-8")
    tree2 = ast.parse(text2, filename=filename)
    finder2 = ModelRunnerFinder()
    finder2.visit(tree2)
    if len(finder2.zero_if_nodes) != 1:
        raise RuntimeError("Could not relocate zeroing block after import patch")
    zero_if2 = finder2.zero_if_nodes[0]

    indent = " " * zero_if2.col_offset
    replacement = f'''{indent}if scheduler_output.new_block_ids_to_zero:
{indent}    block_ids_to_zero = scheduler_output.new_block_ids_to_zero
{indent}    if any(
{indent}        tensor.block_stride > 0
{indent}        for tensor in self.kv_cache_config.kv_cache_tensors
{indent}    ):
{indent}        zero_kv_cache_blocks_inplace(
{indent}            self.kv_caches,
{indent}            self.kv_cache_config.num_blocks,
{indent}            block_ids_to_zero,
{indent}        )
{indent}    else:
{indent}        assert self.kv_block_zeroer is not None
{indent}        self.kv_block_zeroer.zero_block_ids(block_ids_to_zero)
'''.encode("utf-8")
    start = full_line_start(zero_if2, source_with_import)
    end = full_line_end(zero_if2, source_with_import)
    patched = apply_replacements(source_with_import, [(start, end, replacement)])
    compile(patched.decode("utf-8"), filename, "exec")

    verify_tree = ast.parse(patched.decode("utf-8"), filename=filename)
    verify = ModelRunnerFinder()
    verify.visit(verify_tree)
    if len(verify.zero_if_nodes) != 1 or not contains_call(
        verify.zero_if_nodes[0], HELPER_NAME
    ):
        raise RuntimeError("Model runner semantic verification failed")
    return patched, description


def main() -> None:
    root = locate_vllm_root()
    print(f"[{FIX_NAME}] vLLM root: {root}", flush=True)
    utils_target = root / "v1" / "worker" / "utils.py"
    runner_target = root / "v1" / "worker" / "gpu" / "model_runner.py"
    try:
        plans = [
            make_plan(utils_target, patch_utils),
            make_plan(runner_target, patch_model_runner),
        ]
        install_plans(FIX_NAME, plans)
    except (UnicodeDecodeError, SyntaxError, RuntimeError) as exc:
        raise SystemExit(f"[{FIX_NAME}] refusing to patch vLLM: {exc}") from exc


if __name__ == "__main__":
    main()
