#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile

NAME = "packed-kv-zeroing"
TARGET = Path("v1/worker/utils.py")

SETUP_LINE = "        packed_storage_strides: dict[int, int] = {}\n"

NEW_LOOP = '''                el = kv.element_size()
                cur_bytes = kv.stride(block_dim) * el
                assert cur_bytes % 4 == 0

                # Packed cache groups expose per-layer views into one shared,
                # block-major storage slab. cur_bytes is the physical stride
                # for one kernel block; ratio maps a scheduler block to one or
                # more kernel blocks. Compare complete scheduler-block sizes.
                physical_page_bytes = cur_bytes * ratio
                if physical_page_bytes > spec.page_size_bytes:
                    assert block_dim == 0, (
                        "Packed KV cache views must use a blocks-first layout"
                    )
                    storage = kv.untyped_storage()
                    storage_ptr = storage.data_ptr()
                    previous_stride = packed_storage_strides.get(storage_ptr)
                    if previous_stride is not None:
                        assert previous_stride == physical_page_bytes, (
                            "Packed KV cache views sharing a storage allocation "
                            f"have different scheduler-block strides: "
                            f"{previous_stride} vs {physical_page_bytes}"
                        )
                        continue
                    assert storage.nbytes() % physical_page_bytes == 0, (
                        "Packed KV cache storage must contain whole physical "
                        "scheduler blocks"
                    )
                    packed_storage_strides[storage_ptr] = physical_page_bytes
                    cur_page_el = physical_page_bytes // 4
                    seg_addrs.append(storage_ptr)
                    seg_page_sizes.append(cur_page_el)
                    continue

                # Preserve the upstream path for ordinary non-packed cache
                # allocations, including virtual block splitting and outer
                # K/V-buffer enumeration.
                dp = kv.data_ptr()
                if dp in seen_ptrs:
                    continue
                seen_ptrs.add(dp)
                kernel_block_el = cur_bytes // 4
                cur_page_el = kernel_block_el * ratio
                block_stride_bytes = cur_bytes
                outer_dims = [
                    d
                    for d in range(block_dim)
                    if kv.stride(d) * el > block_stride_bytes
                ]
                outer_strides = [kv.stride(d) * el for d in outer_dims]
                for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                    off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                    seg_addrs.append(dp + off_bytes)
                    seg_page_sizes.append(cur_page_el)
'''


def vllm_root() -> Path:
    override = os.environ.get("VLLM_ROOT")
    if override:
        root = Path(override).resolve()
    else:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("unable to locate the installed vllm package")
        roots = list(spec.submodule_search_locations)
        if len(roots) != 1:
            raise RuntimeError(f"expected one vllm package root, found: {roots!r}")
        root = Path(roots[0]).resolve()
    if not root.is_dir():
        raise RuntimeError(f"invalid vLLM root: {root}")
    return root


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def install(path: Path, text: str) -> None:
    backup = path.with_name(path.name + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[{NAME}] created backup: {backup}")

    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as tmp:
        tmp.write(text)
        temporary = Path(tmp.name)
    os.chmod(temporary, mode)
    os.replace(temporary, path)

    pycache = path.parent / "__pycache__"
    if pycache.is_dir():
        for cached in pycache.glob(f"{path.stem}.*.pyc"):
            cached.unlink()


def node_text(node: ast.AST) -> str:
    return ast.unparse(node)


def assignment_targets_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.Assign):
        return any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    return (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
    )


def find_class_method(tree: ast.Module) -> ast.FunctionDef:
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KVBlockZeroer"
    ]
    if len(classes) != 1:
        raise RuntimeError(f"expected one KVBlockZeroer class, found {len(classes)}")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    if len(methods) != 1:
        raise RuntimeError(
            f"expected one KVBlockZeroer.__init__, found {len(methods)}"
        )
    return methods[0]


def find_layer_loop(init: ast.FunctionDef) -> ast.For:
    matches: list[ast.For] = []
    for node in ast.walk(init):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.target, ast.Name) or node.target.id != "layer_name":
            continue
        if node_text(node.iter) == "group.layer_names":
            matches.append(node)
    if len(matches) != 1:
        raise RuntimeError(
            "expected one 'for layer_name in group.layer_names' loop, "
            f"found {len(matches)}"
        )
    return matches[0]


def call_is(
    node: ast.AST,
    receiver: str,
    method: str,
    argument: str | None = None,
) -> bool:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == receiver
        and node.func.attr == method
    ):
        return False
    if argument is None:
        return True
    return len(node.args) == 1 and node_text(node.args[0]) == argument


def find_old_region(layer_loop: ast.For) -> tuple[ast.stmt, ast.stmt]:
    body = layer_loop.body
    starts = [
        index
        for index, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and node_text(node) == "dp = kv.data_ptr()"
    ]
    if len(starts) != 1:
        raise RuntimeError(
            f"expected one 'dp = kv.data_ptr()' in layer loop, found {len(starts)}"
        )
    start = starts[0]

    required_statements = {
        "seen_ptrs.add(dp)",
        "el = kv.element_size()",
        "cur_bytes = kv.stride(block_dim) * el",
        "kernel_block_el = cur_bytes // 4",
        "cur_page_el = kernel_block_el * ratio",
        "block_stride_bytes = cur_bytes",
    }
    rendered = {node_text(node) for node in body[start:]}
    missing = sorted(required_statements - rendered)
    if missing:
        raise RuntimeError(f"unexpected KVBlockZeroer layer loop; missing {missing}")

    outer_loops = [
        node
        for node in body[start:]
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "outer"
    ]
    if len(outer_loops) != 1:
        raise RuntimeError(f"expected one outer-segment loop, found {len(outer_loops)}")
    outer_loop = outer_loops[0]

    has_addr = any(
        call_is(node, "seg_addrs", "append", "dp + off_bytes")
        for node in ast.walk(outer_loop)
    )
    has_size = any(
        call_is(node, "seg_page_sizes", "append", "cur_page_el")
        for node in ast.walk(outer_loop)
    )
    if not has_addr or not has_size:
        raise RuntimeError(
            "expected outer loop to append both segment address and page size"
        )

    return body[start], outer_loop


def offset(lines: list[str], lineno: int, col: int) -> int:
    return sum(len(line) for line in lines[: lineno - 1]) + col


def has_packed_logic(init: ast.FunctionDef, layer_loop: ast.For) -> bool:
    setup = any(
        assignment_targets_name(node, "packed_storage_strides")
        for node in init.body
    )
    storage_call = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "untyped_storage"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "kv"
        for node in ast.walk(layer_loop)
    )
    storage_addr = any(
        call_is(node, "seg_addrs", "append", "storage_ptr")
        for node in ast.walk(layer_loop)
    )

    # The ordinary upstream path already appends cur_page_el, so that call
    # alone does not indicate that the packed-storage fix is present.
    packed_markers = (setup, storage_call, storage_addr)
    if packed_markers == (False, False, False):
        return False
    if packed_markers == (True, True, True):
        storage_size = any(
            call_is(node, "seg_page_sizes", "append", "cur_page_el")
            for node in ast.walk(layer_loop)
        )
        if not storage_size:
            raise RuntimeError(
                "packed-storage path does not append its physical page size"
            )
        return True
    raise RuntimeError("partially applied packed KV zeroing fix")


def main() -> None:
    root = vllm_root()
    path = root / TARGET
    if not path.is_file():
        raise RuntimeError(f"target file not found: {path}")

    original = path.read_text(encoding="utf-8")
    tree = ast.parse(original, filename=str(path))
    init = find_class_method(tree)
    layer_loop = find_layer_loop(init)

    seen_nodes = [
        node
        for node in init.body
        if assignment_targets_name(node, "seen_ptrs")
    ]
    if len(seen_nodes) != 1:
        raise RuntimeError(
            f"expected one seen_ptrs declaration, found {len(seen_nodes)}"
        )

    seg_size_nodes = [
        node
        for node in init.body
        if assignment_targets_name(node, "seg_page_sizes")
    ]
    if len(seg_size_nodes) != 1:
        raise RuntimeError(
            f"expected one seg_page_sizes declaration, found {len(seg_size_nodes)}"
        )

    print(f"[{NAME}] vLLM root: {root}")
    print(f"[{NAME}] target: {path}")

    if has_packed_logic(init, layer_loop):
        print(f"[{NAME}] already applied")
        return

    old_start, old_end = find_old_region(layer_loop)
    if old_end.end_lineno is None or old_end.end_col_offset is None:
        raise RuntimeError("Python AST does not provide source end positions")

    lines = original.splitlines(keepends=True)
    loop_start = offset(lines, old_start.lineno, 0)
    loop_end = offset(lines, old_end.end_lineno, old_end.end_col_offset)
    if loop_end < len(original) and original[loop_end] == "\n":
        loop_end += 1

    seen_node = seen_nodes[0]
    if seen_node.end_lineno is None or seen_node.end_col_offset is None:
        raise RuntimeError("Python AST does not provide setup end positions")
    setup_pos = offset(lines, seen_node.end_lineno, seen_node.end_col_offset)
    if setup_pos < len(original) and original[setup_pos] == "\n":
        setup_pos += 1

    edits = [
        (loop_start, loop_end, NEW_LOOP),
        (setup_pos, setup_pos, SETUP_LINE),
    ]

    updated = original
    for begin, end, replacement in sorted(edits, reverse=True):
        updated = updated[:begin] + replacement + updated[end:]

    compile(updated, str(path), "exec")
    updated_tree = ast.parse(updated, filename=str(path))
    updated_init = find_class_method(updated_tree)
    updated_layer_loop = find_layer_loop(updated_init)
    if not has_packed_logic(updated_init, updated_layer_loop):
        raise RuntimeError("post-patch packed-storage verification failed")

    before = sha256(original.encode("utf-8"))
    install(path, updated)
    installed = path.read_text(encoding="utf-8")
    if installed != updated:
        raise RuntimeError("post-write verification failed")
    after = sha256(installed.encode("utf-8"))

    print(f"[{NAME}] before SHA256: {before}")
    print(f"[{NAME}] installed SHA256: {after}")
    print(
        f"[{NAME}] applied: zero packed KV cache blocks once from their "
        "physical storage base"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{NAME}] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

