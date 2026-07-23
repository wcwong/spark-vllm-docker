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
                # block-major storage slab. In that layout, cur_bytes is the
                # complete physical scheduler-block stride, while
                # spec.page_size_bytes is only this layer's logical page.
                if cur_bytes > spec.page_size_bytes:
                    assert block_dim == 0, (
                        "Packed KV cache views must use a blocks-first layout"
                    )
                    storage = kv.untyped_storage()
                    storage_ptr = storage.data_ptr()
                    previous_stride = packed_storage_strides.get(storage_ptr)
                    if previous_stride is not None:
                        assert previous_stride == cur_bytes, (
                            "Packed KV cache views sharing a storage allocation "
                            f"have different block strides: {previous_stride} "
                            f"vs {cur_bytes}"
                        )
                        continue
                    assert storage.nbytes() % cur_bytes == 0, (
                        "Packed KV cache storage must contain whole physical blocks"
                    )
                    packed_storage_strides[storage_ptr] = cur_bytes
                    cur_page_el = cur_bytes // 4

                    if page_size_el is None:
                        page_size_el = cur_page_el
                    else:
                        assert page_size_el == cur_page_el, (
                            f"Non-uniform page sizes: {page_size_el} vs {cur_page_el}"
                        )
                    seg_addrs.append(storage_ptr)
                    continue

                # Preserve the upstream path for ordinary cache allocations,
                # including virtual block splitting and outer K/V enumeration.
                dp = kv.data_ptr()
                if dp in seen_ptrs:
                    continue
                seen_ptrs.add(dp)

                kernel_block_el = cur_bytes // 4
                cur_page_el = kernel_block_el * ratio

                if page_size_el is None:
                    page_size_el = cur_page_el
                else:
                    assert page_size_el == cur_page_el, (
                        f"Non-uniform page sizes: {page_size_el} vs {cur_page_el}"
                    )

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


def install(path: Path, new_text: str) -> None:
    backup = path.with_name(path.name + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[{NAME}] created backup: {backup}")

    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as tmp:
        tmp.write(new_text)
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, mode)
    os.replace(tmp_path, path)

    pycache = path.parent / "__pycache__"
    if pycache.is_dir():
        for cached in pycache.glob(f"{path.stem}.*.pyc"):
            cached.unlink()


def node_text(node: ast.AST) -> str:
    return ast.unparse(node)


def is_name_target(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
        if isinstance(node, ast.Assign)
        else isinstance(node, ast.AnnAssign)
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
            f"expected one 'for layer_name in group.layer_names' loop, "
            f"found {len(matches)}"
        )
    return matches[0]


def validate_old_loop(layer_loop: ast.For) -> tuple[ast.stmt, ast.stmt]:
    body = layer_loop.body
    dp_indexes = [
        i
        for i, node in enumerate(body)
        if is_name_target(node, "dp") and node_text(node) == "dp = kv.data_ptr()"
    ]
    if len(dp_indexes) != 1:
        raise RuntimeError(
            f"expected one 'dp = kv.data_ptr()' in layer loop, found {len(dp_indexes)}"
        )
    start = dp_indexes[0]
    expected = [
        (ast.Assign, "dp = kv.data_ptr()"),
        (ast.If, "if dp in seen_ptrs:\n    continue"),
        (ast.Expr, "seen_ptrs.add(dp)"),
        (ast.Assign, "el = kv.element_size()"),
        (ast.Assign, "cur_bytes = kv.stride(block_dim) * el"),
        (ast.Assert, "assert cur_bytes % 4 == 0"),
        (ast.Assign, "kernel_block_el = cur_bytes // 4"),
        (ast.Assign, "cur_page_el = kernel_block_el * ratio"),
        (ast.If, None),
        (ast.Assign, "block_stride_bytes = cur_bytes"),
        (ast.Assign, None),
        (ast.Assign, None),
        (ast.For, None),
    ]
    candidate = body[start : start + len(expected)]
    if len(candidate) != len(expected):
        raise RuntimeError("KVBlockZeroer layer-loop body is shorter than expected")
    for index, (node, (kind, exact)) in enumerate(zip(candidate, expected)):
        if not isinstance(node, kind):
            raise RuntimeError(
                f"unexpected node {index} in zeroer sequence: "
                f"expected {kind.__name__}, got {type(node).__name__}"
            )
        if exact is not None and node_text(node) != exact:
            raise RuntimeError(
                f"unexpected statement {index} in zeroer sequence: {node_text(node)!r}"
            )

    page_if = candidate[8]
    assert isinstance(page_if, ast.If)
    if node_text(page_if.test) != "page_size_el is None":
        raise RuntimeError("unexpected page-size guard in KVBlockZeroer")

    outer_dims = candidate[10]
    assert isinstance(outer_dims, ast.Assign)
    if node_text(outer_dims.targets[0]) != "outer_dims":
        raise RuntimeError("expected outer_dims assignment in KVBlockZeroer")
    if "for d in range(block_dim)" not in node_text(outer_dims):
        raise RuntimeError("unexpected outer_dims enumeration in KVBlockZeroer")
    if "kv.stride(d) * el > block_stride_bytes" not in node_text(outer_dims):
        raise RuntimeError("unexpected outer_dims stride filter in KVBlockZeroer")

    outer_strides = candidate[11]
    assert isinstance(outer_strides, ast.Assign)
    if node_text(outer_strides.targets[0]) != "outer_strides":
        raise RuntimeError("expected outer_strides assignment in KVBlockZeroer")

    outer_loop = candidate[12]
    assert isinstance(outer_loop, ast.For)
    if not isinstance(outer_loop.target, ast.Name) or outer_loop.target.id != "outer":
        raise RuntimeError("expected outer-segment loop in KVBlockZeroer")
    if not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "seg_addrs"
        and node.func.attr == "append"
        and node.args
        and node_text(node.args[0]) == "dp + off_bytes"
        for node in ast.walk(outer_loop)
    ):
        raise RuntimeError("expected seg_addrs.append(dp + off_bytes)")

    return candidate[0], candidate[-1]


def offset(lines: list[str], lineno: int, col: int) -> int:
    return sum(len(line) for line in lines[: lineno - 1]) + col


def main() -> None:
    root = vllm_root()
    path = root / TARGET
    if not path.is_file():
        raise RuntimeError(f"target file not found: {path}")

    original = path.read_text(encoding="utf-8")
    tree = ast.parse(original, filename=str(path))
    init = find_class_method(tree)
    layer_loop = find_layer_loop(init)

    setup_old = [node for node in init.body if is_name_target(node, "seen_ptrs")]
    setup_new = [
        node for node in init.body if is_name_target(node, "packed_storage_strides")
    ]
    if len(setup_old) != 1:
        raise RuntimeError(f"expected one seen_ptrs declaration, found {len(setup_old)}")
    if len(setup_new) > 1:
        raise RuntimeError(
            f"expected at most one packed_storage_strides declaration, "
            f"found {len(setup_new)}"
        )

    has_storage_logic = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "untyped_storage"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "kv"
        for node in ast.walk(layer_loop)
    )

    if len(setup_new) == 1 and has_storage_logic:
        print(f"[{NAME}] vLLM root: {root}")
        print(f"[{NAME}] target: {path}")
        print(f"[{NAME}] already applied")
        return
    if bool(setup_new) != has_storage_logic:
        raise RuntimeError(f"partially applied packed KV zeroing fix in {path}")

    old_start, old_end = validate_old_loop(layer_loop)
    if old_start.end_lineno is None or old_end.end_lineno is None:
        raise RuntimeError("Python AST does not provide end positions")

    lines = original.splitlines(keepends=True)
    loop_start = offset(lines, old_start.lineno, 0)
    loop_end = offset(lines, old_end.end_lineno, old_end.end_col_offset)
    if loop_end < len(original) and original[loop_end] == "\n":
        loop_end += 1

    seen_node = setup_old[0]
    if seen_node.end_lineno is None:
        raise RuntimeError("Python AST does not provide setup end position")
    setup_pos = offset(lines, seen_node.end_lineno, seen_node.end_col_offset)
    if setup_pos < len(original) and original[setup_pos] == "\n":
        setup_pos += 1

    edits = [
        (loop_start, loop_end, NEW_LOOP),
        (setup_pos, setup_pos, SETUP_LINE),
    ]
    updated = original
    for start, end, replacement in sorted(edits, reverse=True):
        updated = updated[:start] + replacement + updated[end:]

    compile(updated, str(path), "exec")

    # Verify the installed form structurally before writing it.
    updated_tree = ast.parse(updated, filename=str(path))
    updated_init = find_class_method(updated_tree)
    updated_layer_loop = find_layer_loop(updated_init)
    if not any(
        is_name_target(node, "packed_storage_strides") for node in updated_init.body
    ):
        raise RuntimeError("updated source lacks packed_storage_strides")
    if not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "untyped_storage"
        for node in ast.walk(updated_layer_loop)
    ):
        raise RuntimeError("updated source lacks storage-aware zeroing logic")

    print(f"[{NAME}] vLLM root: {root}")
    print(f"[{NAME}] target: {path}")
    before = sha256(original.encode())
    install(path, updated)
    after = sha256(path.read_bytes())
    print(f"[{NAME}] before SHA256: {before}")
    print(f"[{NAME}] installed SHA256: {after}")
    print(
        f"[{NAME}] applied: zero packed KV cache blocks from their physical "
        "storage base"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{NAME}] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
