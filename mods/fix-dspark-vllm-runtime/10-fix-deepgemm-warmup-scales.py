#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

NAME = "deepgemm-warmup-scales"
TARGET = Path("model_executor/warmup/deep_gemm_warmup.py")

EXPECTED_PATHS = {
    ("_deepgemm_fp8_gemm_nt_warmup",),
    ("_deepgemm_grouped_fp8_gemm_nt_contiguous_warmup", "_warmup"),
}

# torch.empty is unsafe because the synthetic scale tensor contains
# uninitialized data. torch.zeros and torch.ones are both deterministic,
# finite scale values and therefore safe for warmup.
RECOGNIZED_FACTORIES = {"empty", "zeros", "ones"}
SAFE_FACTORIES = {"zeros", "ones"}


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
            raise RuntimeError(
                f"expected one vllm package root, found: {roots!r}"
            )

        root = Path(roots[0]).resolve()

    if not root.is_dir():
        raise RuntimeError(f"invalid vLLM root: {root}")

    return root


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def line_offsets(text: str) -> list[int]:
    result = [0]

    for line in text.splitlines(keepends=True):
        result.append(result[-1] + len(line))

    return result


def span(text: str, node: ast.AST) -> tuple[int, int]:
    if node.end_lineno is None or node.end_col_offset is None:
        raise RuntimeError(
            "Python AST did not provide source end positions"
        )

    offsets = line_offsets(text)

    return (
        offsets[node.lineno - 1] + node.col_offset,
        offsets[node.end_lineno - 1] + node.end_col_offset,
    )


def torch_factory_name(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        and node.attr in RECOGNIZED_FACTORIES
    ):
        return node.attr

    return None


def assigned_name(
    node: ast.Assign | ast.AnnAssign,
) -> str | None:
    targets = (
        node.targets
        if isinstance(node, ast.Assign)
        else [node.target]
    )

    names = [
        target.id
        for target in targets
        if isinstance(target, ast.Name)
    ]

    return names[0] if len(names) == 1 else None


class ScaleFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_stack: list[str] = []

        self.matches: dict[
            tuple[str, ...],
            list[ast.Attribute],
        ] = {
            path: []
            for path in EXPECTED_PATHS
        }

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> None:
        self.function_stack.append(node.name)

        try:
            self.generic_visit(node)
        finally:
            self.function_stack.pop()

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        self.function_stack.append(node.name)

        try:
            self.generic_visit(node)
        finally:
            self.function_stack.pop()

    def inspect_assignment(
        self,
        node: ast.Assign | ast.AnnAssign,
    ) -> None:
        if assigned_name(node) != "a1q_scales":
            return

        function_path = tuple(self.function_stack)

        if function_path not in self.matches:
            return

        value = node.value

        if not isinstance(value, ast.Call):
            rendered = (
                ast.unparse(value)
                if value is not None
                else "None"
            )

            raise RuntimeError(
                "unexpected a1q_scales expression in "
                f"{function_path!r}: {rendered}"
            )

        factory = torch_factory_name(value.func)

        if factory is None:
            raise RuntimeError(
                "unexpected a1q_scales factory in "
                f"{function_path!r}: "
                f"{ast.unparse(value.func)}"
            )

        assert isinstance(value.func, ast.Attribute)
        self.matches[function_path].append(value.func)

    def visit_Assign(
        self,
        node: ast.Assign,
    ) -> None:
        self.inspect_assignment(node)
        self.generic_visit(node)

    def visit_AnnAssign(
        self,
        node: ast.AnnAssign,
    ) -> None:
        self.inspect_assignment(node)
        self.generic_visit(node)


def inspect(
    text: str,
    path: Path,
) -> tuple[
    ScaleFinder,
    dict[tuple[str, ...], str],
]:
    tree = ast.parse(text, filename=str(path))

    finder = ScaleFinder()
    finder.visit(tree)

    states: dict[tuple[str, ...], str] = {}

    for function_path, matches in finder.matches.items():
        if len(matches) != 1:
            raise RuntimeError(
                "expected one a1q_scales assignment in "
                f"{function_path!r}; found {len(matches)}"
            )

        state = torch_factory_name(matches[0])

        if state is None:
            raise RuntimeError(
                "unable to determine a1q_scales factory in "
                f"{function_path!r}"
            )

        states[function_path] = state

    return finder, states


def patch(
    text: str,
    path: Path,
) -> tuple[str, str]:
    finder, states = inspect(text, path)

    unsafe_paths = {
        function_path
        for function_path, factory in states.items()
        if factory == "empty"
    }

    unexpected = {
        function_path: factory
        for function_path, factory in states.items()
        if factory not in RECOGNIZED_FACTORIES
    }

    if unexpected:
        raise RuntimeError(
            f"unexpected warmup scale state: {unexpected}"
        )

    if not unsafe_paths:
        return (
            text,
            "already safe: synthetic scales use "
            f"{sorted(set(states.values()))}",
        )

    edits: list[tuple[int, int, str]] = []

    for function_path in unsafe_paths:
        matches = finder.matches[function_path]

        if len(matches) != 1:
            raise RuntimeError(
                "internal matcher inconsistency in "
                f"{function_path!r}"
            )

        node = matches[0]
        begin, end = span(text, node)
        source = text[begin:end]

        if source != "torch.empty":
            raise RuntimeError(
                "source verification failed in "
                f"{function_path!r}: found {source!r}"
            )

        edits.append(
            (
                begin,
                end,
                "torch.ones",
            )
        )

    updated = text

    for begin, end, replacement in sorted(
        edits,
        reverse=True,
    ):
        updated = (
            updated[:begin]
            + replacement
            + updated[end:]
        )

    compile(updated, str(path), "exec")

    _, updated_states = inspect(updated, path)

    remaining_unsafe = {
        function_path: factory
        for function_path, factory in updated_states.items()
        if factory not in SAFE_FACTORIES
    }

    if remaining_unsafe:
        raise RuntimeError(
            "post-patch semantic verification failed: "
            f"{remaining_unsafe}"
        )

    return (
        updated,
        "replace uninitialized torch.empty synthetic scales "
        f"with torch.ones in {sorted(unsafe_paths)!r}",
    )


def install(
    path: Path,
    text: str,
) -> None:
    backup = path.with_name(path.name + ".orig")

    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[{NAME}] created backup: {backup}")

    mode = stat.S_IMODE(path.stat().st_mode)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp.write(text)
        temporary = Path(tmp.name)

    try:
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    pycache = path.parent / "__pycache__"

    if pycache.is_dir():
        for cached in pycache.glob(
            f"{path.stem}.*.pyc"
        ):
            cached.unlink()


def main() -> None:
    root = vllm_root()
    path = root / TARGET

    if not path.is_file():
        raise RuntimeError(
            f"target file not found: {path}"
        )

    original = path.read_text(encoding="utf-8")
    updated, action = patch(original, path)

    print(f"[{NAME}] vLLM root: {root}")
    print(f"[{NAME}] target: {path}")
    print(f"[{NAME}] action: {action}")

    if updated == original:
        print(
            f"[{NAME}] already applied or equivalent "
            "upstream fix present"
        )
        return

    before = sha256(original.encode("utf-8"))

    install(path, updated)

    installed = path.read_text(encoding="utf-8")

    if installed != updated:
        raise RuntimeError(
            "post-write verification failed"
        )

    after = sha256(installed.encode("utf-8"))

    print(f"[{NAME}] before SHA256: {before}")
    print(f"[{NAME}] installed SHA256: {after}")
    print(f"[{NAME}] applied: {action}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"[{NAME}] ERROR: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
