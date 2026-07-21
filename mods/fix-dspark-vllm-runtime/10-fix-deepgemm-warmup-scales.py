#!/usr/bin/env python3
from __future__ import annotations

import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patchlib import (  # noqa: E402
    apply_replacements,
    install_plans,
    locate_vllm_root,
    make_plan,
    node_byte_span,
)

FIX_NAME = "deepgemm-warmup-scales"
EXPECTED_SCALE_PATHS = {
    ("_deepgemm_fp8_gemm_nt_warmup",),
    ("_deepgemm_grouped_fp8_gemm_nt_contiguous_warmup", "_warmup"),
}


def function_name(func: ast.expr) -> str | None:
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
        and func.attr in {"empty", "ones"}
    ):
        return func.attr
    return None


class FunctionPathVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_stack: list[str] = []

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


class ScaleFinder(FunctionPathVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.matches: dict[tuple[str, ...], list[ast.Attribute]] = {
            path: [] for path in EXPECTED_SCALE_PATHS
        }

    def _visit_assignment(self, node: ast.Assign | ast.AnnAssign) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "a1q_scales"
            for target in targets
        ):
            return
        path = tuple(self.function_stack)
        if path not in self.matches:
            return
        value = node.value
        if not isinstance(value, ast.Call) or function_name(value.func) is None:
            rendered = ast.unparse(value) if value is not None else "None"
            raise RuntimeError(
                f"Unexpected a1q_scales assignment in {path!r}: {rendered}"
            )
        assert isinstance(value.func, ast.Attribute)
        self.matches[path].append(value.func)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._visit_assignment(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._visit_assignment(node)
        self.generic_visit(node)


def patch(source: bytes, filename: str) -> tuple[bytes, str]:
    tree = ast.parse(source.decode("utf-8"), filename=filename)
    finder = ScaleFinder()
    finder.visit(tree)

    nodes: dict[tuple[str, ...], ast.Attribute] = {}
    for path, matches in finder.matches.items():
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one a1q_scales assignment in {path!r}; "
                f"found {len(matches)}"
            )
        nodes[path] = matches[0]

    states = {path: function_name(node) for path, node in nodes.items()}
    state_values = set(states.values())
    description = "DeepGEMM synthetic warmup scales use torch.ones"
    if state_values == {"ones"}:
        return source, description
    if state_values != {"empty"}:
        raise RuntimeError(f"Partial or unexpected patch state: {states}")

    replacements: list[tuple[int, int, bytes]] = []
    for path, node in nodes.items():
        start, end = node_byte_span(node, source)
        if source[start:end] != b"torch.empty":
            raise RuntimeError(
                f"Source verification failed at {path!r}: "
                f"found {source[start:end]!r}"
            )
        replacements.append((start, end, b"torch.ones"))

    patched = apply_replacements(source, replacements)
    compile(patched.decode("utf-8"), filename, "exec")

    verify_tree = ast.parse(patched.decode("utf-8"), filename=filename)
    verify = ScaleFinder()
    verify.visit(verify_tree)
    verify_states = {
        path: function_name(matches[0])
        for path, matches in verify.matches.items()
        if len(matches) == 1
    }
    if verify_states != {path: "ones" for path in EXPECTED_SCALE_PATHS}:
        raise RuntimeError(f"Semantic verification failed: {verify_states}")
    return patched, description


def main() -> None:
    root = locate_vllm_root()
    print(f"[{FIX_NAME}] vLLM root: {root}", flush=True)
    target = root / "model_executor" / "warmup" / "deep_gemm_warmup.py"
    try:
        plan = make_plan(target, patch)
        install_plans(FIX_NAME, [plan])
    except (UnicodeDecodeError, SyntaxError, RuntimeError) as exc:
        raise SystemExit(f"[{FIX_NAME}] refusing to patch vLLM: {exc}") from exc


if __name__ == "__main__":
    main()
