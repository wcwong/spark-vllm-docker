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

FIX_NAME = "kv-zeroer-init"


def expr_ast(expression: str) -> ast.expr:
    return ast.parse(expression, mode="eval").body


def same_ast(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


ORIGINAL_TEST = expr_ast(
    'kv_cache_config.needs_kv_cache_zeroing and '
    'hasattr(self.model_runner, "_init_kv_zero_meta")'
)
TARGETED_TEST = expr_ast(
    '(kv_cache_config.needs_kv_cache_zeroing or '
    'self.cache_config.cache_dtype == "fp8_ds_mla") and '
    'hasattr(self.model_runner, "_init_kv_zero_meta")'
)
UNCONDITIONAL_TEST = expr_ast(
    'hasattr(self.model_runner, "_init_kv_zero_meta")'
)


def is_zero_meta_call(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "_init_kv_zero_meta"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "model_runner"
        and isinstance(call.func.value.value, ast.Name)
        and call.func.value.value.id == "self"
        and not call.args
        and not call.keywords
    )


class Finder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_stack: list[str] = []
        self.matches: list[ast.If] = []

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
            self.function_stack
            and self.function_stack[-1] == "initialize_from_config"
            and any(is_zero_meta_call(stmt) for stmt in node.body)
        ):
            self.matches.append(node)
        self.generic_visit(node)


def classify(test: ast.expr) -> str:
    if same_ast(test, ORIGINAL_TEST):
        return "original"
    if same_ast(test, TARGETED_TEST):
        return "targeted"
    if same_ast(test, UNCONDITIONAL_TEST):
        return "unconditional"
    return "unknown"


def patch(source: bytes, filename: str) -> tuple[bytes, str]:
    tree = ast.parse(source.decode("utf-8"), filename=filename)
    finder = Finder()
    finder.visit(tree)
    if len(finder.matches) != 1:
        raise RuntimeError(
            "Expected exactly one KV zeroer initialization guard in "
            f"initialize_from_config; found {len(finder.matches)}"
        )

    node = finder.matches[0]
    state = classify(node.test)
    description = "initialize KV zeroing metadata for fp8_ds_mla"
    if state in {"targeted", "unconditional"}:
        return source, description
    if state != "original":
        raise RuntimeError(
            "Unexpected KV zeroer initialization condition: "
            f"{ast.unparse(node.test)}"
        )

    replacement = b'''(
            kv_cache_config.needs_kv_cache_zeroing
            or self.cache_config.cache_dtype == "fp8_ds_mla"
        ) and hasattr(
            self.model_runner, "_init_kv_zero_meta"
        )'''
    start, end = node_byte_span(node.test, source)
    patched = apply_replacements(source, [(start, end, replacement)])
    compile(patched.decode("utf-8"), filename, "exec")

    verify_tree = ast.parse(patched.decode("utf-8"), filename=filename)
    verify = Finder()
    verify.visit(verify_tree)
    if len(verify.matches) != 1 or classify(verify.matches[0].test) != "targeted":
        raise RuntimeError("Semantic verification failed")
    return patched, description


def main() -> None:
    root = locate_vllm_root()
    print(f"[{FIX_NAME}] vLLM root: {root}", flush=True)
    target = root / "v1" / "worker" / "gpu_worker.py"
    try:
        plan = make_plan(target, patch)
        install_plans(FIX_NAME, [plan])
    except (UnicodeDecodeError, SyntaxError, RuntimeError) as exc:
        raise SystemExit(f"[{FIX_NAME}] refusing to patch vLLM: {exc}") from exc


if __name__ == "__main__":
    main()
