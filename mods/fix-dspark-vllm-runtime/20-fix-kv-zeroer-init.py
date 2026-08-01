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

NAME = "kv-zeroer-init"
MODEL_RUNNER = Path("v1/worker/gpu/model_runner.py")
GPU_WORKER = Path("v1/worker/gpu_worker.py")
UTILS = Path("v1/worker/utils.py")

ZEROER_IMPORT = "from vllm.v1.worker.utils import KVBlockZeroer\n"

ZEROER_METHOD = '''    def _init_kv_zero_meta(self) -> None:
        """Build KV-block zeroing metadata; invoked from gpu_worker."""
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=(
                group for groups in self.attn_groups for group in groups
            ),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            static_forward_context=(
                self.compilation_config.static_forward_context
            ),
        )

'''

TARGET_GUARD_TEST = ast.parse(
    "kv_cache_config.needs_kv_cache_zeroing "
    'or self.cache_config.cache_dtype == "fp8_ds_mla"',
    mode="eval",
).body


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


def offsets(text: str) -> list[int]:
    result = [0]
    for line in text.splitlines(keepends=True):
        result.append(result[-1] + len(line))
    return result


def start_of(text: str, node: ast.AST, *, decorators: bool = False) -> int:
    line = node.lineno
    if decorators and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.decorator_list:
            line = min(line, *(item.lineno for item in node.decorator_list))
    return offsets(text)[line - 1]


def end_of(text: str, node: ast.AST) -> int:
    if node.end_lineno is None or node.end_col_offset is None:
        raise RuntimeError("Python AST did not provide source end positions")
    end = offsets(text)[node.end_lineno - 1] + node.end_col_offset
    if end < len(text) and text[end] == "\n":
        end += 1
    return end


def apply(text: str, edits: list[tuple[int, int, str]]) -> str:
    updated = text
    for begin, end, replacement in sorted(edits, reverse=True):
        updated = updated[:begin] + replacement + updated[end:]
    return updated


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


def find_class(tree: ast.Module, name: str, path: Path) -> ast.ClassDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} in {path}, found {len(matches)}")
    return matches[0]


def methods(cls: ast.ClassDef, name: str) -> list[ast.FunctionDef]:
    return [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]


def find_worker_class(tree: ast.Module, path: Path) -> ast.ClassDef:
    named = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"Worker", "GPUWorker"}
        and len(methods(node, "initialize_from_config")) == 1
    ]
    if len(named) == 1:
        return named[0]

    structural = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and len(methods(node, "initialize_from_config")) == 1
    ]
    if len(structural) != 1:
        names = [node.name for node in structural]
        raise RuntimeError(
            "expected one worker class defining initialize_from_config in "
            f"{path}; found {names}"
        )
    return structural[0]


def imports_name(tree: ast.Module, name: str) -> bool:
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for alias in node.names:
            if (alias.asname or alias.name.split(".")[-1]) == name:
                return True
    return False


def is_self_attr(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def assigned_value(node: ast.AST, name: str) -> ast.expr | None:
    if isinstance(node, ast.Assign) and any(
        is_self_attr(target, name) for target in node.targets
    ):
        return node.value
    if isinstance(node, ast.AnnAssign) and is_self_attr(node.target, name):
        return node.value
    return None


def validate_zeroer_constructor(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = find_class(tree, "KVBlockZeroer", path)
    init = methods(cls, "__init__")
    if len(init) != 1:
        raise RuntimeError(f"expected one KVBlockZeroer.__init__ in {path}")
    args = init[0].args
    names = {
        arg.arg
        for arg in args.posonlyargs + args.args + args.kwonlyargs
    }
    required = {
        "device",
        "attn_groups_iter",
        "kernel_block_sizes",
        "cache_dtype",
        "static_forward_context",
    }
    missing = sorted(required - names)
    if missing and args.kwarg is None:
        raise RuntimeError(
            f"KVBlockZeroer.__init__ lacks expected parameters: {missing}"
        )


def validate_zeroer_method(method: ast.FunctionDef) -> None:
    calls = []
    for node in ast.walk(method):
        value = assigned_value(node, "kv_block_zeroer")
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "KVBlockZeroer"
        ):
            calls.append(value)
    if len(calls) != 1:
        raise RuntimeError(
            "_init_kv_zero_meta must assign exactly one KVBlockZeroer instance"
        )
    keywords = {item.arg for item in calls[0].keywords if item.arg is not None}
    required = {
        "attn_groups_iter",
        "kernel_block_sizes",
        "cache_dtype",
        "static_forward_context",
    }
    if not required.issubset(keywords):
        raise RuntimeError(
            f"unexpected KVBlockZeroer constructor keywords: {sorted(keywords)}"
        )


def patch_model_runner(text: str, path: Path) -> tuple[str, str]:
    tree = ast.parse(text, filename=str(path))
    cls = find_class(tree, "GPUModelRunner", path)
    init = methods(cls, "__init__")
    if len(init) != 1:
        raise RuntimeError(f"expected one GPUModelRunner.__init__ in {path}")

    initializers = [
        node
        for node in ast.walk(init[0])
        if assigned_value(node, "kv_block_zeroer") is not None
    ]
    if len(initializers) != 1:
        raise RuntimeError(
            "expected GPUModelRunner.__init__ to initialize "
            f"self.kv_block_zeroer once; found {len(initializers)}"
        )

    edits: list[tuple[int, int, str]] = []
    actions: list[str] = []

    if not imports_name(tree, "KVBlockZeroer"):
        imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        if not imports:
            raise RuntimeError(f"no import section found in {path}")
        position = max(end_of(text, node) for node in imports)
        edits.append((position, position, ZEROER_IMPORT))
        actions.append("import KVBlockZeroer")

    zeroer = methods(cls, "_init_kv_zero_meta")
    if len(zeroer) > 1:
        raise RuntimeError(f"multiple _init_kv_zero_meta methods found in {path}")
    if zeroer:
        validate_zeroer_method(zeroer[0])
    else:
        anchor = methods(cls, "_dummy_run")
        if len(anchor) != 1:
            raise RuntimeError(f"expected one GPUModelRunner._dummy_run in {path}")
        position = start_of(text, anchor[0], decorators=True)
        edits.append((position, position, ZEROER_METHOD))
        actions.append("add GPUModelRunner._init_kv_zero_meta")

    if not edits:
        return text, "already applied upstream"

    updated = apply(text, edits)
    compile(updated, str(path), "exec")
    updated_tree = ast.parse(updated, filename=str(path))
    updated_cls = find_class(updated_tree, "GPUModelRunner", path)
    updated_zeroer = methods(updated_cls, "_init_kv_zero_meta")
    if len(updated_zeroer) != 1 or not imports_name(updated_tree, "KVBlockZeroer"):
        raise RuntimeError("post-patch GPUModelRunner verification failed")
    validate_zeroer_method(updated_zeroer[0])
    return updated, "; ".join(actions)


def model_runner_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "model_runner"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
    )


def same_ast(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


def target_guard(node: ast.If) -> bool:
    return (
        same_ast(node.test, TARGET_GUARD_TEST)
        and not node.orelse
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Expr)
        and model_runner_call(node.body[0].value, "_init_kv_zero_meta")
    )


def guard_text(indent: str) -> str:
    return (
        f"{indent}if (\n"
        f"{indent}    kv_cache_config.needs_kv_cache_zeroing\n"
        f'{indent}    or self.cache_config.cache_dtype == "fp8_ds_mla"\n'
        f"{indent}):\n"
        f"{indent}    self.model_runner._init_kv_zero_meta()\n"
    )


def patch_gpu_worker(text: str, path: Path) -> tuple[str, str]:
    tree = ast.parse(text, filename=str(path))
    cls = find_worker_class(tree, path)
    init = methods(cls, "initialize_from_config")
    if len(init) != 1:
        raise RuntimeError(
            f"expected one {cls.name}.initialize_from_config in {path}"
        )

    cache_calls = [
        node
        for node in ast.walk(init[0])
        if isinstance(node, ast.Expr)
        and model_runner_call(node.value, "initialize_kv_cache")
    ]
    if len(cache_calls) != 1:
        raise RuntimeError(
            "expected one self.model_runner.initialize_kv_cache(...) call; "
            f"found {len(cache_calls)}"
        )

    guards = [
        node
        for node in ast.walk(init[0])
        if isinstance(node, ast.If)
        and any(
            model_runner_call(candidate, "_init_kv_zero_meta")
            for candidate in ast.walk(node)
        )
    ]
    if len(guards) > 1:
        raise RuntimeError(f"multiple KV-zero initialization guards found in {path}")
    if guards and target_guard(guards[0]):
        return text, f"already applied in {cls.name}"

    if guards:
        node = guards[0]
        indent = " " * node.col_offset
        updated = apply(
            text,
            [(start_of(text, node), end_of(text, node), guard_text(indent))],
        )
        action = f"replace optional KV-zeroer guard in {cls.name}"
    else:
        node = cache_calls[0]
        indent = " " * node.col_offset
        position = end_of(text, node)
        updated = apply(text, [(position, position, guard_text(indent))])
        action = f"add required KV-zeroer initialization in {cls.name}"

    compile(updated, str(path), "exec")
    updated_tree = ast.parse(updated, filename=str(path))
    updated_cls = find_worker_class(updated_tree, path)
    updated_init = methods(updated_cls, "initialize_from_config")
    verified = [
        node
        for node in ast.walk(updated_init[0])
        if isinstance(node, ast.If) and target_guard(node)
    ]
    if len(verified) != 1:
        raise RuntimeError("post-patch worker verification failed")
    return updated, action


def main() -> None:
    root = vllm_root()
    paths = {
        "model_runner": root / MODEL_RUNNER,
        "gpu_worker": root / GPU_WORKER,
        "utils": root / UTILS,
    }
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"target file not found: {path}")

    print(f"[{NAME}] vLLM root: {root}")
    validate_zeroer_constructor(paths["utils"])
    print(f"[{NAME}] validated KVBlockZeroer: {paths['utils']}")

    originals = {
        paths["model_runner"]: paths["model_runner"].read_text(encoding="utf-8"),
        paths["gpu_worker"]: paths["gpu_worker"].read_text(encoding="utf-8"),
    }
    updates = {
        paths["model_runner"]: patch_model_runner(
            originals[paths["model_runner"]], paths["model_runner"]
        ),
        paths["gpu_worker"]: patch_gpu_worker(
            originals[paths["gpu_worker"]], paths["gpu_worker"]
        ),
    }

    changed = False
    for path, (updated, action) in updates.items():
        original = originals[path]
        print(f"[{NAME}] target: {path}")
        print(f"[{NAME}] action: {action}")
        if updated == original:
            continue
        changed = True
        before = sha256(original.encode("utf-8"))
        install(path, updated)
        installed = path.read_text(encoding="utf-8")
        if installed != updated:
            raise RuntimeError(f"post-write verification failed: {path}")
        after = sha256(installed.encode("utf-8"))
        print(f"[{NAME}] before SHA256: {before}")
        print(f"[{NAME}] installed SHA256: {after}")

    if changed:
        print(
            f"[{NAME}] applied: initialize KVBlockZeroer for fp8_ds_mla "
            "or any cache configuration requesting zeroing"
        )
    else:
        print(f"[{NAME}] already applied")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{NAME}] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
