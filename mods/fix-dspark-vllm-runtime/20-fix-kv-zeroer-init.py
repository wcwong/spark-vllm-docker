#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile

NAME = "kv-zeroer-init"
TARGET = Path("v1/worker/gpu_worker.py")

OLD = """        if kv_cache_config.needs_kv_cache_zeroing and hasattr(\n            self.model_runner, \"_init_kv_zero_meta\"\n        ):\n            self.model_runner._init_kv_zero_meta()\n"""

NEW = """        if kv_cache_config.needs_kv_cache_zeroing:\n            init_kv_zero_meta = getattr(\n                self.model_runner, \"_init_kv_zero_meta\", None\n            )\n            if init_kv_zero_meta is None:\n                raise RuntimeError(\n                    \"KV cache zeroing is required, but the active model runner \"\n                    \"does not implement _init_kv_zero_meta().\"\n                )\n            init_kv_zero_meta()\n"""


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


def main() -> None:
    root = vllm_root()
    path = root / TARGET
    if not path.is_file():
        raise RuntimeError(f"target file not found: {path}")

    original = path.read_text(encoding="utf-8")
    old_count = original.count(OLD)
    new_count = original.count(NEW)

    print(f"[{NAME}] vLLM root: {root}")
    print(f"[{NAME}] target: {path}")

    if old_count == 0 and new_count == 1:
        print(f"[{NAME}] already applied")
        return
    if old_count != 1 or new_count != 0:
        raise RuntimeError(
            f"unexpected source in {path}: old_count={old_count}, "
            f"new_count={new_count}; review this fix before using it"
        )

    updated = original.replace(OLD, NEW, 1)
    compile(updated, str(path), "exec")

    before = sha256(original.encode())
    install(path, updated)
    after = sha256(path.read_bytes())
    print(f"[{NAME}] before SHA256: {before}")
    print(f"[{NAME}] installed SHA256: {after}")
    print(f"[{NAME}] applied: require KV-zero metadata initialization when zeroing is needed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{NAME}] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
