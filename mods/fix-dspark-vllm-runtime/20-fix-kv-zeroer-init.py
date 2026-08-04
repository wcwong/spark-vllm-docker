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

UPSTREAM = '''        if kv_cache_config.needs_kv_cache_zeroing and hasattr(
            self.model_runner, "_init_kv_zero_meta"
        ):
            self.model_runner._init_kv_zero_meta()
'''

BROKEN_PREVIOUS_FIX = '''        if kv_cache_config.needs_kv_cache_zeroing:
            init_kv_zero_meta = getattr(
                self.model_runner, "_init_kv_zero_meta", None
            )
            if init_kv_zero_meta is None:
                raise RuntimeError(
                    "KV cache zeroing is required, but the active model runner "
                    "does not implement _init_kv_zero_meta()."
                )
            init_kv_zero_meta()
'''

FIXED = '''        needs_kv_zeroer = (
            kv_cache_config.needs_kv_cache_zeroing
            or self.cache_config.cache_dtype == "fp8_ds_mla"
        )
        if needs_kv_zeroer:
            init_kv_zero_meta = getattr(
                self.model_runner, "_init_kv_zero_meta", None
            )
            if init_kv_zero_meta is None:
                raise RuntimeError(
                    "KV cache zeroing is required, but the active model runner "
                    "does not implement _init_kv_zero_meta()."
                )
            init_kv_zero_meta()
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


def classify_source(text: str, path: Path) -> str:
    counts = {
        "upstream": text.count(UPSTREAM),
        "broken_previous_fix": text.count(BROKEN_PREVIOUS_FIX),
        "fixed": text.count(FIXED),
    }
    present = [name for name, count in counts.items() if count == 1]
    invalid = {name: count for name, count in counts.items() if count not in (0, 1)}
    if invalid or len(present) != 1:
        raise RuntimeError(
            f"unexpected source in {path}: counts={counts}; "
            "review this fix before using it"
        )
    return present[0]


def main() -> None:
    root = vllm_root()
    path = root / TARGET
    if not path.is_file():
        raise RuntimeError(f"target file not found: {path}")

    original = path.read_text(encoding="utf-8")
    state = classify_source(original, path)

    print(f"[{NAME}] vLLM root: {root}")
    print(f"[{NAME}] target: {path}")
    print(f"[{NAME}] detected source state: {state}")

    if state == "fixed":
        print(f"[{NAME}] already applied")
        return

    old = UPSTREAM if state == "upstream" else BROKEN_PREVIOUS_FIX
    updated = original.replace(old, FIXED, 1)
    compile(updated, str(path), "exec")
    if classify_source(updated, path) != "fixed":
        raise RuntimeError("post-patch semantic state verification failed")

    before = sha256(original.encode())
    install(path, updated)
    after = sha256(path.read_bytes())
    print(f"[{NAME}] before SHA256: {before}")
    print(f"[{NAME}] installed SHA256: {after}")
    print(
        f"[{NAME}] applied: initialize KV-zero metadata for fp8_ds_mla "
        "or any cache configuration that requests block zeroing"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{NAME}] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
