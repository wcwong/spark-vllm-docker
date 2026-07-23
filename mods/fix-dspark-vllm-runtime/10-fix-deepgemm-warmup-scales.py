#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile

NAME = "deepgemm-warmup-scales"
TARGET = Path("model_executor/warmup/deep_gemm_warmup.py")

REPLACEMENTS = [
    (
        """    a1q = torch.empty((max_tokens, k), device=device, dtype=torch.float8_e4m3fn)\n    a1q_scales = torch.empty(\n        (max_tokens, k // block_m), device=device, dtype=torch.float32\n    )\n""",
        """    a1q = torch.empty((max_tokens, k), device=device, dtype=torch.float8_e4m3fn)\n    a1q_scales = torch.ones(\n        (max_tokens, k // block_m), device=device, dtype=torch.float32\n    )\n""",
    ),
    (
        """        a1q = torch.empty((MAX_M, k), device=device, dtype=torch.float8_e4m3fn)\n        a1q_scales = torch.empty(\n            (MAX_M, k // block_m), device=device, dtype=torch.float32\n        )\n""",
        """        a1q = torch.empty((MAX_M, k), device=device, dtype=torch.float8_e4m3fn)\n        a1q_scales = torch.ones(\n            (MAX_M, k // block_m), device=device, dtype=torch.float32\n        )\n""",
    ),
]


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
    states = []
    for old, new in REPLACEMENTS:
        old_count = original.count(old)
        new_count = original.count(new)
        if old_count == 1 and new_count == 0:
            states.append("old")
        elif old_count == 0 and new_count == 1:
            states.append("new")
        else:
            raise RuntimeError(
                f"unexpected source in {path}: old_count={old_count}, "
                f"new_count={new_count}; review this fix before using it"
            )

    print(f"[{NAME}] vLLM root: {root}")
    print(f"[{NAME}] target: {path}")

    if all(state == "new" for state in states):
        print(f"[{NAME}] already applied")
        return
    if not all(state == "old" for state in states):
        raise RuntimeError(f"partially applied fix detected in {path}")

    updated = original
    for old, new in REPLACEMENTS:
        updated = updated.replace(old, new, 1)
    compile(updated, str(path), "exec")

    before = sha256(original.encode())
    install(path, updated)
    after = sha256(path.read_bytes())
    print(f"[{NAME}] before SHA256: {before}")
    print(f"[{NAME}] installed SHA256: {after}")
    print(f"[{NAME}] applied: initialize synthetic DeepGEMM scales with torch.ones")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[{NAME}] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
