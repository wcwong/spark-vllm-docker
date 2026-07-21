#!/usr/bin/env python3
"""Disable DeepSeek-V4 attention multistream overlap in the installed vLLM.

Reviewed against vLLM commit df13b5a. This is a diagnostic isolation patch
for execution through eugr's run-recipe mod sequence. It disables all three
DeepSeek-V4 attention overlap sites while retaining the existing serial
fallbacks:

1. Input projection GEMM overlap.
2. Outer wq_b/KV-insert, indexer, and MLA-compressor overlap.
3. Nested indexer query-projection and compressor overlap.

The patch also inserts a one-time runtime warning in each model worker process
so captured launch logs prove that the patched implementation was imported.
It deliberately adds no per-request logging and no CUDA synchronization.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import NoReturn

PATCH_NAME = "dsv4-attention-multistream"
REVIEWED_COMMIT = "df13b5a"
RELATIVE_TARGET = Path("models/deepseek_v4/attention.py")
MARKER = "BEGIN DSV4 ATTENTION MULTISTREAM ISOLATION"

OLD_BLOCK = """        # Will be None on ROCm for now.\n        self.aux_stream_list = aux_stream_list\n"""

NEW_BLOCK = """        # BEGIN DSV4 ATTENTION MULTISTREAM ISOLATION (df13b5a)\n        # Diagnostic A/B: force every existing multistream helper in this\n        # attention layer to take its serial fallback. The indexer copied\n        # aux_stream_list[2] during construction, so both references must be\n        # cleared. Do not replace this with only self.aux_stream_list = None.\n        if self.indexer is not None:\n            self.indexer.aux_stream = None\n        self.aux_stream_list = None\n\n        # Fail immediately during model construction if a later refactor makes\n        # either overlap path active despite this isolation patch.\n        assert self.aux_stream_list is None\n        assert self.indexer is None or self.indexer.aux_stream is None\n\n        # One line per worker process, not per layer/request. This proves the\n        # installed patched source was imported without perturbing CUDA timing.\n        logger.warning_once(\n            \"DSV4 attention multistream isolation active: input-GEMM, \"\n            \"outer indexer/compressor, and nested indexer/compressor \"\n            \"overlaps are disabled (reviewed at df13b5a).\"\n        )\n        # END DSV4 ATTENTION MULTISTREAM ISOLATION (df13b5a)\n"""

# Structural signatures from df13b5a. These are deliberately checked in
# addition to the replacement anchor so the script fails closed if the source
# has materially diverged while coincidentally retaining the assignment.
REQUIRED_UNPATCHED_SIGNATURES = (
    "indexer_aux_stream = (",
    "aux_stream_list[2] if aux_stream_list is not None else None",
    "enable=hidden_states.shape[0]",
    "enable=aux_streams is not None",
    "self.aux_stream,",
)

REQUIRED_PATCHED_SIGNATURES = (
    MARKER,
    "self.indexer.aux_stream = None",
    "self.aux_stream_list = None",
    "DSV4 attention multistream isolation active",
)


def log(message: str) -> None:
    print(f"[{PATCH_NAME}] {message}", flush=True)


def fail(message: str) -> NoReturn:
    print(f"[{PATCH_NAME}] ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def locate_vllm_root() -> Path:
    override = os.environ.get("VLLM_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if (root / RELATIVE_TARGET).is_file():
            return root
        fail(
            f"VLLM_ROOT={root} does not contain {RELATIVE_TARGET}"
        )

    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, AttributeError, ValueError) as exc:
        fail(f"could not locate installed vllm package: {exc}")

    if spec is None or not spec.submodule_search_locations:
        fail("could not locate installed vllm package")

    roots = [Path(item).resolve() for item in spec.submodule_search_locations]
    matches = [root for root in roots if (root / RELATIVE_TARGET).is_file()]
    if len(matches) != 1:
        rendered = ", ".join(str(root) for root in roots) or "<none>"
        fail(
            "expected exactly one installed vllm package containing "
            f"{RELATIVE_TARGET}; candidates: {rendered}"
        )
    return matches[0]


def validate_unpatched_source(text: str, target: Path) -> None:
    anchor_count = text.count(OLD_BLOCK)
    if anchor_count != 1:
        fail(
            f"expected exactly one df13b5a assignment anchor in {target}; "
            f"found {anchor_count}"
        )

    missing = [item for item in REQUIRED_UNPATCHED_SIGNATURES if item not in text]
    if missing:
        fail(
            "source differs from the reviewed df13b5a multistream layout; "
            "missing signature(s): " + ", ".join(repr(item) for item in missing)
        )


def validate_patched_source(text: str, target: Path) -> None:
    missing = [item for item in REQUIRED_PATCHED_SIGNATURES if item not in text]
    if missing:
        fail(
            f"post-patch validation failed for {target}; missing: "
            + ", ".join(repr(item) for item in missing)
        )

    if text.count(MARKER) != 1:
        fail(
            f"post-patch validation failed for {target}; marker count is "
            f"{text.count(MARKER)}, expected 1"
        )

    if text.count("self.indexer.aux_stream = None") != 1:
        fail("post-patch validation failed: nested stream disable is not unique")

    try:
        ast.parse(text, filename=str(target))
    except SyntaxError as exc:
        fail(f"patched Python source is invalid: {exc}")


def atomic_write(path: Path, text: str) -> None:
    original_mode = path.stat().st_mode
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, original_mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def print_patched_excerpt(text: str) -> None:
    lines = text.splitlines()
    marker_index = next(
        (index for index, line in enumerate(lines) if MARKER in line), None
    )
    if marker_index is None:
        return

    start = max(0, marker_index - 1)
    end = min(len(lines), marker_index + 24)
    log("installed source excerpt:")
    for index in range(start, end):
        print(
            f"[{PATCH_NAME}] {index + 1:5d}: {lines[index]}",
            flush=True,
        )


def main() -> int:
    log(f"reviewed vLLM commit: {REVIEWED_COMMIT}")
    log(f"host: {os.uname().nodename}")
    log(f"python: {sys.executable}")
    log(
        "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD="
        + os.environ.get("VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD", "<unset>")
    )

    vllm_root = locate_vllm_root()
    target = vllm_root / RELATIVE_TARGET
    log(f"vLLM root: {vllm_root}")
    log(f"target: {target}")

    original_bytes = target.read_bytes()
    original_text = original_bytes.decode("utf-8")
    before_hash = sha256_bytes(original_bytes)
    log(f"current SHA256: {before_hash}")

    if MARKER in original_text:
        validate_patched_source(original_text, target)
        log("patch already applied and validated")
        log("input projection GEMM overlap: disabled")
        log("outer indexer/compressor overlap: disabled")
        log("nested indexer compressor overlap: disabled")
        print_patched_excerpt(original_text)
        return 0

    validate_unpatched_source(original_text, target)
    log("validated df13b5a multistream structure")

    patched_text = original_text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    validate_patched_source(patched_text, target)

    backup = target.with_name(target.name + ".orig")
    if backup.exists():
        log(f"backup already exists; preserving: {backup}")
    else:
        shutil.copy2(target, backup)
        log(f"created backup: {backup}")

    atomic_write(target, patched_text)

    installed_bytes = target.read_bytes()
    installed_text = installed_bytes.decode("utf-8")
    validate_patched_source(installed_text, target)
    after_hash = sha256_bytes(installed_bytes)

    if after_hash == before_hash:
        fail("source hash did not change after applying patch")

    log("disabling DSV4 CUDA-stream overlap")
    log("input projection GEMM overlap: disabled")
    log("outer indexer/compressor overlap: disabled")
    log("nested indexer compressor overlap: disabled")
    log("runtime worker confirmation log: enabled once per process")
    log("per-request logging: disabled")
    log("CUDA synchronization instrumentation: disabled")
    log(f"before SHA256: {before_hash}")
    log(f"installed SHA256: {after_hash}")
    print_patched_excerpt(installed_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
