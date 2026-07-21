from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import shutil
import stat
import tempfile
from dataclasses import dataclass
from typing import Callable


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def locate_vllm_root() -> pathlib.Path:
    override = os.environ.get("VLLM_ROOT")
    if override:
        root = pathlib.Path(override).resolve()
        if not root.is_dir():
            raise SystemExit(f"VLLM_ROOT is not a directory: {root}")
        return root

    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Unable to locate the installed vllm package")
    return pathlib.Path(next(iter(spec.submodule_search_locations))).resolve()


def line_starts(source: bytes) -> list[int]:
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def node_byte_span(node, source: bytes) -> tuple[int, int]:
    attrs = ("lineno", "col_offset", "end_lineno", "end_col_offset")
    if not all(hasattr(node, attr) for attr in attrs):
        raise RuntimeError("AST node does not include source positions")
    starts = line_starts(source)
    return (
        starts[node.lineno - 1] + node.col_offset,
        starts[node.end_lineno - 1] + node.end_col_offset,
    )


def full_line_start(node, source: bytes) -> int:
    return line_starts(source)[node.lineno - 1]


def full_line_end(node, source: bytes) -> int:
    starts = line_starts(source)
    return starts[node.end_lineno]


def apply_replacements(
    source: bytes,
    replacements: list[tuple[int, int, bytes]],
) -> bytes:
    previous_start = len(source) + 1
    output = bytearray(source)
    for start, end, replacement in sorted(replacements, reverse=True):
        if not (0 <= start <= end <= len(source)):
            raise RuntimeError(f"Invalid replacement span: {(start, end)}")
        if end > previous_start:
            raise RuntimeError("Replacement spans overlap")
        output[start:end] = replacement
        previous_start = start
    return bytes(output)


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp = pathlib.Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp, stat.S_IMODE(path.stat().st_mode))
        os.replace(temp, path)
        directory_fd = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)


def remove_bytecode(path: pathlib.Path) -> None:
    pycache = path.parent / "__pycache__"
    if pycache.is_dir():
        for pyc in pycache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink()


@dataclass(frozen=True)
class PatchPlan:
    target: pathlib.Path
    original: bytes
    patched: bytes
    description: str

    @property
    def changed(self) -> bool:
        return self.original != self.patched


def make_plan(
    target: pathlib.Path,
    patcher: Callable[[bytes, str], tuple[bytes, str]],
) -> PatchPlan:
    target = target.resolve(strict=True)
    if not target.is_file():
        raise RuntimeError(f"Target is not a regular file: {target}")
    original = target.read_bytes()
    patched, description = patcher(original, str(target))
    return PatchPlan(target, original, patched, description)


def install_plans(fix_name: str, plans: list[PatchPlan]) -> None:
    print(f"[{fix_name}] validating {len(plans)} target file(s)", flush=True)
    changed = [plan for plan in plans if plan.changed]

    if not changed:
        for plan in plans:
            print(
                f"[{fix_name}] no change: {plan.target} "
                f"({plan.description})",
                flush=True,
            )
            print(f"[{fix_name}] SHA256: {sha256(plan.original)}", flush=True)
        return

    for plan in changed:
        backup = plan.target.with_name(f"{plan.target.name}.orig")
        if not backup.exists():
            shutil.copy2(plan.target, backup)
            print(f"[{fix_name}] created backup: {backup}", flush=True)
        else:
            print(f"[{fix_name}] backup already exists: {backup}", flush=True)

    written: list[PatchPlan] = []
    try:
        for plan in changed:
            print(f"[{fix_name}] applying: {plan.description}", flush=True)
            print(f"[{fix_name}] target: {plan.target}", flush=True)
            print(f"[{fix_name}] before SHA256: {sha256(plan.original)}", flush=True)
            atomic_write(plan.target, plan.patched)
            written.append(plan)
    except Exception as exc:
        for plan in reversed(written):
            try:
                atomic_write(plan.target, plan.original)
                print(f"[{fix_name}] rolled back: {plan.target}", flush=True)
            except Exception as rollback_exc:
                print(
                    f"[{fix_name}] WARNING: rollback failed for "
                    f"{plan.target}: {rollback_exc}",
                    flush=True,
                )
        raise SystemExit(
            f"[{fix_name}] patch write failed; rollback attempted: {exc}"
        ) from exc

    for plan in plans:
        remove_bytecode(plan.target)
        installed = plan.target.read_bytes()
        if installed != plan.patched:
            raise SystemExit(
                f"[{fix_name}] post-write verification failed: {plan.target}"
            )
        status = "applied" if plan.changed else "already present"
        print(f"[{fix_name}] {status}: {plan.description}", flush=True)
        print(f"[{fix_name}] installed SHA256: {sha256(installed)}", flush=True)
