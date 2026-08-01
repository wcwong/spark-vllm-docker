#!/usr/bin/env python3

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


PREFIX = "[clear-flashinfer-cache]"
JIT_CACHE_DISTRIBUTION = "flashinfer-jit-cache"

BUILD_METADATA_PATHS = (
    Path("/workspace/build-metadata.yaml"),
    Path("/etc/spark-vllm/build-metadata.yaml"),
)

OBSOLETE_SYMBOL = b"TVMFFIGetCustomAllocator"


def log(message: str) -> None:
    print(f"{PREFIX} {message}", flush=True)


def package_installed(name: str) -> bool:
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return False

    return True


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()

    log(f"removed: {path}")


def uninstall_packaged_jit_cache() -> None:
    if not package_installed(JIT_CACHE_DISTRIBUTION):
        log(f"{JIT_CACHE_DISTRIBUTION} is not installed")
        return

    version = importlib.metadata.version(
        JIT_CACHE_DISTRIBUTION
    )

    log(
        f"uninstalling {JIT_CACHE_DISTRIBUTION} "
        f"version {version}"
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "--yes",
            JIT_CACHE_DISTRIBUTION,
        ],
        check=True,
    )

    importlib.invalidate_caches()

    if package_installed(JIT_CACHE_DISTRIBUTION):
        raise RuntimeError(
            f"{JIT_CACHE_DISTRIBUTION} remains installed"
        )

    log(f"uninstalled: {JIT_CACHE_DISTRIBUTION}")


def module_origin(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)

    if spec is None or spec.origin is None:
        raise RuntimeError(
            f"cannot locate installed module {module_name}"
        )

    return Path(spec.origin).resolve()


def distribution_metadata_path(
    distribution_name: str,
) -> Path | None:
    try:
        distribution = importlib.metadata.distribution(
            distribution_name
        )
    except importlib.metadata.PackageNotFoundError:
        return None

    for entry in distribution.files or ():
        entry_text = str(entry)

        if (
            ".dist-info/" in entry_text
            and entry_text.endswith("/METADATA")
        ):
            path = Path(
                distribution.locate_file(entry)
            ).resolve()

            if path.is_file():
                return path

    return None


def determine_build_stamp() -> tuple[int, Path]:
    """
    Return the best available image/package timestamp.

    Prefer build metadata generated with the image. Fall back to the
    installed FlashInfer and TVM-FFI package files.
    """
    for path in BUILD_METADATA_PATHS:
        if path.is_file():
            return path.stat().st_mtime_ns, path

    candidates: list[Path] = [
        module_origin("flashinfer"),
        module_origin("tvm_ffi"),
    ]

    for distribution_name in (
        "flashinfer-python",
        "apache-tvm-ffi",
    ):
        metadata = distribution_metadata_path(
            distribution_name
        )

        if metadata is not None:
            candidates.append(metadata)

    newest = max(
        candidates,
        key=lambda path: path.stat().st_mtime_ns,
    )

    return newest.stat().st_mtime_ns, newest


def binary_contains(
    path: Path,
    needle: bytes,
) -> bool:
    overlap = max(0, len(needle) - 1)
    previous = b""

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                return False

            data = previous + chunk

            if needle in data:
                return True

            previous = data[-overlap:] if overlap else b""


def newest_mtime_ns(path: Path) -> int:
    newest = path.stat().st_mtime_ns

    for child in path.rglob("*"):
        try:
            child_mtime = child.stat().st_mtime_ns
        except FileNotFoundError:
            continue

        newest = max(newest, child_mtime)

    return newest


def shared_objects(module_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in module_dir.rglob("*.so")
        if path.is_file()
    )


def validate_module_dir(
    module_dir: Path,
    build_stamp_ns: int,
) -> tuple[bool, str]:
    """
    Keep cache entries created after the image/package build.

    Complete modules are also checked for the known incompatible
    TVM-FFI symbol.
    """
    module_stamp_ns = newest_mtime_ns(module_dir)

    if module_stamp_ns < build_stamp_ns:
        return (
            False,
            "cache predates installed image/package build",
        )

    objects = shared_objects(module_dir)

    if not objects:
        return (
            True,
            "recent incomplete build preserved for Ninja resume",
        )

    for shared_object in objects:
        if binary_contains(
            shared_object,
            OBSOLETE_SYMBOL,
        ):
            return (
                False,
                f"{shared_object.name} references "
                "TVMFFIGetCustomAllocator",
            )

    return (
        True,
        f"{len(objects)} recent shared object(s) preserved",
    )


def inspect_runtime_cache(
    jit_dir: Path,
    build_stamp_ns: int,
) -> None:
    jit_dir.mkdir(parents=True, exist_ok=True)

    module_dirs = sorted(
        path
        for path in jit_dir.iterdir()
        if path.is_dir()
    )

    if not module_dirs:
        log("runtime JIT cache is empty")
        return

    kept = 0
    removed = 0

    for module_dir in module_dirs:
        keep, reason = validate_module_dir(
            module_dir,
            build_stamp_ns,
        )

        if keep:
            kept += 1
            log(
                f"keeping cache {module_dir.name}: "
                f"{reason}"
            )
        else:
            removed += 1
            log(
                f"removing cache {module_dir.name}: "
                f"{reason}"
            )
            remove_path(module_dir)

    log(
        f"runtime cache summary: "
        f"kept={kept}, removed={removed}"
    )


def validate_final_state() -> None:
    if package_installed(JIT_CACHE_DISTRIBUTION):
        raise RuntimeError(
            f"{JIT_CACHE_DISTRIBUTION} remains installed"
        )

    if importlib.util.find_spec(
        "flashinfer_jit_cache"
    ) is not None:
        raise RuntimeError(
            "flashinfer_jit_cache remains importable"
        )

    import flashinfer

    log(
        "FlashInfer Python package remains available: "
        f"{Path(flashinfer.__file__).resolve()}"
    )


def main() -> int:
    log(
        "removing packaged FlashInfer cache and "
        "checking runtime cache timestamps"
    )

    uninstall_packaged_jit_cache()

    # Ensure subsequent imports do not retain references to the
    # uninstalled packaged cache.
    for module_name in tuple(sys.modules):
        if (
            module_name == "flashinfer"
            or module_name.startswith("flashinfer.")
            or module_name == "flashinfer_jit_cache"
            or module_name.startswith(
                "flashinfer_jit_cache."
            )
        ):
            sys.modules.pop(module_name, None)

    importlib.invalidate_caches()

    from flashinfer.jit import env as jit_env

    jit_dir = Path(
        jit_env.FLASHINFER_JIT_DIR
    ).resolve()

    build_stamp_ns, build_stamp_source = (
        determine_build_stamp()
    )

    log(
        "image/package stamp source: "
        f"{build_stamp_source}"
    )
    log(f"runtime JIT directory: {jit_dir}")

    inspect_runtime_cache(
        jit_dir,
        build_stamp_ns,
    )

    validate_final_state()

    log("completed successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
