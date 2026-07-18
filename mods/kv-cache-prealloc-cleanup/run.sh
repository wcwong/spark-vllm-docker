#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
TARGET="$PYTHON_ROOT/vllm/v1/worker/gpu_worker.py"
CACHE_CONFIG="$PYTHON_ROOT/vllm/config/cache.py"

if [ ! -f "$TARGET" ]; then
  echo "[kv-cache-prealloc-cleanup] vLLM gpu_worker.py not found at $TARGET" >&2
  exit 1
fi

if [ ! -f "$CACHE_CONFIG" ]; then
  echo "[kv-cache-prealloc-cleanup] vLLM cache.py not found at $CACHE_CONFIG" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[kv-cache-prealloc-cleanup] python3 is required to apply this mod." >&2
  exit 1
fi

python3 - "$TARGET" "$CACHE_CONFIG" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
lines = text.splitlines(keepends=True)
changed = False


def find_line(pattern: str) -> tuple[int, re.Match[str]]:
    regex = re.compile(pattern)
    for index, line in enumerate(lines):
        match = regex.match(line)
        if match:
            return index, match
    raise SystemExit(
        f"[kv-cache-prealloc-cleanup] Could not find expected pattern: {pattern}"
    )


profile_call = (
    r"^(?P<indent>[ \t]+)cudagraph_memory_estimate = "
    r"self\.model_runner\.profile_cudagraph_memory\(\)\n$"
)
index, match = find_line(profile_call)
indent = match.group("indent")
guard_line = f"{indent[:-4]}if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:\n"
graph_skip_supported = (
    "Skipping CUDA graph memory profiling" in text
    or guard_line in lines[max(0, index - 3):index]
)
if not graph_skip_supported:
    lines[index : index + 1] = [
        f"{indent}# spark-vllm-docker: skip CUDA graph memory profiling when disabled\n",
        f"{indent}if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:\n",
        f"{indent}    cudagraph_memory_estimate = "
        "self.model_runner.profile_cudagraph_memory()\n",
        f"{indent}else:\n",
        f"{indent}    logger.info_once(\n",
        f'{indent}        "Skipping CUDA graph memory profiling because "\n',
        f'{indent}        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0."\n',
        f"{indent}    )\n",
    ]
    changed = True

if changed:
    path.write_text("".join(lines))
    print("[kv-cache-prealloc-cleanup] Made CUDA graph profiling skip env-respectful.")
else:
    print("[kv-cache-prealloc-cleanup] CUDA graph profiling skip is already supported; skipping.")

cache_path = Path(sys.argv[2])
cache_text = cache_path.read_text()
cache_lines = cache_text.splitlines(keepends=True)
cache_changed = False

if "Cannot specify both gpu_memory_utilization_gb" in cache_text:
    validator_pattern = re.compile(
        r"^(?P<indent>[ \t]+)def _validate_memory_params"
        r"\(self\) -> \"CacheConfig\":\n$"
    )
    for index, line in enumerate(cache_lines):
        match = validator_pattern.match(line)
        if match:
            body_indent = match.group("indent") + "    "
            cache_lines[index + 1:index + 1] = [
                f"{body_indent}# spark-vllm-docker: allow fixed GiB reservation with manual KV cache\n",
                f"{body_indent}return self\n",
                "\n",
            ]
            cache_changed = True
            break
    else:
        raise SystemExit(
            "[kv-cache-prealloc-cleanup] Found the memory-parameter "
            "conflict validator in cache.py, but could not patch it."
        )

if cache_changed:
    cache_path.write_text("".join(cache_lines))
    print(
        "[kv-cache-prealloc-cleanup] Allowed --gpu-memory-utilization-gb "
        "with --kv-cache-memory-bytes."
    )
else:
    print(
        "[kv-cache-prealloc-cleanup] Manual KV cache compatibility is "
        "already supported; skipping."
    )
PY

echo "=====> vLLM can skip CUDA graph profiling when VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0"
echo "=====> vLLM can combine --gpu-memory-utilization-gb with --kv-cache-memory-bytes"
