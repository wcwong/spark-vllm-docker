#!/usr/bin/env bash

set -euo pipefail

PATCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PATCHES=(
  "10-fix-deepgemm-warmup-scales.py"
  "20-fix-kv-zeroer-init.py"
  "30-fix-packed-kv-zeroing.py"
  "50-clear-flashinfer-cache.py"
)

echo "=== Applying DGX Spark vLLM runtime fixes ==="
echo "Patch directory: ${PATCH_DIR}"
echo "Host: $(hostname)"
echo "Python: $(python3 --version)"
echo "Patch sequence: ${PATCHES[*]}"
echo

for patch in "${PATCHES[@]}"; do
  patch_path="${PATCH_DIR}/${patch}"

  if [[ ! -f "${patch_path}" ]]; then
    echo "ERROR: patch script not found: ${patch_path}" >&2
    exit 1
  fi

  echo "--- Running ${patch} ---"
  python3 "${patch_path}"
  echo
done

echo "=== All DGX Spark vLLM runtime fixes completed successfully ==="
