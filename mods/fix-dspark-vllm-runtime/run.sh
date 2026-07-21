#!/usr/bin/env bash
set -euo pipefail

PATCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

printf '%s\n' '=== Applying DGX Spark vLLM runtime fixes ==='
printf 'Patch directory: %s\n' "$PATCH_DIR"
printf 'Host: %s\n' "$(hostname)"
printf 'Python: %s\n' "$(python3 --version 2>&1)"
printf 'VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: %s\n' \
    "${VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD:-<unset>}"

patches=(
    "10-fix-deepgemm-warmup-scales.py"
    "20-fix-kv-zeroer-init.py"
)
#    "30-fix-packed-kv-zeroing.py"
#    "40-disable-dsv4-attention-multistream.py"


printf 'Patch sequence:'
for patch in "${patches[@]}"; do
    printf ' %s' "$patch"
done
printf '\n'

for patch in "${patches[@]}"; do
    path="$PATCH_DIR/$patch"
    if [[ ! -f "$path" ]]; then
        printf 'ERROR: required patch script not found: %s\n' "$path" >&2
        exit 1
    fi

    printf '\n--- Running %s ---\n' "$patch"
    python3 "$path"
done

printf '\n%s\n' '=== All DGX Spark vLLM runtime fixes completed successfully ==='
