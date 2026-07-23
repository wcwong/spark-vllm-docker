#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

scripts=(
  "10-fix-deepgemm-warmup-scales.py"
  "20-fix-kv-zeroer-init.py"
  "30-fix-packed-kv-zeroing.py"
)

printf '%s\n' "=== Applying DGX Spark vLLM runtime fixes ==="
printf 'Patch directory: %s\n' "$script_dir"
printf 'Host: %s\n' "$(hostname)"
printf 'Python: %s\n' "$(python3 --version 2>&1)"
printf 'Patch sequence:'
printf ' %s' "${scripts[@]}"
printf '\n'

for script in "${scripts[@]}"; do
  path="$script_dir/$script"
  if [[ ! -f "$path" ]]; then
    printf 'ERROR: missing patch script: %s\n' "$path" >&2
    exit 1
  fi
  printf '\n--- Running %s ---\n' "$script"
  python3 "$path"
done

printf '\n%s\n' "=== All DGX Spark vLLM runtime fixes completed successfully ==="
