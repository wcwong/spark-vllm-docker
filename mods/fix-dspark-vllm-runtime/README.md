# DGX Spark vLLM runtime fixes

`run-recipe.sh` copies this directory into the running container and invokes
`run.sh`. The three Python scripts modify the installed vLLM package in place.

Files:

- `10-fix-deepgemm-warmup-scales.py`
- `20-fix-kv-zeroer-init.py`
- `30-fix-packed-kv-zeroing.py`
- `run.sh`

Each script uses exact source replacements and exits without changing its target
when the installed source no longer matches the reviewed form. Existing `.orig`
backups are preserved. No `.patch` files or `git` command are used.
