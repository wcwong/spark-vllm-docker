# DGX Spark vLLM runtime fixes

`run.sh` applies each patch independently and in a fixed order:

1. `10-fix-deepgemm-warmup-scales.py` — initializes synthetic DeepGEMM warmup scales with `torch.ones` rather than uninitialized `torch.empty` memory.
2. `20-fix-kv-zeroer-init.py` — initializes V2 KV-zeroing metadata for `fp8_ds_mla`.
3. `30-fix-packed-kv-zeroing.py` — routes packed KV layouts through storage-aware physical-block zeroing rather than zeroing from offset view pointers.

Every patch is:

- source-structure aware and refuses unexpected code;
- idempotent;
- atomically written with `.orig` backups;
- post-write verified;
- verbose about the fix, target file, and before/after SHA256 values.

The package normally locates the installed `vllm` module automatically. For fixture testing, set `VLLM_ROOT` to the package directory that contains `model_executor/` and `v1/`.
