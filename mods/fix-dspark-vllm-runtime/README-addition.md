## 40-disable-dsv4-attention-multistream.py

Diagnostic isolation patch reviewed against vLLM commit `df13b5a`.

It disables:

- input projection GEMM overlap;
- outer query/KV-insert, sparse-indexer, and MLA-compressor overlap;
- nested indexer query-projection and compressor overlap.

The mathematical operations and existing serial fallback order are unchanged.
The patch emits a one-time worker-process warning during model construction:

```text
DSV4 attention multistream isolation active: input-GEMM, outer indexer/compressor, and nested indexer/compressor overlaps are disabled (reviewed at df13b5a).
```

This warning proves each worker imported the patched source. The patch does not
add per-request logs, CUDA synchronization, or hot-path tracing because those
could hide the stream-ordering failure being tested.
