# Qwen-Image Runner Fragmentation E2E

This run uses real `DiffusionModelRunner.execute_stepwise()` with
`cache_backend=cache_dit` and original contiguous CUDA tensors. It does not use
the paged cache path.

Dataset: Qwen-Image Dataset C.

Result:

- oom_observed: `False`
- max_frag_ratio: `0.8693`
- mean_frag_ratio: `0.8443`
- max_resident_cache_mib: `150.38`
- finished_requests: `0`
- released_requests: `2`
- pending_requests: `0`
- error: ``

Files:

- `trace.json`: generated random arrivals and request profiles
- `requests.csv`: request lifecycle summary
- `timeline.csv`: allocator/cache snapshot after each runner event
- `summary.json`: aggregate metrics
- `charts/fragmentation_ratio.svg`
- `charts/inactive_free_blocks.svg`
- `charts/resident_cache.svg`
