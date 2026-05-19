# Qwen-Image Runner Fragmentation E2E

This run uses real `DiffusionModelRunner.execute_stepwise()` with
`cache_backend=cache_dit` and original contiguous CUDA tensors. It does not use
the paged cache path.

Dataset: Qwen-Image Dataset C.

Result:

- oom_observed: `False`
- oom_probe_attempted: `True`
- oom_probe_oom_observed: `False`
- oom_probe_allocation_mib: `1792.00`
- oom_probe_pressure_allocated_mib: `21168.75`
- max_frag_ratio: `0.9321`
- mean_frag_ratio: `0.8736`
- max_resident_cache_mib: `2571.69`
- finished_requests: `0`
- released_requests: `32`
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
