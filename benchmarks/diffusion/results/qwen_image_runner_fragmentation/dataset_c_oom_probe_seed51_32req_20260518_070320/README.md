# Qwen-Image Runner Fragmentation E2E

This run uses real `DiffusionModelRunner.execute_stepwise()` with
`cache_backend=cache_dit` and original contiguous CUDA tensors. It does not use
the paged cache path.

Dataset: Qwen-Image Dataset C.

Result:

- oom_observed: `True`
- max_frag_ratio: `0.9321`
- mean_frag_ratio: `0.8736`
- max_resident_cache_mib: `2571.69`
- finished_requests: `0`
- released_requests: `32`
- pending_requests: `0`
- error: `CUDA out of memory. Tried to allocate 3.00 GiB. GPU 0 has a total capacity of 79.14 GiB of which 2.09 GiB is free. Including non-PyTorch memory, this process has 77.04 GiB memory in use. Of the allocated memory 75.40 GiB is allocated by PyTorch, and 1.15 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)`

Files:

- `trace.json`: generated random arrivals and request profiles
- `requests.csv`: request lifecycle summary
- `timeline.csv`: allocator/cache snapshot after each runner event
- `summary.json`: aggregate metrics
- `charts/fragmentation_ratio.svg`
- `charts/inactive_free_blocks.svg`
- `charts/resident_cache.svg`
