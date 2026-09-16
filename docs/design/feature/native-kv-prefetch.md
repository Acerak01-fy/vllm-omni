# HunyuanImage3 native KV prefetch (GPU)

The DiT scheduler can receive the next waiting request's AR KV while the
current request computes. This uses the native Mooncake connector and
Scheduler-owned paged KV allocations.

## Enable

In the DiT stage's existing native transfer configuration:

```yaml
max_num_seqs: 1
diffusion_kv_mode: paged_scheduler
kv_transfer_config:
  kv_connector: MooncakeConnector
  kv_role: kv_consumer
  engine_id: hunyuan-image3-dit
  kv_connector_extra_config:
    mooncake_protocol: tcp
    enable_kv_async_prefetch: true
    transfer_timeout: 60.0
```

The flag defaults to false. The initial implementation accepts CUDA,
HunyuanImage3Pipeline, Mooncake TCP, and one running request per DiT replica.
CFG companion collectors are excluded. Both producer and consumer must use
the same transport. The legacy `omni_kv_config` prefetch flag has no effect
on this path.

## Execution

```text
DiT schedule(A)
  ├─ reserve A pages
  ├─ reserve all CFG rows of waiting B, if capacity permits
  └─ submit={A0,A1,B0,B1}, required={A0,A1}
       ↓
DiT Worker: submit A; wait for A on every rank; then submit B separately
       ↓
compute A  ║  AR Mooncake writes B into its reserved DiT pages
       ↓
poll completion → intersect cumulative results from all Worker ranks
       ↓
DiT schedule(B): reuse B's reservation; wait only if still loading
```

`kv_transfer_request_ids` identifies new submissions.
`kv_required_request_ids` identifies the loads that must finish before the
current computation; `None` retains the previous synchronous behavior.
`KVReceiveProgress` retains events consumed by `post_forward()` across RPCs.
The completion sets are removed when Scheduler sends the internal sequence
IDs in `kv_finished_request_ids`.

A and B use separate connector metadata and RPC submissions. Mooncake can
coalesce ready requests from one producer into a single write; submitting
A+B together would make A wait for B's bytes as well. The Executor completes
the current request's receive on all ranks before launching B without waiting.

Only one waiting request owns the prefetch slot. It stays in WAITING and
its attention rows are installed only at normal admission. The slot becomes
available when that request is admitted or safely cancelled. If all CFG rows
cannot be reserved, the atomic allocation rolls back and normal admission
handles the request later.

Cancellation waits for the already submitted transfer before freeing pages.
Shutdown drains outstanding receives and stops Workers before releasing
Scheduler allocations. A receive error or timeout fails the engine without
resubmitting the transfer. Timeouts count from the original submission and
are checked when the Worker next polls. Sleep is rejected while native KV
receive records are live; finish requests and process their cleanup first.

Prefetch changes transfer timing, not the number of blocks sent per request.
It requires a request with complete source metadata to already be waiting
when the scheduler chooses a computation. A serial client may never trigger
prefetch. The DiT KV pool must fit the current and prefetched requests
together; compare off/on with the same pool size.

## Validation

CPU tests in `tests/diffusion/diffusion_kv/test_native_prefetch.py` cover
out-of-order completion, all-rank readiness, duplicate submissions, timeout,
invalid blocks, atomic CFG allocation, reservation reuse and cancellation.
Run with the repository's normal pytest environment.

GPU validation should run the native paged attention tests and concurrent
HunyuanImage3 requests with the flag off/on. Verify actual prefetch submissions,
successful images, no duplicate internal sequence submissions, and KV writes
overlapping another request's DiT execution. Use identical prompts, seeds,
pool sizes and transport settings for both runs.
