# vLLM Codebase Walkthrough for Agentic Workloads

This document is a practical reading guide for understanding the current vLLM
codebase, with emphasis on the pieces that matter for agentic workloads:
request scheduling, KV cache management, prefix reuse, and instrumentation.

The target reader already knows the high-level idea of LLM serving, but may not
remember how vLLM is organized or how the V0/V1 transition landed.

## Current State: V1 Is The Main Engine

In this checkout, V1 is not a side experiment. It is the main engine path:

- `vllm/engine/llm_engine.py` aliases `LLMEngine` to
  `vllm.v1.engine.llm_engine.LLMEngine`.
- `vllm/engine/async_llm_engine.py` aliases `AsyncLLMEngine` to
  `vllm.v1.engine.async_llm.AsyncLLM`.
- `docs/usage/v1_guide.md` says V0 has been fully deprecated.

The important nuance is that V1 did not replace every directory. V1 reworked
the scheduler, KV cache manager, worker/model-runner flow, sampler, and API
server path, but still reuses many stable components from the older codebase:
models, attention kernels, platform utilities, distributed helpers, and
entrypoint-level compatibility shims.

For code reading, treat `vllm/v1/` as the control-plane source of truth, and
treat `vllm/model_executor/`, `vllm/platforms/`, `vllm/distributed/`,
`vllm/v1/attention/`, and kernels as shared execution substrate.

## The Best Reading Order

Read in this order if your goal is to understand performance behavior rather
than every feature:

1. `docs/usage/v1_guide.md`

   Start here for the V0/V1 delta and feature status. The most important line
   for scheduling is that V1 uses a unified scheduler: prompt and output tokens
   are both scheduled as token work against a fixed per-step token budget.

2. `docs/design/arch_overview.md`

   Use this for process boundaries. In online serving, the API server handles
   HTTP/input processing, the engine core owns scheduling and KV metadata, and
   GPU workers execute model forward passes.

3. `vllm/v1/engine/core.py`

   This is the inner engine loop. It initializes workers, profiles available KV
   memory, builds the scheduler, and repeatedly schedules work and executes it.

4. `vllm/v1/core/sched/scheduler.py`

   This is the highest-value file for agentic workload analysis. It decides
   which requests run each step, how many tokens each request gets, when waiting
   requests are admitted, and when requests are preempted.

5. `vllm/v1/core/kv_cache_manager.py`

   This is the scheduler-facing KV manager. It hides the details of multiple KV
   cache groups and exposes operations like prefix-cache lookup, slot
   allocation, free, and usage.

6. `vllm/v1/core/block_pool.py` and
   `vllm/v1/core/single_type_kv_cache_manager.py`

   These files explain the actual block lifecycle: free queue, ref counts,
   prefix-cache hashes, LRU-like eviction order, sliding-window block skipping,
   and per-attention-type allocation behavior.

7. `vllm/v1/worker/gpu_model_runner.py`

   Read this after the scheduler/KV metadata path. The model runner allocates
   physical KV tensors, reshapes them to backend-required layouts, binds them
   into attention layers, and constructs model inputs from scheduler outputs.

## Directory Map

The useful mental map is:

- `vllm/entrypoints/`: CLI, OpenAI-compatible server, offline APIs, request
  parsing, response streaming.
- `vllm/engine/`: public compatibility layer. In this checkout, core engine
  names point to V1 implementations.
- `vllm/v1/engine/`: engine core, client transport, output processing, DP
  coordination, request lifecycle.
- `vllm/v1/core/`: scheduler, KV cache manager, block pool, encoder cache.
- `vllm/v1/worker/`: GPU/CPU workers and model runners.
- `vllm/v1/attention/`: V1 attention backend interfaces and backend-specific
  metadata/planning.
- `vllm/model_executor/`: model definitions, layers, quantization, attention
  modules, shared model loading/execution code.
- `vllm/distributed/kv_transfer/`: KV connectors for disaggregated prefill,
  external KV storage, and offloading.
- `vllm/v1/kv_offload/`: V1 offload managers and policies.
- `vllm/v1/metrics/`: scheduler/request/cache stats and Prometheus/logging
  publishers.
- `benchmarks/`: workload generators and benchmark clients.
- `docs/design/metrics.md`: metrics design and observability model.

## Request Flow

At a high level:

1. The frontend receives a request and performs input processing.
2. A V1 engine client sends an `EngineCoreRequest` to an engine core.
3. `EngineCore` creates a `Request`, including token IDs and optional block
   hashes for prefix caching.
4. `Scheduler.schedule()` chooses the next batch.
5. The scheduler emits a `SchedulerOutput` with:
   - new requests to add to workers,
   - cached/running requests to update,
   - per-request scheduled token counts,
   - KV block IDs,
   - finished/preempted request IDs,
   - optional KV connector metadata.
6. The model executor sends that work to workers.
7. The model runner prepares tensors, attention metadata, block tables, and KV
   cache views.
8. The worker executes the model and returns sampled token IDs.
9. The scheduler consumes outputs, updates request state, frees finished KV
   blocks, and repeats.

The core loop matters because agentic workloads are often dominated by many
small, bursty turns rather than a simple large batch of homogeneous prompts.
That means queueing, admission, prefix hits, chunked prefill, decode priority,
and per-step CPU overhead can dominate the observed latency.

## Scheduler Mental Model

The scheduler is in `vllm/v1/core/sched/scheduler.py`. The central function is
`Scheduler.schedule()`.

V1 does not maintain a hard "prefill phase" versus "decode phase" inside the
scheduler. Each request has:

- `num_computed_tokens`: how much has already been computed.
- `num_tokens_with_spec`: the current target including prompt, generated tokens,
  and speculative tokens.

Each step tries to schedule enough tokens for requests to catch up, while
respecting:

- `max_num_scheduled_tokens`, usually `max_num_batched_tokens`;
- `max_num_seqs`;
- KV block availability;
- encoder/multimodal budgets;
- LoRA constraints;
- structured-output and remote-KV readiness;
- optional speculative decode lookahead.

The key scheduling order is:

1. Schedule already-running requests first.
2. Then admit waiting requests if there is still token budget and no request was
   preempted.
3. If a running request cannot allocate KV blocks, preempt another request and
   recompute later.

This is why V1 chunked prefill tends to prioritize decode work: decode requests
are already running, so they consume budget before new prefills are admitted.
The optimization guide describes this user-facing behavior as batching pending
decode requests before pending prefills.

## FCFS/FIFO Policy Details

The policy code is small and worth reading directly:
`vllm/v1/core/sched/request_queue.py`.

`SchedulerConfig.policy` defaults to `"fcfs"` in `vllm/config/scheduler.py`.
FCFS is implemented as `FCFSRequestQueue`, a `deque`:

- `add_request()` appends to the right.
- `pop_request()` pops from the left.
- `peek_request()` reads index 0.
- `prepend_request()` appends to the left for skipped or resumed work.

There is also `"priority"`, implemented with a heap. Lower numeric priority is
served first, and arrival time breaks ties.

Important caveats for interpreting "FCFS":

- FCFS applies primarily to the waiting queues. Already-running requests are
  scheduled before waiting requests.
- In FCFS mode, `_select_waiting_queue_for_scheduling()` returns
  `skipped_waiting` before `waiting`. Requests blocked on remote KV,
  structured-output grammar, or streaming input can re-enter through this path.
- `schedule()` explicitly has cases where it continues past a running request
  that cannot be scheduled in the current step, so lower-priority/later requests
  may still get work. This avoids wasting the step when one request is
  temporarily blocked.
- When KV blocks are exhausted under FCFS, the scheduler preempts from the tail
  of the running list.

For agentic workloads, FCFS can therefore be "FIFO-ish" at admission time while
still producing non-FIFO service at token-step granularity. This is usually good
for GPU utilization, but it can obscure why a particular tool-call turn saw high
TTFT or inter-token latency.

## KV Cache Mental Model

There are two levels of KV cache state:

1. Scheduler-side metadata.

   This lives in `vllm/v1/core/`. It tracks which request owns which logical KV
   blocks, prefix-cache hashes, ref counts, free blocks, and eviction order.

2. Worker-side tensors.

   These live in worker/model-runner code. The scheduler deals in block IDs; the
   model runner maps those IDs to actual GPU tensor views and attention metadata.

The startup path is:

1. `EngineCore._initialize_kv_caches()` asks executors/workers for KV cache
   specs.
2. Workers derive specs by inspecting attention layers through
   `GPUModelRunner.get_kv_cache_spec()`.
3. The engine profiles available memory and builds a `KVCacheConfig`.
4. The scheduler creates `KVCacheManager`.
5. The worker creates actual KV tensors in
   `GPUModelRunner.initialize_kv_cache_tensors()`.

The scheduler allocation path is:

1. For a new waiting request, `Scheduler.schedule()` calls
   `KVCacheManager.get_computed_blocks()` to find local prefix-cache hits.
2. If a KV connector exists, the scheduler may also query external KV hits.
3. `KVCacheManager.allocate_slots()` checks whether the request can fit.
4. It frees skipped blocks for sliding-window-like attention.
5. It touches prefix-hit blocks so they cannot be evicted.
6. It allocates new blocks from `BlockPool`.
7. It caches newly completed full blocks by assigning block hashes.

The important block data structure is `KVCacheBlock` in
`vllm/v1/core/kv_cache_utils.py`:

- `block_id`: physical block ID.
- `ref_cnt`: number of active references.
- `block_hash`: prefix-cache hash key once a full block is cacheable.
- free-list links for O(1) queue operations.
- `is_null`: sentinel block for skipped/sparse attention positions.

`BlockPool` owns all blocks. It keeps:

- a free queue in eviction order;
- a hash map from prefix block hash to cached block;
- an always-reserved null block;
- optional KV cache events;
- optional residency metrics.

Freeing a request decrements block ref counts and appends refcount-zero blocks
back into the free queue. With prefix caching enabled, freed full blocks can stay
hash-addressable as eviction candidates until reused.

## Prefix Caching and Sliding Windows

Prefix caching is hash based. A `Request` computes a chain of block hashes from
its tokens. For full attention, cache lookup walks hashes left to right until the
first miss. For sliding-window attention, the manager searches for a suffix of
contiguous cached blocks that satisfies the window requirement and pads skipped
positions with the null block.

This is important for agentic workloads because many requests may share system
prompts, tool schemas, conversation prefixes, or retrieval boilerplate. However,
small differences in prompt serialization, tool ordering, cache salt, LoRA, or
multimodal extra keys can break reuse.

For experiments, log both:

- user-visible cache hit rate from prefix-cache stats;
- block-level residency/eviction behavior, because hit rate alone does not tell
  you whether useful blocks are being evicted between agent turns.

## Existing Observability Hooks

Start with built-in observability before adding new instrumentation:

- Prometheus metrics are documented in `docs/usage/metrics.md` and designed in
  `docs/design/metrics.md`.
- `SchedulerStats` in `vllm/v1/metrics/stats.py` includes running/waiting
  request counts, KV cache usage, prefix-cache stats, connector cache stats,
  optional KV eviction samples, speculative decoding stats, and connector stats.
- `--kv-cache-metrics` enables sampled KV block residency metrics: lifetime,
  idle time, and reuse gaps. The sample rate is controlled by
  `--kv-cache-metrics-sample`.
- `--kv-events-config` can enable publishing KV cache events through the
  `KVEventsConfig` path. When enabled, `BlockPool` emits block stored/removed
  events and the scheduler publishes them as `KVEventBatch`.
- `ObservabilityConfig.enable_logging_iteration_details` causes `EngineCore` to
  log per-iteration context/generation request and token details.

For timeline visualization, useful data is split across:

- request lifecycle events in `Request` / scheduler output processing;
- scheduler step outputs: scheduled token counts, preemptions, queue sizes;
- KV events: block stored and block removed;
- prefix-cache stats and KV residency metrics;
- benchmark client timestamps, especially TTFT, ITL, and E2E latency.

## Suggested Visualization For Agentic Experiments

For an agentic workload, the most useful visualization is not just aggregate
throughput. Build a per-request and per-step timeline:

- x-axis: time or scheduler step.
- rows: request ID or conversation/session ID.
- colored spans: prefill compute, decode compute, waiting, skipped waiting,
  remote-KV wait, preempted, finished.
- markers: first token, tool-call boundary, resumed turn, preemption,
  prefix-cache hit/miss, KV block eviction.
- side plots: running queue size, waiting queue size, skipped-waiting queue
  size, token budget used, KV cache usage, prefix hit rate.

Minimal instrumentation points:

- `Scheduler.schedule()` after `SchedulerOutput` is constructed.
- `Scheduler.make_stats()` for aggregate queue/cache stats.
- `KVCacheManager.allocate_slots()` for allocation success/failure and block
  counts.
- `BlockPool.cache_full_blocks()` and `_maybe_evict_cached_block()` for block
  store/remove events.
- `OutputProcessor` request-finish stats for TTFT, decode time, queued time,
  and generated token counts.

If you want a low-risk first pass, do not modify the hot path initially. Run
with Prometheus/log stats/KV metrics enabled, then correlate server-side metrics
with benchmark output. Add per-step JSONL tracing only after you know which
missing fields block the analysis.

## Why vLLM May Be Suboptimal For Agentic Workloads

These are hypotheses to test against the code and metrics:

1. Decode-first chunked prefill helps ITL for running requests but can delay
   TTFT for newly arrived short turns during bursts.

2. FCFS admission is not the same as per-token FIFO service. Already-running
   requests and skipped-waiting behavior can make service order diverge from
   arrival order.

3. Agent prompts often have high prefix-sharing potential, but small formatting
   differences can destroy prefix-cache hits.

4. Short, bursty requests amplify CPU overhead in input processing, scheduler
   bookkeeping, output processing, and streaming.

5. KV block churn can evict useful shared prefixes between turns, especially
   when many conversations interleave.

6. Tool calls create phase changes: a request may decode a small tool call,
   leave the model, then return with a longer updated prompt. Aggregate metrics
   can hide this sawtooth pattern.

7. The default policy optimizes broad serving throughput/latency tradeoffs, not
   necessarily session affinity, conversation-level fairness, or tool-loop
   locality.

## Experiment Checklist

For a first experiment series:

1. Pick one small model and one stable agent trace format.
2. Run a baseline with default V1 settings.
3. Collect TTFT, ITL, E2E, request prompt/generation lengths, queue sizes,
   KV cache usage, prefix-cache stats, and eviction samples.
4. Sweep `max_num_batched_tokens`.
5. Compare `--scheduling-policy fcfs` and `--scheduling-policy priority` only if
   your client can assign meaningful priorities.
6. Test prompt serialization stability: same tool schema order, same system
   prefix, same cache salt behavior.
7. Repeat with prefix caching on/off.
8. Add timeline tracing only for fields that built-in metrics cannot answer.

The first question to answer is not "is vLLM slow for agents?" It is: which
phase dominates for each agent turn: queueing, prefill, decode, output
streaming, or KV churn?

## Files To Keep Open While Debugging

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/sched/request_queue.py`
- `vllm/config/scheduler.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/metrics/stats.py`
- `vllm/v1/core/kv_cache_metrics.py`
- `docs/design/metrics.md`
- `docs/configuration/optimization.md`
- `docs/design/prefix_caching.md`
