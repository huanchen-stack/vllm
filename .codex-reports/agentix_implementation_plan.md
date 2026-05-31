# Agentix Implementation Planning Notes

## Goal

The implementation goal for this branch is not to recreate Agentix's PLAS,
ATLAS, or published result numbers. The immediate goal is to make vLLM produce
enough program-aware runtime evidence to replay the Agentix motivation
questions against Agentix-like data:

- How much waiting happens before short LLM calls get scheduled?
- Do short programs suffer disproportionate slowdown under request-level
  serving?
- How much local or external prefix reuse is visible at schedule time?
- Where would a batched host/device KV movement optimization enter the vLLM
  path?

The code is deliberately transparent to normal vLLM behavior. It adds an
opt-in trace sink and marks an existing batched KV transfer path; it does not
change scheduling policy or request priority.

## Runtime Contract

Enable tracing only by setting:

```bash
AGENTIX_EXPLORE_TRACE_JSONL=/tmp/agentix-scheduler.jsonl
```

When unset, the trace sink returns immediately and does no file I/O. Agent
program identity is encoded outside vLLM by using request ids of the form:

```text
program_id::call_id
```

This convention keeps the branch compatible with current vLLM request structs.
The trace parser can still consume ordinary request ids, but `program_id` will
be null unless the delimiter is present.

## Scheduler Instrumentation Plan

The v1 scheduler is the right first instrumentation point because it already
sees queue admission, prefix-cache lookup results, token budgets, preemption,
and finish events. The branch adds `AgentixTraceSink` under
`vllm/v1/core/sched/agentix_trace.py` and wires it into:

- `Scheduler.__init__`: instantiate the disabled-by-default trace sink.
- `add_request`: emit `queued`.
- `schedule`, running loop: emit `scheduled_running`.
- `schedule`, waiting loop: emit `scheduled_new`, `scheduled_resumed`, or
  `waiting_for_remote_kv`.
- `schedule` end: emit `schedule_step` counters.
- `_preempt_request`: emit `preempted`.
- `_free_request`: emit `finished`.

Each request event records request id, inferred program id, scheduler timestamp,
wall timestamp, arrival time, status, priority, prompt tokens, current tokens,
computed tokens, output token count, max tokens, and preemption count. Schedule
events add scheduled token counts and prefix-cache token counts where available.

## Motivation Data Enabled by This Trace

The trace is enough to derive the following Agentix-style quantities after
joining with the client-side call DAG:

- request queue wait: first schedule timestamp minus queued timestamp;
- execution span: finish timestamp minus first schedule timestamp;
- slowdown: finish timestamp minus ready timestamp, divided by model work
  proxy;
- short-call penalty: wait/execution ratio binned by scheduled tokens;
- intra-program locality: cached prompt tokens grouped by inferred program id;
- preemption incidence: count and timing of `preempted` events.

The trace is not enough to prove an Agentix replacement scheduler is better.
That needs a policy implementation plus a workload replay driver that can issue
dependent calls according to the original agent DAG.

## KV Transfer Implementation Hook

Agentix calls out a batched host/device KV transfer optimization. Current vLLM
already has the relevant building block:

- Python caller: `vllm/v1/kv_offload/cpu/gpu_worker.py`
- custom op wrapper: `vllm/_custom_ops.py::swap_blocks_batch`
- CUDA implementation: `csrc/libtorch_stable/cache_kernels.cu`

The custom op uses `cuMemcpyBatchAsync` when available and falls back internally
when it is not. This branch therefore does not add a new experimental kernel
from scratch. It marks the call site with an `AGENTIX EXPLORE` comment so the
user can audit the exact host/device KV batching path while reading the paper.

The next implementation step, after the trace shape is accepted, is to add
measurement around `SingleDirectionOffloadingHandler.transfer_async` so an
evaluation can split schedule wait, model execution, KV offload, and KV reload.
That should be a separate branch because it touches transfer timing rather than
scheduler accounting.

## Known Limitations

- Program identity is inferred from request id text, not a typed vLLM API.
- The trace uses append-only JSONL file I/O and is intended for local
  experiments, not production metrics.
- `schedule_step` counts are scheduler-step aggregates, not per-program DAG
  state.
- The branch does not change SLO handling; chat, search, and coding workloads
  can still need different objectives.
- The branch does not recreate Agentix's attained-service scheduler.
