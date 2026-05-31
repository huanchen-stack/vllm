# Agentix Evaluation Experiment Plan

## Objective

This branch plans evaluation experiments after the transparent instrumentation
branch. The goal is to let the user inspect whether Agentix-style motivation
claims still hold under vLLM, and then decide whether a scheduler policy change
is worth implementing.

The evaluation is intentionally split into layers:

1. Offline workload validation: confirm the trace has agent-program structure.
2. Baseline vLLM replay: collect scheduler traces without changing policy.
3. Transfer-path inspection: measure whether host/device KV batching matters.
4. Workload gap analysis: test cases Agentix motivation does not cover well.

## Required Inputs

- A client-side call DAG trace in the `benchmarks/agentix_explore` schema.
- vLLM scheduler JSONL from `AGENTIX_EXPLORE_TRACE_JSONL`.
- Request ids encoded as `program_id::call_id`.
- A workload manifest recording dataset source, conversion assumptions, model,
  max tokens, arrival process, and tool-delay replay policy.

The scheduler trace alone cannot reconstruct a true agent DAG. It gives engine
lifecycle evidence for requests. The client-side DAG trace supplies program
dependencies, ready times, and tool gaps.

## Baseline Experiments

Run the same replay with these configurations:

- `vllm-baseline`: current scheduler, no trace.
- `vllm-traced`: current scheduler, trace enabled.
- `vllm-offload`: CPU KV offload enabled if the target experiment needs host
  and device transfer pressure.
- `vllm-offload-traced`: offload plus scheduler trace.

The first comparison estimates trace overhead. The second comparison tells us
whether transfer effects are visible enough to justify adding transfer timers.

## Core Metrics

Per call:

- queue wait: first scheduled timestamp minus queued timestamp;
- execution span: finished timestamp minus first scheduled timestamp;
- wait/execution ratio;
- prompt tokens, output tokens, scheduled token count;
- local and external cached token counts at first schedule;
- preemption count.

Per program:

- end-to-end latency from first ready call to final finished call;
- total queued wait and total execution span;
- critical-path latency if parent dependencies are available;
- wait/execution ratio binned by calls per program;
- slowdown versus an isolated or low-load replay.

System:

- throughput in calls/s and tokens/s;
- p50/p95/p99 call and program latency;
- SLO miss rate under workload-specific targets;
- KV offload bytes, transfer time, and batching degree once transfer timers are
  added.

## Motivation Replication Tables

The motivation replication should produce these tables before trying to match
Agentix figures:

- Program length distribution: calls per program and token work per program.
- Short-call penalty: wait/execution ratio by output-token or total-token bin.
- Short-program penalty: program wait/execution ratio by calls-per-program bin.
- Prefix locality: intra-program versus inter-program cache hit proxy.
- Attained-service proxy: served tokens or served turns versus remaining work.

If these tables do not show the same qualitative pressure points, there is no
reason to implement an Agentix-like scheduler yet.

## Workloads To Add Beyond Agentix Motivation

Agentix motivation is strongest for many dependent calls with meaningful prefix
reuse and moderate program length. The evaluation should also include:

- Coding agents: many turns, heavy tool gaps, large file-context prompts, and
  uneven success criteria.
- Search agents: fan-out/fan-in DAGs, short SLOs for interactive search, and
  high value for first useful answer.
- Chat agents: few turns, strict latency SLOs, and user-visible tail latency.
- Batch analysis agents: long calls where throughput can matter more than
  per-call latency.

These workloads probe whether "longer remaining program equals higher
priority" is enough. In coding and search, priority can depend more on SLO,
critical path, uncertainty, or user-visible milestone than on raw remaining
tokens.

## Planned Analysis Workflow

1. Convert client DAG traces to normalized call JSONL.
2. Run `analyze_traces.py` on workload-only traces.
3. Replay through vLLM with request ids set to `program_id::call_id`.
4. Convert scheduler JSONL with `merge_scheduler_trace.py`.
5. Join client and scheduler traces by `(program_id, call_id)`.
6. Re-run `analyze_traces.py` on the joined trace.
7. Produce workload tables and decide whether policy implementation is justified.

No experiment should be interpreted from scheduler JSONL alone unless the
question is purely about request-level queueing inside vLLM.
