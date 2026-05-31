# Agentix Motivation Replication Plan

This note focuses on replicating the *motivation* evidence in Agentix, not the
Agentix speedup results. The goal is to reproduce the workload characterization
and measurement logic behind Section 3: program-level wait time, request-level
head-of-line blocking, program-level head-of-line blocking, and intra-program
KV-cache locality.

## Source Grounding

Primary paper:

- Local PDF: `agent-papers/Agentix- An Efficient Serving Engine for LLM Agents as General Programs.pdf`
- USENIX page: https://www.usenix.org/conference/nsdi26/presentation/luo

Public workload anchors used by the paper:

- ShareGPT conversations:
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- Berkeley Function Calling Leaderboard:
  https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- LATS code:
  https://github.com/lapisrocks/LanguageAgentTreeSearch
- HotpotQA, used by LATS in the paper:
  https://huggingface.co/datasets/hotpotqa/hotpot_qa

Local vLLM code that already helps:

- `vllm/benchmarks/serve.py`: modern `vllm bench serve` implementation,
  request-rate control, Poisson/Gamma arrivals, latency metrics.
- `docs/benchmarking/cli.md`: documented ShareGPT download and load-pattern
  flags.
- `benchmarks/multi_turn/benchmark_serving_multi_turn.py`: multi-turn chat
  replay client that records per-turn request statistics and a rough cached
  prefix percentage.
- `benchmarks/multi_turn/bench_dataset.py`: ShareGPT parser and synthetic
  multi-turn dataset helpers.

## What Agentix Uses In Motivation

Agentix treats each user/session/workflow as a program. A program contains LLM
calls and external interrupts. The motivation section uses three workload types:

| Workload | Agentix role | Public source | Shape to preserve |
| --- | --- | --- | --- |
| ShareGPT | chatbot program | ShareGPT Vicuna unfiltered cleaned split | multi-turn conversations; mean around 6.66 LLM calls, max around 80 in the paper |
| BFCL | ReAct/tool-call program | BFCL V3 multi-turn data | tool-signature-heavy prefills, short decodes, mean around 10.75 calls |
| LATS on HotpotQA | MCTS/multi-threaded program | LATS official repo plus HotpotQA | many parallel calls; paper reports mean around 159.7 LLM calls |

The important replication point is not exact model answers. For motivation, we
need traces with realistic per-program call counts, input lengths, output
lengths, dependency structure, and program IDs. We can replay recorded traces or
generate trace manifests from datasets and use `ignore_eos`/fixed output lengths
where needed to control decode lengths.

## Measurements To Reproduce

### Workload Statistics

Corresponds to Agentix Figure 11.

For every program and every LLM call:

- `program_id`
- `call_id`
- `parent_call_ids` or thread/group ID for DAG workloads
- `arrival_time_s`
- `prompt_tokens`
- `output_tokens`
- optional `tool_name`
- optional `tool_latency_s`

Derived plots:

- input/prefill token distribution per workload
- output/decode token distribution per workload
- number of LLM calls per program
- for DAG workloads, active fanout and critical path length

### Program Execution And Wait Time

Corresponds to Agentix Figure 5.

Agentix defines single-threaded program latency as:

- wait time: total time spent by the program's LLM calls in the engine queue
- execution time: cumulative model forward/decode time of LLM calls
- interceptions: tool or human time outside the LLM engine

For vLLM replication, request-level client timing is not enough. We need one of:

- server-side request lifecycle events: request admitted, first scheduled,
  scheduled tokens per step, finished; or
- a careful approximation using client send/receive times plus server TTFT/TPOT.

Minimum useful exact server fields:

- `queued_at_s`
- `first_scheduled_at_s`
- `finished_at_s`
- `model_time_s` or step-level scheduled token timing
- `preempted_count`
- `num_computed_tokens`
- `num_cached_tokens`

Program-level derived metrics:

- `program_wait_s = sum(call.first_scheduled_at_s - call.queued_at_s)`
- `program_execution_s = sum(call.finished_at_s - call.first_scheduled_at_s)`
- `program_wall_s = max(call.finished_at_s) - program_arrival_s`
- `program_external_s = program_wall_s - program_wait_s - program_execution_s`

For multi-threaded/DAG programs, use critical-path wall time separately from
sum-over-all-calls work. Agentix reports program-level token latency for DAGs as
critical-path response time divided by total tokens across threads.

### Wait / Execution Ratios

Corresponds to Agentix Figure 6.

Request-level HoL:

- bucket calls by output/decode tokens
- plot or tabulate `call_wait_s / call_execution_s`
- short calls with high ratios indicate long calls blocked them

Program-level HoL:

- bucket programs by number of LLM calls
- plot or tabulate `program_wait_s / program_execution_s`
- short programs with high ratios indicate long programs blocked them

Important control:

- FCFS alone tests request and program blocking.
- MLFQ-style preemption tests whether request-level preemption alone fixes the
  problem. Agentix argues it does not, because later calls from long programs
  are still treated as fresh high-priority requests.

### Prefix Cache Locality

Corresponds to Agentix Figure 7.

Agentix compares cache-hit rates:

- intra-program: call N in a program compared with prior calls from the same
  program
- inter-program: call N compared with calls from other programs

Replicable approximation without a custom engine:

- tokenize prompts for every call
- compute longest common prefix against the previous call in the same program
- compute longest common prefix against a sample of calls from other programs
- bucket by input length and compute `lcp_tokens / input_tokens`

Exact engine-side version:

- use vLLM prefix-cache block hashes and per-request cached block counts
- record `num_cached_tokens / prompt_tokens` at request admission
- compare same-program routing versus mixed-program routing

The current multi-turn benchmark already has an `approx_cached_percent`, but it
is client-side and conversation-specific. It is enough for first-pass motivation
reproduction, not enough for engine cache claims.

## Replication Sequence

1. Build trace manifests.

   ShareGPT is easiest because vLLM already parses it. Treat each full
   conversation as one program rather than sampling only the first turn.

   BFCL requires JSONL loading from the repo/dataset files, not HuggingFace
   `load_dataset`; the dataset card explicitly says the files are JSON lines and
   not compatible with `load_dataset`.

   LATS requires running or emulating LATS over HotpotQA. For motivation-only
   reproduction, a recorded trace schema is preferable: the serving engine does
   not need the correct final answer, but it does need the same call fanout and
   token-length distribution.

2. Generate arrival traces.

   Use Poisson program arrivals, matching the Agentix paper methodology and
   existing vLLM benchmark support. In current vLLM, `--request-rate` plus
   `--burstiness 1.0` gives Poisson arrivals for requests. For Agentix-style
   experiments, the unit of arrival should be the *program*, not the individual
   LLM call.

3. Replay programs.

   A trace replay harness should submit a program's first ready call at its
   program arrival time, then submit child calls only after dependencies or tool
   gaps complete. This matters for LATS/MCTS and BFCL.

4. Instrument server lifecycle.

   Start with client-side logs, but plan for a small server-side trace hook.
   Motivation figures require queueing time, which is cleaner if emitted by the
   scheduler rather than inferred from TTFT.

5. Analyze outputs.

   Produce tables/plots for:

   - calls per program
   - input/output token distributions
   - program wait/execution/external split
   - request wait/execution ratio versus output tokens
   - program wait/execution ratio versus calls per program
   - intra/inter-program prefix locality

## What Not To Claim

- Do not claim exact Agentix speedups without implementing PLAS/ATLAS and the
  Agentix load balancer.
- Do not compare against Agentix from client-only measurements; the paper's
  scheduler behavior depends on server-side queueing and preemption.
- Do not collapse each dataset into an average trace. Agentic serving is
  sensitive to high variance in turns, prompt growth, output lengths, and tool
  gaps.

## Minimal First Scaffold

Add a trace schema and offline analyzer first. This lets us inspect whether the
workload has the properties Agentix relies on before touching vLLM internals:

- trace JSONL: one record per LLM call with `program_id`, token lengths,
  dependency IDs, and optional timing
- analyzer: workload summaries, wait/execution summaries if timing exists,
  prefix-locality estimates
- example manifest: ShareGPT, BFCL, LATS/HotpotQA, and candidate coding/search
  workloads for the critique phase

