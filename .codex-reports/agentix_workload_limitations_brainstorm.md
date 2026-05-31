# Agentix Workload Limitations Brainstorm

This note asks whether the Agentix motivation workloads cover the agent-serving
space well enough. The short answer is no: they are useful, but they mainly
stress program-level head-of-line blocking and prefix locality. They do not
fully cover long-horizon tool-interleaved agents, coding agents, tool latency,
cache retention across tool gaps, or SLO differences between workload classes.

## What Agentix Covers Well

Agentix covers these axes:

- multi-call programs instead of independent requests
- short versus long programs
- single-threaded chat/tool-call patterns
- multi-threaded MCTS-like DAGs
- prefix reuse within a program
- request and program head-of-line blocking

This is enough to motivate program-aware scheduling over request-only FCFS or
request-only MLFQ.

## What Is Under-Covered

### Long-Horizon Tool-Interleaved Agents

The Continuum paper argues that modern ReAct-style agents repeatedly alternate
LLM requests with tool calls over dozens or hundreds of turns. Its local paper
copy reports collected SWE-Bench and BFCL traces, and explicitly says Autellix
or Agentix-style PLAS underperforms on SWE-Bench because it assumes programs
that have already received more service have longer expected remaining time.

That assumption is fragile for coding agents. A coding task can be long because
it is close to completion but still needs several short verification/edit turns.
Lowering its priority can create repeated per-turn queueing bubbles and destroy
interactive job completion time.

### Tool Latency And Turn Gaps

Agentix models external interrupts as outside the LLM engine and mostly removes
them from the optimization target. That is reasonable for pure scheduler theory,
but in real agents tool gaps determine whether KV cache should remain resident
or be evicted.

Continuum highlights two missing costs:

- turn-based eviction cost: the engine treats a tool call as request
  completion, so the next turn may pay prefill/reload again
- per-turn queueing delay: after a tool returns, the next LLM request must
  queue behind unrelated work even if the program is logically continuous

### External Interrupts Can Dominate

Agentix decomposes program latency into LLM-engine waiting time, LLM execution
time, and external interruptions such as human input or tool calls. Its
motivation focuses on reducing waiting and execution time inside the serving
engine. That boundary is clean, but it can hide the dominant term in many
agentic workloads.

Different task classes create different intra-program LLM-call arrival
patterns:

- chatbot agents wait on human thinking and typing between turns
- search agents wait on web fetches, database checks, parsing, and reranking
- coding agents wait on shell commands, tests, builds, profiling, and file I/O
- data-analysis agents wait on SQL queries, notebook execution, or remote jobs

Agentix's runtime DAG/process-table construction can observe new LLM calls when
they arrive and can track parent or critical-path metadata for active calls. It
does not model, predict, or optimize the external process that decides when the
next call becomes ready. As a result, Agentix can be very relevant when many
LLM calls are ready or when serving wait dominates, but much less relevant when
external interrupts dominate end-to-end task time.

This should be an explicit replication axis: vary
`external_interrupt_time / (llm_wait_time + llm_execution_time)` and measure
when Agentix-style scheduling still changes end-to-end latency, SLO goodput, or
completed-task rate.

### SLO Heterogeneity

Agentix mainly reports throughput at equal latency and program-level token
latency. It does not deeply separate workload objectives:

- chat: TTFT and low tail latency matter; completion throughput is secondary
- coding agents: job completion time, pass rate under wall-clock limits, and
  continuity across turns matter
- search/research agents: tail job completion time and budgeted breadth matter
- batch/offline: makespan and cost per completed task matter more than TTFT
- background agents: throughput and fairness can dominate interactive latency

One scheduling policy can look good on average program-token latency and still
miss SLOs for interactive or pass/fail workloads.

### Outcome Quality Coupling

Agentix motivation is serving-centric. For coding agents, serving latency can
affect quality indirectly:

- wall-clock task limits can reduce pass rate
- slower tool/LLM loop can trigger client-side timeouts
- cache eviction and context reconstruction can change prompt formatting or
  truncation behavior if the client compresses history

The serving benchmark should keep outcome metrics nearby: pass rate, timeout
rate, and completed-task rate under SLO.

### Parallel Tool Calls

Agentix's MCTS workload covers LLM-call DAGs, but production agent traces often
have parallel tool calls whose outputs merge into one next LLM request. From
the engine perspective, this appears as one larger prefill after a tool gap, not
as independent decode calls. The scheduling pressure is different.

## Workloads To Add

| Workload | Why it matters | Candidate source |
| --- | --- | --- |
| SWE-Bench via mini-swe-agent | coding loops, many tool turns, pass/fail outcome | mini-swe-agent plus SWE-Bench/SWE-Bench Verified |
| OpenHands / multi-SWE-bench | realistic coding agent harness and repository tools | OpenHands examples |
| BFCL V4 Web Search | modern web-search tool loops | BFCL V4 |
| Tau-bench or tau2-bench | user/tool interaction and policy tasks | public benchmark repos |
| Deep research / search trace | long-running browsing and summarization | recorded trace or synthetic harness |
| Office/document agent trace | many filesystem/document tool calls | public harness or synthetic trace |
| Code QA trace | very high turn-count tail, up to hundreds of turns in recent public workload reports | trace-style workload if accessible |

## Experiments To Run Next

### 1. Remaining-Work Assumption Test

Question: does "more served tokens or turns means lower priority" correlate
with larger remaining work?

For every trace and every turn:

- `served_so_far_tokens`
- `served_so_far_turns`
- `remaining_tokens`
- `remaining_turns`
- `remaining_wall_time`
- `remaining_success_probability` if outcome data exists

Plot correlations per workload. If the correlation is weak or negative, PLAS is
not a reliable approximation for shortest remaining processing time.

### 2. Tool-Gap Sensitivity Test

Question: how much latency is caused by gaps outside the LLM engine?

Replay the same traces with tool latencies scaled:

- zero tool delay
- recorded tool delay
- p50-only delay
- p95-heavy delay
- random long-tail delay

Measure:

- job completion time
- queueing after tool return
- cache hit or recompute rate
- resident KV memory pressure

### 3. SLO-Specific Goodput

Question: do scheduling policies optimize the right thing for each workload?

Report goodput instead of only average token latency:

- chat: requests meeting TTFT and TPOT SLOs
- coding: completed tasks under wall-clock limit and pass-rate-preserving
  goodput
- search/research: completed traces under job SLO and token budget
- batch: makespan and cost per completed trace

### 4. Cache Retention Versus Scheduling

Question: is program-aware priority enough without cross-turn KV retention?

Compare:

- FCFS
- request priority / MLFQ-like
- PLAS-like priority
- session-aware FCFS
- PLAS plus TTL pinning
- TTL pinning without PLAS

This disentangles whether Agentix's gains come from program priority, prefix
locality, or both.

### 5. Parallelism Shape Test

Question: does ATLAS help real agent DAGs or mostly synthetic/MCTS DAGs?

Classify traces by:

- sequential turns
- fanout/fanin width
- parallel LLM calls
- parallel tool calls
- critical path fraction of total work

Then evaluate wait/execution and makespan per class.

## Adequacy Criteria

Before trusting an agent-serving scheduling result, the benchmark should cover:

- turn-count distribution, including long tails
- prompt growth and context-window pressure
- output length distribution
- tool-call latency distribution
- tool-output token distribution
- dependency graph shape
- prefix/KV reuse opportunity
- outcome/SLO metrics, not just average latency
- multi-tenant workload mix
- admission mode: online Poisson, bursty, and offline batch

Agentix satisfies several of these for LLM-call scheduling, but not enough for
coding/search agents where tool continuity and workload-specific SLOs dominate.

## Practical Next Step

Start with a trace schema that can represent both Agentix-style workloads and
Continuum-style tool-interleaved workloads. Then build analyzers that answer
two questions before running a server:

1. Does this workload have Agentix-friendly structure: short programs should
   have low attained service, and longer attained service should imply more
   remaining work?
2. Does this workload have Continuum-friendly structure: repeated turns, short
   tool gaps, and high value from retaining KV between tool calls?

Only after that should we implement scheduler or cache-retention experiments.
