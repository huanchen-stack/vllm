# Agentix Evaluation Reflection

## Main Concern

Agentix's motivation is useful, but it does not fully characterize modern
agentic serving. It emphasizes program-level fairness, prefix locality, and
remaining-work-aware scheduling. Those are important, but they are not the only
objectives for agents.

The biggest missing axis is SLO diversity. A chat agent, a search agent, and a
coding agent can all be "agentic" while having different notions of success:

- Chat wants low first-token and turn latency.
- Search often wants fast partial results and useful fan-in behavior.
- Coding may tolerate long runtime but is sensitive to many sequential turns,
  repository context size, and tool/result feedback.
- Batch agents may care more about throughput and cost than latency.

An evaluation that reports only aggregate wait/execution ratios can miss these
differences.

## External Interrupts

Agentix cleanly separates program latency into LLM-engine waiting time, LLM
execution time, and external interruptions. It then focuses on the first two.
That is a reasonable serving-engine boundary, but it is also a major
evaluation risk because external interruptions can dominate real agent loops.

The runtime DAG and process table help Agentix track calls after they arrive:
program/session membership, call arrival, waiting time, service time, and
critical-path-style metadata for multi-threaded programs. They do not solve the
workload-specific process that creates the next call. Chatbot agents can wait
on human thinking, search agents can wait on web or database checks, and coding
agents can wait on tests, builds, profiling, and file I/O.

Therefore Agentix is program-aware inside the LLM serving layer, but it is not
a full agent-runtime scheduler. An evaluation should vary the external
interrupt ratio:

```text
external_interrupt_time / (llm_wait_time + llm_execution_time)
```

If this ratio is high, improved LLM-call scheduling may barely move end-to-end
task latency even if it improves engine-local waiting time. If it is low, or if
many calls become ready in bursts, Agentix-style scheduling is more likely to
matter.

## Workload Limitations To Probe

Coding-agent workloads are the most important gap. They can have many turns,
but many turns do not automatically mean the scheduler should prioritize them.
The useful priority signal may be "is this on the critical path to a patch or
test result" rather than "how much work remains." Tool time can dominate model
time, and prompts can grow because of file context rather than reusable
conversation history.

Search-agent workloads are another gap. They can generate wide DAGs with many
small LLM calls. Prioritizing the longest remaining program can be harmful if
the user-visible SLO depends on the fastest good branch or the final fan-in
node.

Interactive chat is underrepresented if the motivation focuses on program
throughput. Chat often has short programs and strict tail latency targets, so a
program-aware scheduler must avoid starving simple one-turn requests.

## Adequacy Questions

Before accepting an Agentix-style scheduler, the evaluation should answer:

- Does program-level priority improve p95 or p99 latency for short programs, or
  only improve averages?
- Does it increase SLO misses for chat-like single-turn traffic?
- Does it help coding agents after tool gaps and repository-context prompts are
  included?
- Does prefix locality remain high when prompts include retrieved files,
  search snippets, or generated tool output?
- Does KV transfer batching matter compared with scheduler wait and model
  execution?
- At what external-interrupt ratio do scheduling gains stop changing
  end-to-end task latency or SLO goodput?
- Are improvements still visible under realistic arrival bursts rather than a
  trace replay with convenient pacing?

## Reflection On Implementation Scope

The current branch should stay limited to trace conversion and experiment
planning. It should not add a scheduler policy until the motivation tables show
that vLLM baseline suffers the same qualitative problems Agentix describes.

If the motivation does hold, the next policy branch should still be staged:

1. Add transfer timers and request-level trace joins.
2. Add a replay driver with explicit program DAG dependencies.
3. Add a policy simulator offline.
4. Only then add a runtime scheduler policy.

This order keeps the user in control of the assumptions before changing vLLM's
serving behavior.
