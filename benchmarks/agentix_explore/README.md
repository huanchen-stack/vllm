# Agentix Motivation Exploration Scaffold

This directory is a reasoning scaffold for reproducing the motivation-style
measurements from Agentix. It does not implement PLAS, ATLAS, or any scheduler
change. The first objective is to make workload assumptions explicit before
running expensive serving experiments.

## Trace Unit

Use one JSONL record per LLM call:

```json
{
  "workload": "sharegpt",
  "program_id": "conv-0001",
  "call_id": "turn-0003",
  "parent_call_ids": ["turn-0002"],
  "ready_time_s": 12.5,
  "queued_at_s": 12.6,
  "first_scheduled_at_s": 12.9,
  "finished_at_s": 14.2,
  "prompt_tokens": 1034,
  "output_tokens": 128,
  "cached_prompt_tokens": 960,
  "tool_name": null,
  "tool_latency_s": null
}
```

The required fields for workload characterization are `program_id`,
`call_id`, `prompt_tokens`, and `output_tokens`. Timing and cache fields are
optional, but they are needed for Agentix Figure 5/6/7 style measurements.

## Intended Workflow

1. Convert ShareGPT, BFCL, LATS/HotpotQA, or coding-agent traces into this
   JSONL format.
2. Use `workload_manifest.example.json.template` as the checklist for dataset
   sources and conversion notes.
3. Run the analyzer locally only after the user approves the exact trace:

   ```bash
   .venv/bin/python benchmarks/agentix_explore/analyze_traces.py \
     --trace-jsonl /path/to/trace.jsonl \
     --output-md /tmp/agentix-trace-summary.md
   ```

4. Inspect whether the trace supports Agentix's assumptions:

   - programs have long-tailed LLM call counts;
   - short calls and short programs suffer high wait/execution ratios;
   - intra-program prefix locality is higher than inter-program locality;
   - attained service correlates with remaining work.

5. Only after the trace looks right should scheduler instrumentation be added.

## Why This Is Separate From `vllm bench serve`

`vllm bench serve` is request-oriented. Agentix motivation needs
program-oriented accounting:

- one program arrival can create many dependent LLM calls;
- a tool gap is not engine execution time but changes KV-cache value;
- request-level latency is not enough to compute program wait time;
- DAG workloads need critical-path accounting, not just total token work.

The analyzer here is intentionally offline and simple so the user can supervise
the workload semantics before touching vLLM runtime code.
