#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract a sub-grid of a lookup-table policy for use as a small fixture.

The frontier and prompt-bucket axes are truncated to the first ``N`` cells;
the live-batch axis is kept whole.  Inputs that fall inside the kept region
index exactly as in the full table (the flat layout is frontier-major), so a
replay restricted to that region is unchanged.  Metadata other than the
table is copied verbatim; ``description`` is annotated.

    trim_policy.py --policy full.json --out trimmed.json \
        --frontier-count 44 --prompt-bucket-count 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def trim_lookup_table(
    table: dict[str, Any], frontier_count: int, prompt_bucket_count: int
) -> dict[str, Any]:
    old_frontiers = int(table["frontier_count"])
    old_prompts = int(table["prompt_bucket_count"])
    lives = int(table["live_batch_count"])
    frontier_count = min(frontier_count, old_frontiers)
    prompt_bucket_count = min(prompt_bucket_count, old_prompts)
    cells = table["committed_frontiers"]
    kept: list[int] = []
    for frontier_index in range(frontier_count):
        for prompt_index in range(prompt_bucket_count):
            start = (frontier_index * old_prompts + prompt_index) * lives
            kept.extend(int(value) for value in cells[start : start + lives])
    trimmed = dict(table)
    trimmed["frontier_count"] = frontier_count
    trimmed["prompt_bucket_count"] = prompt_bucket_count
    trimmed["committed_frontiers"] = kept
    return trimmed


def trim_policy(
    policy: dict[str, Any], frontier_count: int, prompt_bucket_count: int
) -> dict[str, Any]:
    trimmed = dict(policy)
    trimmed["lookup_table"] = trim_lookup_table(
        policy["lookup_table"], frontier_count, prompt_bucket_count
    )
    trimmed["description"] = (
        f"{policy.get('description', '')} [trimmed fixture: first {frontier_count} "
        f"frontiers x {prompt_bucket_count} prompt buckets]"
    ).strip()
    return trimmed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frontier-count", type=int, required=True)
    parser.add_argument("--prompt-bucket-count", type=int, required=True)
    args = parser.parse_args(argv)
    policy = json.loads(Path(args.policy).read_text())
    trimmed = trim_policy(policy, args.frontier_count, args.prompt_bucket_count)
    Path(args.out).write_text(json.dumps(trimmed, separators=(",", ":")) + "\n")
    print(f"{args.out}: {Path(args.out).stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
