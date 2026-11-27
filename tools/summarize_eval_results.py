#!/usr/bin/env python3
"""Summarize RoboTwin episode JSON results without importing the model stack."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=None)
    args = parser.parse_args()
    records = []
    malformed = []
    for path in sorted(args.results.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value.get("task"), str) or not isinstance(value.get("success"), bool):
                raise ValueError("requires string task and boolean success")
            records.append(value)
        except Exception as exc:
            malformed.append({"path": str(path), "error": str(exc)})
    grouped = defaultdict(lambda: [0, 0])
    for record in records:
        grouped[record["task"]][0] += 1
        grouped[record["task"]][1] += int(record["success"])
    episodes = len(records)
    successes = sum(v[1] for v in grouped.values())
    output = {
        "results": str(args.results),
        "episodes": episodes,
        "successes": successes,
        "failures": episodes - successes,
        "success_rate_percent": round(100 * successes / episodes, 2) if episodes else None,
        "malformed_json": malformed,
        "tasks": [],
    }
    for task, values in sorted(grouped.items()):
        item = {"task": task, "episodes": values[0], "successes": values[1], "success_rate_percent": round(100 * values[1] / values[0], 2)}
        if args.episodes_per_task is not None:
            item["expected_episodes"] = args.episodes_per_task
            item["complete"] = values[0] == args.episodes_per_task
        output["tasks"].append(item)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if malformed or (args.episodes_per_task and any(not task["complete"] for task in output["tasks"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
