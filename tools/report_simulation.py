"""Aggregate completed RoboTwin and LIBERO Plus rollouts with strict denominators."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROBOTWIN_PLUS = {"mixed", "warm", "underexposure", "contrast"}


def robotwin_campaigns(directories):
    """Require exactly one campaign per corruption, all from one policy."""
    campaigns = []
    for directory in directories:
        manifest = directory.parent.parent / "campaign.json"
        if not manifest.is_file():
            raise ValueError(f"missing RoboTwin campaign manifest: {manifest}")
        campaign = json.loads(manifest.read_text())
        if campaign.get("tag") != directory.name or campaign.get("episodes_per_task") != 100:
            raise ValueError(f"campaign manifest does not match result directory: {directory}")
        campaigns.append(campaign)
    if {c.get("condition") for c in campaigns} != ROBOTWIN_PLUS:
        raise ValueError("the RoboTwin-Plus mean requires mixed, warm, underexposure, and contrast")
    for field in ("checkpoint", "recipe_sha256", "vae", "stats_sha256", "noise_seed_base"):
        if any(not c.get(field) for c in campaigns) or len({c[field] for c in campaigns}) != 1:
            raise ValueError(f"RoboTwin-Plus campaign mismatch: {field}")


def report(rows, fields):
    out = {}
    for field in ("overall", *fields):
        total, success = Counter(), Counter()
        for row in rows:
            key = row[field] if field != "overall" else "all"
            total[key] += 1
            success[key] += row["success"]
        out[field] = {key: {"episodes": n, "successes": success[key],
                            "success_rate": success[key] / n} for key, n in sorted(total.items())}
    return out


def read_results(directory: Path, benchmark: str):
    for path in sorted(directory.glob("*/*.json")):
        row = json.loads(path.read_text())
        if benchmark == "libero" and row.get("status") != "complete":
            continue
        if type(row.get("success")) is not bool:
            raise ValueError(f"Missing Boolean success: {path}")
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=["robotwin", "libero"])
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if len(args.results) not in ((1, 4) if args.benchmark == "robotwin" else (1,)):
        parser.error("provide one result directory, or the four RoboTwin corruption directories")
    if args.benchmark == "libero" and not args.manifest:
        parser.error("LIBERO Plus requires --manifest")
    if len(args.results) == 4:
        try:
            robotwin_campaigns(args.results)
        except ValueError as exc:
            parser.error(str(exc))
    reports = {}
    for directory in args.results:
        if not directory.is_dir():
            parser.error(f"not a directory: {directory}")
        rows = list(read_results(directory, args.benchmark))
        if args.benchmark == "robotwin":
            tasks = (Path(__file__).resolve().parents[1] /
                     "eval/scripts/fastwam_eval/tasks/all.txt").read_text().split()
            counts = Counter(r["task"] for r in rows)
            ids = {(r["task"], r["seed"]) for r in rows}
            scene_ids = {(r["task"], r.get("actual_seed")) for r in rows}
            complete = counts == dict.fromkeys(tasks, 100)
            if len(ids) != len(rows):
                parser.error(f"duplicate trial: {directory}")
            if not args.allow_partial and (len(scene_ids) != len(rows) or any(
                r.get("actual_seed") is None for r in rows
            )):
                parser.error(f"missing or duplicate scene seeds: {directory}")
            fields = ("task",)
        else:
            jobs = json.loads(args.manifest.read_text())
            expected = {(r["suite"], r["task_id"], r["seed"]): r for r in jobs}
            ids = {(r["suite"], r["task_id"], r["seed"]) for r in rows}
            if len(ids) != len(rows) or not ids <= expected.keys():
                parser.error("duplicate or unexpected LIBERO Plus trial")
            for row in rows:
                job = expected[row["suite"], row["task_id"], row["seed"]]
                if any(row[key] != job[key] for key in ("name", "category", "language")):
                    parser.error("LIBERO Plus result does not match the manifest")
            complete = len(ids) == len(expected) == 10030
            fields = ("suite", "category")
        if not complete and not args.allow_partial:
            parser.error(f"incomplete paper protocol: {directory} ({len(rows)} episodes)")
        reports[str(directory)] = {"complete": complete, **report(rows, fields)}
    result = {"conditions": reports}
    if len(reports) == 4 and all(r["complete"] for r in reports.values()):
        result["robotwin_plus_success_rate"] = sum(
            r["overall"]["all"]["success_rate"] for r in reports.values()
        ) / 4
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
