"""Run immutable bag pairs and compare estimates using a single manifest."""

import csv
import json
from pathlib import Path
from time import monotonic
import yaml
from .bags import Bag, load_config, open_bag
from .eval import evaluate
from .run import process, save_json


def bench(path, out):
    root = Path(path).resolve().parent
    cases = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))["cases"]
    rows = []
    names = set()
    for case in cases:
        name = case["name"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in (".", "..")
            or name in names
        ):
            raise ValueError(f"Expected unique case name: {name}")
        names.add(name)
        results = out / name
        results.mkdir()
        cfgpath = lambda key: root / case[key] if case.get(key) else None
        common = load_config(cfgpath("config"))
        variants = (
            ["kinematic", "survey", "inspection"]
            if common["odom"]
            else ["survey", "inspection"]
        )
        for mode in variants:
            role = "inspection" if mode == "inspection" else "survey"
            if not case.get(role):
                continue
            dest = results / mode
            dest.mkdir()
            cfg = (
                load_config(cfgpath("inspection_config"))
                if role == "inspection" and case.get("inspection_config")
                else common.copy()
            )
            if mode == "kinematic":
                cfg.update(imu=None, loop=False, point_time_field=None)
            row = dict(case=name, mode=mode)
            start = monotonic()
            try:
                with open_bag(root / case[role], cfgpath("msg_dir")) as reader:
                    report = process(
                        Bag(reader, cfg),
                        cfg,
                        dest,
                        results / "survey" if mode == "inspection" else None,
                        case.get("initial") if mode == "inspection" else None,
                        baseline=mode == "kinematic",
                    )
                row.update(
                    scans=report["scans"],
                    matched=report["matched"],
                    prior_only=report["prior_only"],
                    tracking=report["tracking"],
                    median_nn_m=report["median_match_residual_m"],
                )
                gt = cfgpath(role + "_gt")
                if gt:
                    align = case.get(role + "_align", "none")
                    fixed = None
                    if align == "fixed-map":
                        if mode != "inspection":
                            raise ValueError("fixed-map applies to inspection")
                        fixed = json.loads(
                            (results / "survey" / "metrics.json").read_text(encoding="utf-8-sig")
                        )["gt_T_map"]
                        align = "none"
                    metrics = evaluate(
                        dest / "trajectory.tum",
                        gt,
                        align,
                        case.get("max_gap", 0.25),
                        case.get("delta", 1.0),
                        fixed,
                    )
                    save_json(dest / "metrics.json", metrics)
                    row.update(
                        alignment=metrics["alignment"],
                        coverage=metrics["coverage"],
                        ate_m=metrics["ate_rmse_m"],
                        rpe_m=metrics["rpe_translation_rmse_m"],
                        rpe_deg=metrics["rpe_rotation_rmse_deg"],
                    )
                row["status"] = "ok"
            except Exception as e:
                row.update(status="error", detail=str(e))
            row["seconds"] = round(monotonic() - start, 2)
            rows.append(row)
            print(f"{name} / {mode}: {row['status']}", flush=True)
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "mode",
                "status",
                "detail",
                "seconds",
                "scans",
                "matched",
                "prior_only",
                "tracking",
                "median_nn_m",
                "alignment",
                "coverage",
                "ate_m",
                "rpe_m",
                "rpe_deg",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return dict(runs=len(rows), errors=sum(r["status"] == "error" for r in rows))
