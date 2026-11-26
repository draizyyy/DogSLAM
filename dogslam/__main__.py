import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from .bags import Bag, inspect_bag, load_config, open_bag
from .bench import bench
from .demo import generate
from .eval import evaluate
from .run import process, save_json


@contextmanager
def destination(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Output exists: {path}. Choose a new run directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        yield stage
        stage.rename(path)


def cli():
    p = argparse.ArgumentParser(prog="dogslam")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "map", "localize"):
        q = sub.add_parser(name)
        q.add_argument("bag")
        q.add_argument("--msg-dir")
        if name == "inspect":
            q.add_argument("--samples", type=int, default=120)
        else:
            q.add_argument("--config")
            q.add_argument("--out", required=True)
            q.add_argument("--limit", type=int, help="maximum emitted scans")
            if name == "localize":
                q.add_argument("--map", required=True)
                q.add_argument(
                    "--initial", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG")
                )
                q.add_argument(
                    "--initial-z", type=float, help="start height in map metres with --initial"
                )
            else:
                q.add_argument("--kinematic-only", action="store_true")
    q = sub.add_parser('export-bag')
    q.add_argument('run', type=Path)
    q.add_argument('--map', required=True, type=Path)
    q.add_argument('--out', required=True)
    q = sub.add_parser('bench')
    q.add_argument("manifest")
    q.add_argument("--out", required=True)
    q = sub.add_parser("demo")
    q.add_argument("--out", required=True)
    q = sub.add_parser("evaluate")
    q.add_argument("estimate")
    q.add_argument("truth")
    q.add_argument("--align", choices=["none", "first", "se3"], default="none")
    q.add_argument("--delta", type=float, default=1.0)
    q.add_argument("--max-gap", type=float, default=0.25)
    q.add_argument("--json", type=Path)
    q.add_argument(
        "--map-alignment", type=Path, help="survey evaluate JSON with gt_T_map"
    )
    a = p.parse_args()
    if a.cmd == "localize" and a.initial_z is not None and a.initial is None:
        p.error("--initial-z requires --initial")
    if a.cmd == "demo":
        with destination(a.out) as out:
            generate(out, False)
            generate(out, True)
        return
    if a.cmd == 'export-bag':
        from .export import export_bag
        with destination(a.out) as out:
            result = export_bag(a.run, a.map, out / 'bag')
    elif a.cmd == 'bench':
        with destination(a.out) as out:
            result = bench(a.manifest, out)
        if result["errors"]:
            sys.exit(f"Failed runs: {result}; see summary.csv")
    elif a.cmd == "evaluate":
        if min(a.delta, a.max_gap) <= 0:
            p.error("delta and max-gap must be positive")
        fixed = (
            json.loads(a.map_alignment.read_text())["gt_T_map"]
            if a.map_alignment
            else None
        )
        result = evaluate(a.estimate, a.truth, a.align, a.max_gap, a.delta, fixed)
        if a.json:
            save_json(a.json, result)
    else:
        with open_bag(a.bag, a.msg_dir) as reader:
            if a.cmd == "inspect":
                result = inspect_bag(reader, a.samples)
            else:
                cfg = load_config(a.config)
                baseline = getattr(a, "kinematic_only", False)
                if baseline:
                    if not cfg["odom"]:
                        p.error("--kinematic-only needs an explicit odom topic")
                    cfg.update(imu=None, loop=False, point_time_field=None)
                with destination(a.out) as out:
                    result = process(
                        Bag(reader, cfg),
                        cfg,
                        out,
                        getattr(a, "map", None),
                        getattr(a, "initial", None),
                        a.limit,
                        baseline,
                        initial_z=getattr(a, "initial_z", None) or 0.0,
                    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        cli()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        sys.exit(f"dogslam: {exc}")
