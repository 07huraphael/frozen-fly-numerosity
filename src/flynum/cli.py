from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Frozen FlyWire numerosity and transfer experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Build a performance-independent empirical subcircuit")
    prep.add_argument("--raw", type=Path, default=Path("data/raw/zenodo_21549559"))
    prep.add_argument("--out", type=Path, default=Path("data/processed/flywire_visual.npz"))
    for command in ("run", "demo"):
        r = sub.add_parser(command)
        r.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
        r.add_argument("--out", type=Path, required=True)
        r.add_argument("--include-c", action="store_true")
        r.add_argument("--resume", action="store_true")
        if command == "run":
            r.add_argument("--graph", type=Path, default=Path("data/processed/flywire_visual.npz"))
    report = sub.add_parser("report")
    report.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare
        prepare(args.raw, args.out)
    elif args.command in ("run", "demo"):
        from .experiment import run
        from .graph import Graph
        graph = Graph.load(args.graph) if args.command == "run" else None
        config = json.loads(args.config.read_text(encoding="utf-8"))
        run(config, args.out, graph, args.include_c, args.resume)
    else:
        from .report import make_report
        make_report(args.path)


if __name__ == "__main__":
    main()
