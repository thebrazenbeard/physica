"""Physica CLI: machine-readable runs and deliberately bounded claims."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .experiment import simulate
from .learning import train_poisson


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="physica")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("simulate", help="reproducible idealized mechanical simulation")
    p.add_argument("model", choices=["oscillator", "pendulum", "cartpole"])
    p.add_argument("--initial", nargs="+", type=float, required=True)
    p.add_argument("--t1", type=float, default=10.)
    p.add_argument("--dt", type=float, default=.01)
    p.add_argument("--method", choices=["euler", "rk4", "velocity_verlet"], default="rk4")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--out", type=Path)
    q = sub.add_parser("pinn-poisson", help="train a 1D PyTorch Poisson PINN")
    q.add_argument("--steps", type=int, default=250)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--hidden", type=int, default=24)
    q.add_argument("--collocation", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        if args.command == "simulate":
            values = {}
            for item in args.param:
                key, sep, val = item.partition("=")
                if not sep or not key or key in values:
                    raise ValueError("--param must be distinct NAME=VALUE pairs")
                values[key] = float(val)
            result = simulate(args.model, initial=tuple(args.initial),
                              t1=args.t1, dt=args.dt, method=args.method,
                              parameters=values,
                              include_trajectory=not args.summary_only)
            payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
            if args.out:
                args.out.write_text(payload, encoding="utf-8")
            else:
                sys.stdout.write(payload)
        else:
            _, report = train_poisson(steps=args.steps, seed=args.seed,
                                      hidden=args.hidden, collocation=args.collocation)
            from dataclasses import asdict
            sys.stdout.write(json.dumps(asdict(report), indent=2) + "\n")
    except (ValueError, TypeError, RuntimeError, FloatingPointError) as exc:
        print(f"physica: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
