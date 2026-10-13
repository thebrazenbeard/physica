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
    p.add_argument("--method", choices=["euler", "rk4", "rk45", "velocity_verlet"], default="rk4")
    p.add_argument("--rtol", type=float, help="RK45 relative local-error tolerance")
    p.add_argument("--atol", type=float, help="RK45 absolute local-error tolerance")
    p.add_argument("--max-attempts", type=int, help="RK45 accepted + rejected step budget")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--out", type=Path)
    q = sub.add_parser("pinn-poisson", help="train a 1D PyTorch Poisson PINN")
    q.add_argument("--steps", type=int, default=250)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--hidden", type=int, default=24)
    q.add_argument("--collocation", type=int, default=32)
    inv = sub.add_parser("inverse-oscillator", help="fit known-mass oscillator against observations")
    inv.add_argument("--data", type=Path, required=True, help="JSON array of {time,position,sigma} observations")
    inv.add_argument("--initial", nargs=2, type=float, required=True)
    inv.add_argument("--guess", nargs=2, type=float, required=True, metavar=("STIFFNESS", "DAMPING"))
    inv.add_argument("--mass", type=float, required=True)
    inv.add_argument("--holdout-every", type=int, default=5)
    inv.add_argument("--source-label", required=True, help="caller-asserted, not verified provenance")
    inv.add_argument("--out", type=Path)
    rl = sub.add_parser("rl-cartpole", help="synthetic tabular Q-learning cart-pole baseline")
    rl.add_argument("--episodes", type=int, default=50)
    rl.add_argument("--evaluate", type=int, default=10)
    rl.add_argument("--seed", type=int, default=0)
    rl.add_argument("--max-steps", type=int, default=200)
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
                              include_trajectory=not args.summary_only,
                              rtol=args.rtol, atol=args.atol,
                              max_attempts=args.max_attempts)
            payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
            if args.out:
                args.out.write_text(payload, encoding="utf-8")
            else:
                sys.stdout.write(payload)
        elif args.command == "inverse-oscillator":
            from dataclasses import asdict
            from .inverse import Observation, fit_oscillator
            if args.holdout_every < 2:
                raise ValueError("--holdout-every must be at least 2")
            # json.loads(bytes) recognizes UTF-8/16/32 BOMs, including
            # Windows PowerShell 5.1 UTF-16 redirected JSON output.
            raw = json.loads(args.data.read_bytes())
            if not isinstance(raw, list):
                raise ValueError("--data must be a JSON observation array")
            data = tuple(Observation(**entry) for entry in raw)
            holdout = tuple(range(args.holdout_every - 1, len(data), args.holdout_every))
            result = fit_oscillator(
                data, initial=tuple(args.initial), mass=args.mass,
                initial_guess=tuple(args.guess), holdout_indices=holdout,
                source_label=args.source_label,
            )
            payload = json.dumps(asdict(result), indent=2, allow_nan=False) + "\n"
            if args.out:
                args.out.write_text(payload, encoding="utf-8")
            else:
                sys.stdout.write(payload)
        elif args.command == "rl-cartpole":
            from dataclasses import asdict
            from .rl import train_tabular_cartpole
            report = train_tabular_cartpole(
                episodes=args.episodes, evaluation_episodes=args.evaluate,
                seed=args.seed, max_episode_steps=args.max_steps,
            )
            sys.stdout.write(json.dumps(asdict(report), indent=2) + "\n")
        else:
            _, report = train_poisson(steps=args.steps, seed=args.seed,
                                      hidden=args.hidden, collocation=args.collocation)
            from dataclasses import asdict
            sys.stdout.write(json.dumps(asdict(report), indent=2) + "\n")
    except (ValueError, TypeError, RuntimeError, FloatingPointError, OSError) as exc:
        print(f"physica: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
