"""Emit an explicitly synthetic known-parameter inverse benchmark as JSON.

Run from the repository root:
    python examples/synthetic_inverse_data.py > observations.json
    python -m physica inverse-oscillator --data observations.json \
        --initial 1 0 --mass 1 --guess 2 0.1 --holdout-every 5 \
        --source-label "synthetic example"
"""
from dataclasses import asdict
from json import dumps

from physica.inverse import synthetic_oscillator_observations


if __name__ == "__main__":
    samples = synthetic_oscillator_observations(
        times=[i * 0.2 for i in range(31)],
        mass=1., stiffness=4., damping=.35,
        initial=(1., 0.), sigma=.01,
        noise=True, seed=123,
    )
    print(dumps([asdict(s) for s in samples], indent=2, allow_nan=False))
