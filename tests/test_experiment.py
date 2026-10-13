import copy
import json
import subprocess
import sys

import pytest

from physica.experiment import simulate, verify_receipt


def test_reproducible_experiment_exact_digest():
    a = simulate("oscillator", initial=(1., 0.), t1=2., dt=.1)
    b = simulate("oscillator", initial=(1., 0.), t1=2., dt=.1)
    assert a == b
    assert verify_receipt(a)
    assert a["energy_check"]["passed"]
    assert a["qualification"] == "NUMERICAL_OUTPUT_ONLY"


def test_receipt_detects_tampering():
    a = simulate("oscillator", initial=(1., 0.), t1=1.)
    b = copy.deepcopy(a)
    b["final_state"][0] += .01
    assert verify_receipt(a) and not verify_receipt(b)


def test_nonconservative_check_not_falsely_passed():
    a = simulate("oscillator", initial=(1., 0.),
                 parameters={"damping": .2}, t1=1.)
    assert a["energy_check"] is None
    assert "dissipation" in a["energy_check_explanation"]


def test_invalid_verlet_model_rejected():
    with pytest.raises(ValueError):
        simulate("pendulum", initial=(.1, 0.), method="velocity_verlet")


def test_bad_state_dimension_rejected():
    with pytest.raises(ValueError):
        simulate("cartpole", initial=(0., 0.))


def test_malformed_receipts_fail_closed():
    assert not verify_receipt({"sha256": "x", "state": float("nan")})
    assert not verify_receipt(None)


def test_receipt_with_summary_still_valid():
    a = simulate("oscillator", initial=(1., 0.), include_trajectory=False)
    assert "trajectory" not in a
    assert verify_receipt(a)


def test_cli_invocation_from_module():
    r = subprocess.run([sys.executable, "-m", "physica", "simulate", "oscillator",
                        "--initial", "1", "0", "--t1", "1", "--dt", ".1",
                        "--summary-only"],
                       text=True, capture_output=True, check=True)
    record = json.loads(r.stdout)
    assert verify_receipt(record)
    assert record["model"] == "oscillator"


def test_cli_bad_parameter_is_error():
    r = subprocess.run([sys.executable, "-m", "physica", "simulate", "oscillator",
                        "--initial", "1", "0", "--param", "nonsense"],
                       text=True, capture_output=True)
    assert r.returncode == 2
    assert "NAME=VALUE" in r.stderr
