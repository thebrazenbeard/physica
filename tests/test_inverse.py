"""Inverse oscillator behavior and defensible identifiability boundaries."""
import math

import pytest

pytest.importorskip("scipy")
from physica.inverse import (Observation, fit_oscillator, synthetic_oscillator_observations)


def _data():
    return synthetic_oscillator_observations(
        times=tuple(i * .2 for i in range(31)),
        mass=1., stiffness=4., damping=.35, initial=(1., 0.),
        sigma=.01, seed=0, noise=False,
    )


def test_recover_identifiable_parameters_with_independent_holdout():
    fit = fit_oscillator(_data(), initial=(1., 0.), mass=1.,
                         initial_guess=(2., .1),
                         holdout_indices=tuple(range(4, 31, 5)),
                         source_label="analytic synthetic test")
    assert fit.optimizer_converged
    assert fit.identifiability == "LOCAL_FULL_RANK"
    assert fit.jacobian_rank == 2
    assert math.isfinite(fit.condition_number)
    assert abs(fit.stiffness - 4.) < .02
    assert abs(fit.damping - .35) < .02
    assert fit.holdout_rmse < .005
    assert fit.training_count == 25
    assert fit.holdout_count == 6
    assert fit.evidence_ceiling == "UNVERIFIED_MODEL_FIT_ONLY"


def test_caller_source_label_cannot_upgrade_evidence():
    result = fit_oscillator(_data(), initial=(1., 0.), mass=1.,
                            initial_guess=(2., .1),
                            holdout_indices=(4, 9, 14, 19),
                            source_label="synthetic trust me")
    assert result.evidence_ceiling == "UNVERIFIED_MODEL_FIT_ONLY"


def test_zero_excitation_is_rank_deficient_not_certified():
    data = (Observation(i*.25, 0., .01) for i in range(20))
    fit = fit_oscillator(list(data), initial=(0., 0.), mass=1.,
                         initial_guess=(2., .2),
                         holdout_indices=(4, 9, 14, 19),
                         source_label="zero excitation")
    assert fit.optimizer_converged
    assert fit.jacobian_rank == 0
    assert fit.identifiability == "RANK_DEFICIENT"
    assert fit.holdout_rmse == 0.


def test_holdout_not_used_in_optimization():
    data = _data()
    hold = tuple(range(4, 31, 5))
    original = fit_oscillator(data, initial=(1., 0.), mass=1.,
                              initial_guess=(2., .1), holdout_indices=hold,
                              source_label="synthetic")
    changed = list(data)
    for i in hold:
        item = changed[i]
        changed[i] = Observation(item.time, item.position + .5, item.sigma)
    result = fit_oscillator(changed, initial=(1., 0.), mass=1.,
                            initial_guess=(2., .1), holdout_indices=hold,
                            source_label="synthetic")
    assert abs(original.stiffness - result.stiffness) < 1e-9
    assert abs(original.damping - result.damping) < 1e-9
    assert result.holdout_rmse > original.holdout_rmse + .3


@pytest.mark.parametrize("change", [
    lambda x: [Observation(x[0].time, x[0].position, 0.)] + x[1:],
    lambda x: [Observation(x[0].time, math.nan, .01)] + x[1:],
    lambda x: [x[1], x[0]] + list(x[2:]),
    lambda x: [x[0], x[0]] + list(x[2:]),
])
def test_invalid_observation_input_rejected(change):
    with pytest.raises(ValueError):
        fit_oscillator(change(_data()), initial=(1.,0.), mass=1.,
                       initial_guess=(2., .1), holdout_indices=(4, 9, 14, 19),
                       source_label="synthetic")


def test_invalid_split_and_provenance_rejected():
    data = _data()
    for indices in ((), (0, 0), tuple(range(29)), (99,)):
        with pytest.raises(ValueError):
            fit_oscillator(data, initial=(1.,0.), mass=1.,
                           initial_guess=(2., .1),
                           holdout_indices=indices, source_label="synthetic")
    with pytest.raises(ValueError):
        fit_oscillator(data, initial=(1.,0.), mass=1.,
                       initial_guess=(2., .1), holdout_indices=(4,9),
                       source_label="")


def test_reproducible_seeded_noisy_synthetic_data():
    opts = dict(times=tuple(i*.2 for i in range(20)), mass=1.,
                stiffness=4., damping=.35, initial=(1.,0.),
                sigma=.01, seed=27, noise=True)
    a = synthetic_oscillator_observations(**opts)
    b = synthetic_oscillator_observations(**opts)
    assert a == b
    assert any(abs(x.position - y.position) > 1e-5
               for x,y in zip(a, synthetic_oscillator_observations(**{**opts, "seed":28})))


def test_documented_synthetic_dataset_generator():
    import json
    import subprocess
    import sys
    cp = subprocess.run(
        [sys.executable, "examples/synthetic_inverse_data.py"],
        capture_output=True, text=True,
    )
    assert cp.returncode == 0, cp.stderr
    rows = json.loads(cp.stdout)
    assert len(rows) == 31
    assert all(set(x) == {"time", "position", "sigma"} for x in rows)
    assert len({x["time"] for x in rows}) == 31


def test_cli_reads_powershell_utf16_redirected_json(tmp_path):
    import json
    import subprocess
    import sys
    source = tmp_path / "observations-ps.json"
    # Windows PowerShell redirection may produce UTF-16 with a BOM.
    source.write_bytes(json.dumps([vars(o) for o in _data()]).encode("utf-16"))
    result = subprocess.run([
        sys.executable, "-m", "physica", "inverse-oscillator",
        "--data", str(source), "--initial", "1", "0", "--mass", "1",
        "--guess", "2", ".1", "--holdout-every", "5",
        "--source-label", "caller-asserted synthetic",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["holdout_count"] > 0


def test_cli_explicit_source_and_partition(tmp_path):
    import json
    import subprocess
    import sys
    source = tmp_path / "observations.json"
    source.write_text(json.dumps([vars(o) for o in _data()]), encoding="utf-8")
    cp = subprocess.run([
        sys.executable, "-m", "physica", "inverse-oscillator",
        "--data", str(source), "--initial", "1", "0", "--mass", "1",
        "--guess", "2", ".1", "--holdout-every", "5",
        "--source-label", "synthetic tests"
    ], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    j = json.loads(cp.stdout)
    assert j["identifiability"] == "LOCAL_FULL_RANK"
    assert j["holdout_count"] > 0
    assert j["source_label"] == "synthetic tests"
