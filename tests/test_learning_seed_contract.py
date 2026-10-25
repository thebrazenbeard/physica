"""Optional learned dynamics expose caller-controlled initialization."""

from inspect import signature
from types import SimpleNamespace

from physica.learning import _default_device_is_cpu, neural_oscillator


def test_neural_oscillator_accepts_optional_explicit_seed():
    assert "seed" in signature(neural_oscillator).parameters


def test_default_device_guard_works_without_torch_2_3_device_introspection():
    class Torch22Shape:
        # Torch 2.2 supports tensor creation with its default device, but not
        # the later torch.get_default_device convenience API.
        def __init__(self, device):
            self.device = device

        def empty(self, count):
            assert count == 0
            return SimpleNamespace(device=SimpleNamespace(type=self.device))

    assert _default_device_is_cpu(Torch22Shape("cpu"))
    assert not _default_device_is_cpu(Torch22Shape("cuda"))
