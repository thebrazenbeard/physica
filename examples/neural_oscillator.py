"""Train a neural acceleration residual against a deliberately synthetic force.

This is an educational example, not a field-calibrated physical model.
Run after: pip install -e ".[learn]"
"""
import torch
from physica.learning import neural_oscillator, torch_rk4


def main():
    torch.manual_seed(7)
    rhs = neural_oscillator()
    optimizer = torch.optim.Adam(rhs.parameters(), lr=.025)
    initial = torch.tensor([1., 0.])
    # Reference is a spring of stiffness 1.2, while the baseline knows 1.0.
    reference = torch_rk4(lambda t, y: torch.stack((y[1], -1.2 * y[0])),
                          initial, t1=1., steps=20).detach()
    for _ in range(80):
        optimizer.zero_grad()
        prediction = torch_rk4(rhs, initial, t1=1., steps=20)
        loss = (prediction - reference).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        error = float((prediction - reference).square().mean())
    print({"synthetic_trajectory_mse": error, "qualification": "TRAINING_DEMO_ONLY"})


if __name__ == "__main__":
    main()
