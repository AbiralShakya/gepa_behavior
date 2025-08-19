import torch
from gepa.models import TorchBehaviorModel


def test_mlp_forward_and_grad():
    model = TorchBehaviorModel(
        architecture="mlp",
        input_dim=10,
        action_dim=4,
        hidden_sizes=(32, 32),
        learning_rate=1e-3,
    )
    obs = torch.randn(8, 10)
    target = torch.randn(8, 4)
    out = model.select_action(obs).action
    assert out.shape == (8, 4)
    metrics = model.train_step_supervised(obs, target)
    assert "loss" in metrics
