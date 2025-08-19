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


def test_mlp_prompt_conditioning_shapes():
    model = TorchBehaviorModel(
        architecture="mlp",
        input_dim=10,
        action_dim=4,
        hidden_sizes=(16,),
        learning_rate=1e-3,
        prompt_conditioning=True,
        prompt_embed_dim=8,
    )
    obs = torch.randn(2, 10)
    prompt = "move smoothly to the target"
    out = model.select_action(obs, prompt=prompt).action
    assert out.shape == (2, 4)
    metrics = model.train_step_supervised(obs, torch.randn(2, 4), prompts=[prompt, prompt])
    assert "loss" in metrics
