import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))

from rsl_rl.modules import ActorCriticMoECTS


CHECKPOINT = (
    ROOT
    / "logs"
    / "win_flat_slow_moe_cts"
    / "Jul05_17-22-50_flat_slow"
    / "model_85000.pt"
)


def test_selected_checkpoint_is_strictly_compatible():
    model = ActorCriticMoECTS(
        48,
        266,
        12,
        1,
        5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        teacher_encoder_hidden_dims=[512, 256],
        student_encoder_hidden_dims=[512, 256, 256],
        expert_num=8,
        latent_dim=32,
        norm_type="l2norm",
    )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert len(checkpoint["model_state_dict"]) == 37
