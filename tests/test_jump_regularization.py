import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))

from rsl_rl.algorithms.jump_regularization import JumpMirror
from rsl_rl.algorithms.moe_cts import MoECTS
from rsl_rl.modules import ActorCriticMoECTS


DOF_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]


def _model(num_envs=4):
    return ActorCriticMoECTS(
        48,
        266,
        12,
        num_envs,
        5,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        teacher_encoder_hidden_dims=[32, 16],
        student_encoder_hidden_dims=[32, 16],
        expert_num=8,
        latent_dim=32,
    )


def test_mirror_is_an_involution_and_has_expected_joint_signs():
    mirror = JumpMirror(DOF_NAMES, "cpu")
    observations = torch.randn(7, 48)
    actions = torch.randn(7, 12)
    torch.testing.assert_close(mirror.observations(mirror.observations(observations)), observations)
    torch.testing.assert_close(mirror.actions(mirror.actions(actions)), actions)

    action = torch.zeros(1, 12)
    action[0, DOF_NAMES.index("FR_hip_joint")] = 2.0
    action[0, DOF_NAMES.index("FR_thigh_joint")] = 3.0
    mirrored = mirror.actions(action)
    assert mirrored[0, DOF_NAMES.index("FL_hip_joint")] == -2.0
    assert mirrored[0, DOF_NAMES.index("FL_thigh_joint")] == 3.0


def test_walk_behavior_loss_ignores_jump_samples():
    model = _model()
    algorithm = MoECTS(model, 4, 5, walk_behavior_coef=0.1)
    algorithm.set_frozen_reference()
    with torch.no_grad():
        next(model.actor.parameters()).add_(0.1)
    history = torch.randn(4, 240)
    observations = torch.randn(4, 48)
    observations[:, 9] = -1.0
    assert algorithm.compute_walk_behavior_loss(observations, history).item() == 0.0
    observations[:, 9] = 0.0
    assert algorithm.compute_walk_behavior_loss(observations, history).item() > 0.0


def test_idle_jump_behavior_matches_neutral_reference_action():
    model = _model()
    algorithm = MoECTS(model, 4, 5, idle_jump_behavior_coef=1.0)
    algorithm.set_frozen_reference()
    history = torch.randn(4, 240)
    observations = torch.randn(4, 48)
    observations[:, 6] = 0.0
    observations[:, 9] = -1.0
    observations[:, 10:12] = 0.0
    assert algorithm.compute_idle_jump_behavior_loss(observations, history).item() > 0.0

    observations[:, 6] = 0.6
    assert algorithm.compute_idle_jump_behavior_loss(observations, history).item() == 0.0


def test_jump_symmetry_only_backpropagates_to_actor():
    model = _model()
    algorithm = MoECTS(model, 4, 5, jump_symmetry=True)
    algorithm.configure_jump_training(DOF_NAMES)
    observations = torch.randn(4, 48)
    observations[:, 9] = -1.0
    history = torch.randn(4, 240)
    with torch.no_grad():
        latent, _ = model.student_moe_encoder(history)
    original_mean = model.actor(torch.cat([latent, observations], dim=1))
    loss, count = algorithm.compute_jump_symmetry_loss(observations, history, original_mean)
    loss.backward()
    assert count == 4
    assert any(parameter.grad is not None for parameter in model.actor.parameters())
    assert all(parameter.grad is None for parameter in model.student_moe_encoder.parameters())
