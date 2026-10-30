"""Loads the real lerobot/smolvla_base checkpoint on CPU.

Skipped unless LeRobot with SmolVLA support is installed. The first run downloads the
weights from the Hugging Face Hub.
"""

import pytest
import torch

from deploycheck.policy import LeRobotPolicy

pytest.importorskip("lerobot.policies.smolvla.modeling_smolvla")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def policy() -> LeRobotPolicy:
    return LeRobotPolicy("lerobot/smolvla_base", device="cpu")


def test_smolvla_observation_matches_config(policy):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    cfg = policy.module().config
    obs = policy.example_observation()
    assert cfg.image_features
    for key, ft in cfg.image_features.items():
        assert obs[key].shape == (1, *ft.shape)
        assert 0 <= obs[key].min() and obs[key].max() <= 1
    assert obs[OBS_STATE].shape == (1, *cfg.robot_state_feature.shape)
    assert obs[OBS_LANGUAGE_TOKENS].dtype == torch.long
    assert obs[OBS_LANGUAGE_ATTENTION_MASK].dtype == torch.bool
    assert obs[OBS_LANGUAGE_ATTENTION_MASK].any()
    # optimize.py and verify.py cast every value, so the observation must be tensors only.
    assert all(isinstance(v, torch.Tensor) for v in obs.values())


def test_smolvla_act_returns_one_action(policy):
    action = policy.act(policy.example_observation())
    assert action.shape == (1, policy.module().config.action_feature.shape[0])
    assert torch.isfinite(action).all()
