"""One interface for any policy: observation in, action out.

Two implementations:
- DummyPolicy: a small torch model, so tests and the CLI run anywhere.
- LeRobotPolicy: wraps a LeRobot SmolVLA checkpoint (optional dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class Policy(Protocol):
    name: str
    device: str

    def example_observation(self) -> dict[str, torch.Tensor]: ...
    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor: ...
    def module(self) -> torch.nn.Module: ...


class _DummyNet(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, chunk: int) -> None:
        super().__init__()
        self.vision = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(16 + state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim * chunk),
        )
        self.chunk, self.action_dim = chunk, action_dim

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.vision(image), state], dim=1)
        return self.head(x).view(-1, self.chunk, self.action_dim)


@dataclass
class DummyPolicy:
    """A stand-in with a VLA-like shape: image + state in, action chunk out."""

    name: str = "dummy"
    device: str = "cpu"
    image_size: int = 96
    state_dim: int = 7
    action_dim: int = 7
    chunk: int = 10

    def __post_init__(self) -> None:
        torch.manual_seed(0)
        self._net = (
            _DummyNet(self.state_dim, self.action_dim, self.chunk)
            .to(self.device)
            .eval()
        )

    def example_observation(self) -> dict[str, torch.Tensor]:
        return {
            "image": torch.rand(
                1, 3, self.image_size, self.image_size, device=self.device
            ),
            "state": torch.rand(1, self.state_dim, device=self.device),
        }

    @torch.inference_mode()
    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._net(obs["image"], obs["state"])

    def module(self) -> torch.nn.Module:
        return self._net


class LeRobotPolicy:
    """Wraps a LeRobot SmolVLA checkpoint, e.g. 'lerobot/smolvla_base'.

    Keep this class the only place that knows LeRobot internals.
    """

    def __init__(self, repo_id: str, device: str = "cuda", task: str = "pick up the cube") -> None:
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore
            from lerobot.utils.constants import OBS_STATE  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "LeRobot with SmolVLA support is not installed. Run: uv pip install -e '.[lerobot]'"
            ) from e
        self.name = repo_id
        self.device = device
        self.task = task
        self._policy = SmolVLAPolicy.from_pretrained(repo_id).to(device).eval()
        cfg = self._policy.config
        # Per-key input shapes without the batch dim, under the keys select_action reads.
        self._shapes = {key: tuple(ft.shape) for key, ft in cfg.image_features.items()}
        self._shapes[OBS_STATE] = tuple(cfg.robot_state_feature.shape)
        self._language = self._tokenize(task)

    def _tokenize(self, task: str) -> dict[str, torch.Tensor]:
        """Tokenize the task the way LeRobot's SmolVLA preprocessor does.

        select_action reads token ids, not text, so do it once here instead of per step.
        """
        from lerobot.utils.constants import (  # type: ignore
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )
        from transformers import AutoTokenizer  # type: ignore

        cfg = self._policy.config
        tokens = AutoTokenizer.from_pretrained(cfg.vlm_model_name)(
            task if task.endswith("\n") else f"{task}\n",
            max_length=cfg.tokenizer_max_length,
            truncation=True,
            padding=cfg.pad_language_to,
            padding_side="right",
            return_tensors="pt",
        )
        return {
            OBS_LANGUAGE_TOKENS: tokens["input_ids"].to(self.device),
            OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].to(self.device, dtype=torch.bool),
        }

    def example_observation(self) -> dict[str, torch.Tensor]:
        """Random camera images in [0, 1] and state, plus the tokenized task. Batch size 1."""
        obs = {k: torch.rand(1, *shape, device=self.device) for k, shape in self._shapes.items()}
        return {**obs, **self._language}

    @torch.inference_mode()
    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """One control step. SmolVLA predicts n_action_steps actions at once and select_action
        replays them from a queue, so only every n-th call runs the model."""
        return self._policy.select_action(obs)

    def module(self) -> torch.nn.Module:
        return self._policy


def load_policy(spec: str, device: str) -> Policy:
    """'dummy' -> DummyPolicy; anything else is treated as a LeRobot repo id."""
    if spec == "dummy":
        return DummyPolicy(device=device)
    return LeRobotPolicy(spec, device=device)
