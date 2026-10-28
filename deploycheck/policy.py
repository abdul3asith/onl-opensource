"""One interface for any policy: observation in, action out.

Two implementations:
- DummyPolicy: a small torch model, so tests and the CLI run anywhere.
- LeRobotPolicy: wraps a Hugging Face / LeRobot pretrained policy (optional dependency).
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
    """Wraps a LeRobot pretrained policy, e.g. 'lerobot/smolvla_base'."""

    def __init__(self, repo_id: str, device: str = "cuda") -> None:
        try:
            from lerobot.policies.factory import get_policy_class  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "LeRobot is not installed. Run: uv pip install -e '.[lerobot]'"
            ) from e
        self.name = repo_id
        self.device = device
        # TODO(day 1): load via the LeRobot factory for the policy type in the repo config.
        # Keep this file the only place that knows LeRobot internals.
        self._policy = (
            get_policy_class("smolvla").from_pretrained(repo_id).to(device).eval()
        )

    def example_observation(self) -> dict[str, torch.Tensor]:
        # TODO: build from the policy's config (camera keys, state dim, task text).
        raise NotImplementedError

    @torch.inference_mode()
    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._policy.select_action(obs)

    def module(self) -> torch.nn.Module:
        return self._policy


def load_policy(spec: str, device: str) -> Policy:
    """'dummy' -> DummyPolicy; anything else is treated as a LeRobot repo id."""
    if spec == "dummy":
        return DummyPolicy(device=device)
    return LeRobotPolicy(spec, device=device)
