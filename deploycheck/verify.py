"""Check that an optimized policy still does the task.

Two levels:
- action_drift: cheap, no simulator. How far do the variant's actions move from the
  baseline's on the same observations? A large drift is an early warning.
- success_rate: the real check. Run episodes in a simulator (LIBERO) and count successes.
  TODO(day 3): wire LIBERO via lerobot-eval. Keep the function signature stable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from deploycheck.policy import Policy


@dataclass
class VerifyResult:
    variant: str
    action_drift: float  # mean abs difference vs baseline actions
    success_rate: float | None  # None until a simulator backend is wired
    episodes: int

    def to_dict(self) -> dict:
        return asdict(self)


@torch.inference_mode()
def action_drift(baseline: Policy, variant: Policy, samples: int = 50) -> float:
    torch.manual_seed(0)
    total = 0.0
    for _ in range(samples):
        obs = baseline.example_observation()
        obs_v = variant.example_observation()
        # Use the same random observation for both, cast to the variant's dtype.
        obs_v = {k: obs[k].to(obs_v[k].dtype) for k in obs}
        a0 = baseline.act(obs).float()
        a1 = variant.act(obs_v).float()
        total += float((a0 - a1).abs().mean())
    return total / samples


def success_rate(policy: Policy, task: str = "libero_object", episodes: int = 10) -> float | None:
    """Return fraction of successful episodes, or None if no simulator is available."""
    # TODO: call lerobot-eval (or LIBERO directly) and parse its success metric.
    return None


def verify(baseline: Policy, variant: Policy, variant_name: str, episodes: int = 10) -> VerifyResult:
    return VerifyResult(
        variant=variant_name,
        action_drift=action_drift(baseline, variant),
        success_rate=success_rate(variant, episodes=episodes),
        episodes=episodes,
    )
