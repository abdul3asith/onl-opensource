"""Measure how a policy behaves on the target device: latency, jitter, memory, control rate."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from deploycheck.policy import Policy


@dataclass
class ProfileResult:
    policy: str
    variant: str
    device: str
    steps: int
    mean_ms: float
    p50_ms: float
    p99_ms: float
    jitter_ms: float  # standard deviation of step latency
    max_ms: float
    hz: float  # achievable control rate = 1000 / p99 (worst-case, not mean)
    peak_mem_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


def _time_once(policy: Policy, obs: dict[str, torch.Tensor]) -> float:
    """One inference step, in milliseconds. Uses CUDA events on GPU, wall clock on CPU."""
    if policy.device.startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        policy.act(obs)
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))
    t0 = time.perf_counter()
    policy.act(obs)
    return (time.perf_counter() - t0) * 1000.0


def profile(policy: Policy, variant: str = "baseline", steps: int = 200, warmup: int = 20) -> ProfileResult:
    obs = policy.example_observation()
    for _ in range(warmup):
        policy.act(obs)

    if policy.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    lat = np.array([_time_once(policy, obs) for _ in range(steps)])
    peak_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if policy.device.startswith("cuda") else float("nan")
    )
    p99 = float(np.percentile(lat, 99))
    return ProfileResult(
        policy=policy.name,
        variant=variant,
        device=policy.device,
        steps=steps,
        mean_ms=float(lat.mean()),
        p50_ms=float(np.percentile(lat, 50)),
        p99_ms=p99,
        jitter_ms=float(lat.std()),
        max_ms=float(lat.max()),
        hz=1000.0 / p99 if p99 > 0 else float("inf"),
        peak_mem_mb=peak_mb,
    )
