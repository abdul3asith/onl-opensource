"""Produce faster variants of a policy for the target device.

Each optimizer takes a Policy and returns a new Policy with the same interface,
so profile() and verify() can treat every variant the same way.
"""

from __future__ import annotations

import copy

import torch

from deploycheck.policy import Policy

VARIANTS = ("baseline", "fp16", "compile", "int8")


def _clone(policy: Policy) -> Policy:
    return copy.deepcopy(policy)


def to_fp16(policy: Policy) -> Policy:
    """Half precision. Only meaningful on GPU.

    Weights go to fp16 and act() runs under autocast, so float32 tensors a model makes
    inside its forward (e.g. SmolVLA's denoising noise) are cast to match the weights.
    """
    p = _clone(policy)
    p.module().half()
    orig_obs, orig_act = p.example_observation, p.act
    device_type = torch.device(policy.device).type

    def obs_fp16() -> dict[str, torch.Tensor]:
        return {
            k: v.half() if v.is_floating_point() else v for k, v in orig_obs().items()
        }

    def act_fp16(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.autocast(device_type, dtype=torch.float16):
            return orig_act(obs)

    p.example_observation = obs_fp16  # type: ignore[method-assign]
    p.act = act_fp16  # type: ignore[method-assign]
    p.name = policy.name
    return p


def compiled(policy: Policy) -> Policy:
    """torch.compile the underlying module."""
    p = _clone(policy)
    net = torch.compile(p.module())
    # DummyPolicy stores the net in _net; LeRobotPolicy in _policy. Keep this the only
    # place that knows that.
    if hasattr(p, "_net"):
        p._net = net  # type: ignore[attr-defined]
    else:
        p._policy = net  # type: ignore[attr-defined]
    return p


def _fake_quantize_int8_(module: torch.nn.Module) -> None:
    """Round every Linear weight to int8 and back. Same numerics as weight-only int8,
    no speedup. Lets drift and success checks run on machines with no int8 kernels."""
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                w = m.weight
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
                m.weight.copy_((w / scale).round().clamp(-128, 127) * scale)


def to_int8(policy: Policy) -> Policy:
    """Int8 Linear layers.

    CPU: real dynamic quantization if a quantized engine exists (x86 fbgemm, ARM qnnpack),
    otherwise fake quantization (numerics only, no speedup).
    GPU: TODO(day 3) torchao / TensorRT int8.
    """
    p = _clone(policy)
    net = p.module()
    engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if policy.device == "cpu" and engines:
        torch.backends.quantized.engine = engines[0]
        try:
            net = torch.ao.quantization.quantize_dynamic(
                net, {torch.nn.Linear}, dtype=torch.qint8
            )
        except RuntimeError:
            _fake_quantize_int8_(net)
    else:
        _fake_quantize_int8_(net)
    if hasattr(p, "_net"):
        p._net = net  # type: ignore[attr-defined]
    else:
        p._policy = net  # type: ignore[attr-defined]
    return p


def make_variant(policy: Policy, variant: str) -> Policy:
    if variant == "baseline":
        return policy
    if variant == "fp16":
        return to_fp16(policy)
    if variant == "compile":
        return compiled(policy)
    if variant == "int8":
        return to_int8(policy)
    raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
