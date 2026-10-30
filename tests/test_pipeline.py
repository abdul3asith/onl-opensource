"""Runs on CPU with no LeRobot installed, so CI and new contributors can run it anywhere."""

import json

import torch
from typer.testing import CliRunner

from deploycheck.cli import app
from deploycheck.optimize import make_variant
from deploycheck.policy import DummyPolicy
from deploycheck.profile import profile
from deploycheck.verify import verify


def test_profile_reports_tail_latency():
    p = DummyPolicy(device="cpu")
    r = profile(p, steps=30, warmup=3)
    assert r.steps == 30
    assert r.p99_ms >= r.p50_ms > 0
    assert r.jitter_ms >= 0
    assert r.hz > 0


def test_int8_variant_keeps_interface_and_small_drift():
    base = DummyPolicy(device="cpu")
    q = make_variant(base, "int8")
    v = verify(base, q, "int8", episodes=0)
    assert v.action_drift < 0.5  # dynamic int8 should not wreck a tiny model
    assert v.success_rate is None  # simulator not wired yet


class _Fp32InsidePolicy(DummyPolicy):
    """Like SmolVLA: act() makes a float32 tensor itself and feeds it to the net."""

    def act(self, obs):
        noise = torch.rand(1, self.state_dim, device=self.device)  # always float32
        return self._net(obs["image"], noise)


def test_fp16_variant_runs_when_act_makes_fp32_tensors():
    p = make_variant(_Fp32InsidePolicy(device="cpu"), "fp16")
    assert p.act(p.example_observation()).dtype == torch.float16


def test_cli_writes_report(tmp_path):
    out = tmp_path / "r.json"
    result = CliRunner().invoke(
        app, ["run", "--policy", "dummy", "--device", "cpu", "--variants", "baseline,int8",
              "--steps", "10", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert {p["variant"] for p in report["profiles"]} == {"baseline", "int8"}
    assert report["verify"][0]["variant"] == "int8"
