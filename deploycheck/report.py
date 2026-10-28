"""Turn profile + verify results into a JSON file and a terminal table."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from deploycheck import __version__
from deploycheck.profile import ProfileResult
from deploycheck.verify import VerifyResult


def _device_info(device: str) -> dict:
    info = {"device": device, "platform": platform.platform(), "torch": torch.__version__}
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def build_report(
    profiles: list[ProfileResult], verifies: list[VerifyResult], device: str
) -> dict:
    return {
        "deploycheck_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": _device_info(device),
        "profiles": [p.to_dict() for p in profiles],
        "verify": [v.to_dict() for v in verifies],
    }


def write_json(report: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return out


def print_table(report: dict, console: Console | None = None) -> None:
    console = console or Console()
    verify_by_variant = {v["variant"]: v for v in report["verify"]}
    table = Table(title=f"deploycheck · {report['hardware'].get('gpu', report['hardware']['device'])}")
    for col in ("variant", "p50 ms", "p99 ms", "jitter ms", "Hz (p99)", "peak MB", "drift", "success"):
        table.add_column(col, justify="right" if col != "variant" else "left")
    for p in report["profiles"]:
        v = verify_by_variant.get(p["variant"], {})
        success = v.get("success_rate")
        table.add_row(
            p["variant"],
            f"{p['p50_ms']:.2f}",
            f"{p['p99_ms']:.2f}",
            f"{p['jitter_ms']:.2f}",
            f"{p['hz']:.1f}",
            f"{p['peak_mem_mb']:.0f}" if p["peak_mem_mb"] == p["peak_mem_mb"] else "n/a",
            f"{v['action_drift']:.4f}" if "action_drift" in v else "–",
            f"{success:.0%}" if success is not None else "not run",
        )
    console.print(table)
