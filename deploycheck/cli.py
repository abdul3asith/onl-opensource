"""Command line entry point.

deploycheck run --policy dummy --device cpu
deploycheck run --policy lerobot/smolvla_base --device cuda --variants baseline,fp16,compile
"""

from __future__ import annotations

from pathlib import Path

import torch
import typer
from rich.console import Console

from deploycheck import RESULTS_DIR
from deploycheck.optimize import VARIANTS, make_variant
from deploycheck.policy import load_policy
from deploycheck.profile import profile
from deploycheck.report import build_report, print_table, write_json
from deploycheck.verify import verify

app = typer.Typer(
    add_completion=False, help="Profile, optimize and verify robot policies."
)
console = Console()

torch.set_float32_matmul_precision("high")


@app.command()
def run(
    policy: str = typer.Option(
        "dummy", help="'dummy' or a LeRobot repo id, e.g. lerobot/smolvla_base"
    ),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    variants: str = typer.Option(
        "baseline,fp16", help=f"comma-separated subset of {VARIANTS}"
    ),
    steps: int = typer.Option(200, help="timed inference steps per variant"),
    episodes: int = typer.Option(10, help="simulator episodes for success rate"),
    out: Path = typer.Option(
        RESULTS_DIR / "latest.json", help="where to write the JSON report"
    ),
) -> None:
    """Run the full deployability check and print a before/after table."""
    names = [v.strip() for v in variants.split(",") if v.strip()]
    if device == "cpu" and "fp16" in names:
        console.print("[yellow]fp16 skipped on CPU[/yellow]")
        names.remove("fp16")

    base = load_policy(policy, device)
    profiles, verifies = [], []
    for name in names:
        console.print(f"[bold]{name}[/bold]: building variant")
        variant = make_variant(base, name)
        profiles.append(profile(variant, variant=name, steps=steps))
        if name != "baseline":
            verifies.append(verify(base, variant, name, episodes=episodes))

    report = build_report(profiles, verifies, device)
    path = write_json(report, out)
    print_table(report, console)
    console.print(f"report written to {path}")


@app.command()
def devices() -> None:
    """List what this machine can run on."""
    console.print(f"cpu: available")
    console.print(
        f"cuda: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}"
    )


if __name__ == "__main__":
    app()
