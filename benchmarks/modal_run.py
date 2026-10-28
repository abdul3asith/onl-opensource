"""Run deploycheck on a Modal GPU.

    modal run benchmarks/modal_run.py --gpu A10G --policy dummy
    modal run benchmarks/modal_run.py --gpu L40S --policy lerobot/smolvla_base --variants baseline,fp16,compile
    modal run benchmarks/modal_run.py --gpu H100 --policy lerobot/smolvla_base --steps 500

The JSON report is saved to a Modal volume AND printed, so you can paste the table.
Change GPU_TYPES below if you want to sweep several GPUs in one call.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1")
    .pip_install("uv")
    # Install the repo itself (with LeRobot) into the image.
    .add_local_dir(
        REPO_ROOT, remote_path="/repo", copy=True, ignore=[".venv", "results", ".git"]
    )
    .run_commands("cd /repo && uv pip install --system -e '.[lerobot]'")
)

app = modal.App("deploycheck", image=image)
results_vol = modal.Volume.from_name("deploycheck-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("deploycheck-hf-cache", create_if_missing=True)


def _run(policy: str, variants: str, steps: int, episodes: int, gpu_label: str) -> dict:
    """Runs inside the container. Imports happen here so the image has them."""
    from deploycheck.optimize import make_variant
    from deploycheck.policy import load_policy
    from deploycheck.profile import profile
    from deploycheck.report import build_report, print_table, write_json
    from deploycheck.verify import verify

    names = [v.strip() for v in variants.split(",") if v.strip()]
    base = load_policy(policy, "cuda")
    profiles, verifies = [], []
    for name in names:
        variant = make_variant(base, name)
        profiles.append(profile(variant, variant=name, steps=steps))
        if name != "baseline":
            verifies.append(verify(base, variant, name, episodes=episodes))

    report = build_report(profiles, verifies, "cuda")
    report["hardware"]["modal_gpu"] = gpu_label
    safe = policy.replace("/", "_")
    out = Path("/results") / f"{safe}_{gpu_label}.json"
    write_json(report, out)
    results_vol.commit()
    print_table(report)
    return report


# One function per GPU type. Modal picks the GPU from the decorator.
@app.function(
    gpu="T4",
    timeout=1800,
    volumes={"/results": results_vol, "/root/.cache/huggingface": hf_cache},
)
def run_t4(policy: str, variants: str, steps: int, episodes: int) -> dict:
    return _run(policy, variants, steps, episodes, "T4")


@app.function(
    gpu="L4",
    timeout=1800,
    volumes={"/results": results_vol, "/root/.cache/huggingface": hf_cache},
)
def run_l4(policy: str, variants: str, steps: int, episodes: int) -> dict:
    return _run(policy, variants, steps, episodes, "L4")


@app.function(
    gpu="A10G",
    timeout=1800,
    volumes={"/results": results_vol, "/root/.cache/huggingface": hf_cache},
)
def run_a10g(policy: str, variants: str, steps: int, episodes: int) -> dict:
    return _run(policy, variants, steps, episodes, "A10G")


@app.function(
    gpu="L40S",
    timeout=1800,
    volumes={"/results": results_vol, "/root/.cache/huggingface": hf_cache},
)
def run_l40s(policy: str, variants: str, steps: int, episodes: int) -> dict:
    return _run(policy, variants, steps, episodes, "L40S")


@app.function(
    gpu="A100",
    timeout=1800,
    volumes={"/results": results_vol, "/root/.cache/huggingface": hf_cache},
)
def run_a100(policy: str, variants: str, steps: int, episodes: int) -> dict:
    return _run(policy, variants, steps, episodes, "A100")


@app.function(
    gpu="H100",
    timeout=1800,
    volumes={"/results": results_vol, "/root/.cache/huggingface": hf_cache},
)
def run_h100(policy: str, variants: str, steps: int, episodes: int) -> dict:
    return _run(policy, variants, steps, episodes, "H100")


RUNNERS = {
    "T4": run_t4,
    "L4": run_l4,
    "A10G": run_a10g,
    "L40S": run_l40s,
    "A100": run_a100,
    "H100": run_h100,
}


@app.local_entrypoint()
def main(
    gpu: str = "A10G",
    policy: str = "dummy",
    variants: str = "baseline,fp16,compile",
    steps: int = 200,
    episodes: int = 10,
) -> None:
    """Use gpu='all' to sweep every GPU type in parallel."""
    targets = list(RUNNERS) if gpu == "all" else [gpu]
    calls = [RUNNERS[g].spawn(policy, variants, steps, episodes) for g in targets]
    local_dir = REPO_ROOT / "results"
    local_dir.mkdir(exist_ok=True)
    for g, call in zip(targets, calls):
        report = call.get()
        out = local_dir / f"{policy.replace('/', '_')}_{g}.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"[{g}] saved {out}")
