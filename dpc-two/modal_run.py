# modal_run.py: run deploycheck2 on a cloud GPU.
#   modal run modal_run.py
#   modal run modal_run.py --variants baseline,bf16 --steps 500

import subprocess
from pathlib import Path

import modal

GPU = "A10G"                      # change to "L4", "L40S", "A100", "H100", ...
HERE = Path(__file__).parent      # the deploycheck2 folder on your Mac

# The cloud machine: Linux + Python + torch + numpy + your code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc")                                   # needed by torch.compile on GPU
    .pip_install("torch", "numpy")
    .add_local_dir(HERE, remote_path="/root/app",
                   ignore=[".venv", "__pycache__", "*.json"])   # copy your code, skip junk
)

app = modal.App("deploycheck2", image=image)


@app.function(gpu=GPU, timeout=900)
def run_on_gpu(variants: str, steps: int) -> str:
    """Runs in the cloud: execute run.py on the GPU and send back the results file."""
    subprocess.run(
        ["python", "run.py", "--device", "cuda",
         "--variants", variants, "--steps", str(steps),
         "--out", "/tmp/results.json"],
        cwd="/root/app",
        check=True,                                       # stop with an error if run.py fails
    )
    return Path("/tmp/results.json").read_text()


@app.local_entrypoint()
def main(variants: str = "baseline,fp16,bf16,compile", steps: int = 200):
    """Runs on your Mac: start the cloud job, then save its results locally."""
    results = run_on_gpu.remote(variants, steps)
    out = HERE / f"results_{GPU}.json"
    out.write_text(results)
    print(f"saved {out}")
