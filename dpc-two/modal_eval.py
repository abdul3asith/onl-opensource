import subprocess
from pathlib import Path

import modal

GPU = "A10G"
POLICY = "HuggingFaceVLA/smolvla_libero"
HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "cmake",
        "pkg-config",
        "libegl1",
        "libegl1-mesa-dev",
        "libgl1",
        "libgl1-mesa-dev",
        "libgles2-mesa-dev",
        "libglvnd-dev",
        "libosmesa6",
        "libosmesa6-dev",
        "libglib2.0-0",
    )
    .env(
        {
            "CMAKE_POLICY_VERSION_MINIMUM": "3.5",  # let new CMake build old packages like eglprobe
            "MUJOCO_GL": "egl",
        }
    )
    .pip_install("pip", "setuptools", "wheel")
    .pip_install(
        "lerobot[smolvla,libero] @ git+https://github.com/huggingface/lerobot.git"
    )
)

app = modal.App("deploycheck2-eval", image=image)

# Lerobot --> smolVLA(policy) --> compare og/optimized versions --> In LIBERO -->

# app = modal.App("deploycheck2-eval", image=image)

# Storage that survives between runs
hf_cache = modal.Volume.from_name(
    "hf-cache", create_if_missing=True
)  # downloaded models
work = modal.Volume.from_name(
    "deploycheck-work", create_if_missing=True
)  # your scripts and outputs


@app.function(
    gpu=GPU,
    timeout=3600,  # up to 1 hour per run
    volumes={"/root/.cache/huggingface": hf_cache, "/work": work},
)
def evaluate(suite: str = "libero_object", episodes: int = 10, batch: int = 5) -> str:
    """Runs in the cloud: LeRobot plays LIBERO episodes with SmolVLA and writes a summary."""
    out_dir = f"/work/eval/{suite}"
    subprocess.run(
        [
            "lerobot-eval",
            f"--policy.path={POLICY}",  # which model
            "--policy.device=cuda",  # run it on the GPU
            "--env.type=libero",  # which simulator
            f"--env.task={suite}",  # which group of tasks
            f"--eval.n_episodes={episodes}",  # how many attempts in total
            f"--eval.batch_size={batch}",  # how many run at the same time
            f"--output_dir={out_dir}",  # where results are saved
        ],
        check=True,
        input="N\n",  # answers LIBERO's first-run question with "No"
        text=True,  # stop with an error if lerobot-eval fails
    )
    hf_cache.commit()  # keep the downloaded model for next time
    work.commit()  # keep the results

    # Collect every JSON summary LeRobot wrote and send it back
    files = sorted(Path(out_dir).rglob("*.json"))
    if not files:
        return "No JSON results found. Check the log above for the success rate."
    return "\n\n".join(f"== {f.name} ==\n{f.read_text()}" for f in files)


@app.local_entrypoint()
def main(suite: str = "libero_object", episodes: int = 10, batch: int = 5):
    """Runs on your Mac: start the cloud job, then save the results here."""
    result = evaluate.remote(suite, episodes, batch)
    out = HERE / f"eval_{suite}.txt"
    out.write_text(result)
    print(result[-2000:])  # the summary is at the end
    print(f"\nsaved {out}")
