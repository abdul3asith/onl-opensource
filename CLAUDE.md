# deploycheck — project rules for Claude Code

## What this is
A CLI that takes a robot policy (VLA) and a target device, then:
1. profiles it (p50/p99 latency, jitter, control rate, memory),
2. optimizes it (fp16, int8, torch.compile; TensorRT later),
3. verifies success rate still holds (LIBERO sim; real arm later),
4. writes a before/after report (JSON + terminal table).

Phase 1 only. No fleet deployment, OTA, containers or rollback.

## Stack
- Python 3.10+, managed with `uv`. Install: `uv pip install -e ".[dev]"`.
- Lint/format: `ruff` (run `ruff check . && ruff format .`).
- Tests: `pytest`. Every new function gets a test. Tests must run without a GPU
  and without LeRobot installed (use `DummyPolicy`).
- CLI: `typer`. Terminal output: `rich`.

## Layout
- `deploycheck/policy.py`   load a policy (LeRobot or DummyPolicy) behind one interface
- `deploycheck/profile.py`  timing + memory measurement
- `deploycheck/optimize.py` precision / compile variants
- `deploycheck/verify.py`   success-rate evaluation
- `deploycheck/report.py`   JSON + table output
- `deploycheck/cli.py`      wires the steps together
- `benchmarks/`             configs and scripts for published runs
- `results/`                JSON outputs (committed only for published benchmarks)

## Rules
- Never hard-code absolute paths. Use `pathlib` relative to the repo root,
  CLI options, or environment variables (`DEPLOYCHECK_CACHE`).
- Time GPU work with `torch.cuda.Event` and synchronize; never `time.time()` alone on CUDA.
- Always report p99 and jitter, not only the mean.
- Never change a success-rate number by hand; it must come from `verify.py`.
- Keep functions small; one responsibility each. Prefer plain functions over classes.
- Work in small steps: propose a plan, implement one piece, run `pytest`, then continue.
- No secrets in code. Tokens come from environment variables.
