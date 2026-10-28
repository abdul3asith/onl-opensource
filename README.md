# deploycheck

Profile, optimize and verify robot policies (VLAs) on real hardware.

Robot models are trained on big GPUs and then run on small chips. `deploycheck` tells you,
before deployment, whether a policy meets its timing budget on the target device, makes it
faster, and checks the task success rate still holds.

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"          # add ".[lerobot]" for real policies
deploycheck run --policy dummy --device cpu --variants baseline,int8
```

Example output:

```
variant   p50 ms  p99 ms  jitter ms  Hz (p99)  peak MB  drift   success
baseline    1.20    1.90       0.15     526.3      n/a      –   not run
int8        0.95    1.40       0.10     714.3      n/a 0.0123   not run
```

## Results

| Policy | Device | Variant | p99 ms | Hz | Success |
|---|---|---|---|---|---|
| [fill in after first real run] | | | | | |

## Layout

- `deploycheck/policy.py` one interface for any policy
- `deploycheck/profile.py` p50 / p99 latency, jitter, control rate, memory
- `deploycheck/optimize.py` fp16, torch.compile, int8 variants
- `deploycheck/verify.py` action drift now; LIBERO success rate next
- `deploycheck/report.py` JSON + table
- `benchmarks/` configs for published runs · `results/` their outputs

## Roadmap (phase 1)

- [ ] LeRobot loader for SmolVLA, π0.5, GR00T N1.7
- [ ] LIBERO success rate in `verify.py`
- [ ] Jetson Orin target, then Qualcomm
- [ ] TensorRT / torchao int8 on GPU
- [ ] Public leaderboard from `results/`

Apache-2.0.
