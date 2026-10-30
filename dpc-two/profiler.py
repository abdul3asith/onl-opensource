import time

import numpy as np
import torch


def time_one_step(step_fn, device):
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step_fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    t0 = time.perf_counter()
    step_fn()
    return (time.perf_counter() - t0) * 1000


def profile(step_fn, device, steps=200, warmup=20):
    for _ in range(warmup):
        step_fn()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    timings = np.array([time_one_step(step_fn, device) for _ in range(steps)])
    p50 = np.percentile(timings, 50)
    p99 = np.percentile(timings, 99)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0

    return {
        "p50_ms": float(p50),  # typical step time
        "p99_ms": float(p99),  # worst-case step time (except rare outliers)
        "jitter_ms": float(
            timings.std()
        ),  # how much step times vary: low = steady rhythm
        "hz": float(
            1000 / p99
        ),  # steps per second the robot can count on (uses the worst case)
        "peak_mb": float(peak_mb),
    }
