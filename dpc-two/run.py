import argparse

import torch
from optimizer import make_step_fn
from policy import load_policy, make_observation
from profiler import profile
from report import print_table, save_json
from verify import drift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--variants", default="baseline,bf16")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    device = args.device
    model = load_policy(device)  # 1. policy
    image, state = make_observation(device)
    baseline_fn = make_step_fn(model, "baseline", device)

    rows = []
    for variant in args.variants.split(","):
        act_fn = make_step_fn(model, variant, device)  # 3. optimize
        stats = profile(
            lambda fn=act_fn: fn(image, state), device, steps=args.steps
        )  # 2. profile
        stats["drift"] = (
            None if variant == "baseline" else drift(baseline_fn, act_fn, device)
        )  # 4. verify
        stats["variant"] = variant
        rows.append(stats)

    print_table(rows)  # 5. report
    save_json(rows, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
