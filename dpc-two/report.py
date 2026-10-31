import json


def print_table(rows):
    header = (
        f"{'variant':<10}{'p50 ms':>10}{'p99 ms':>10}{'jitter':>10}"
        f"{'Hz':>10}{'peak MB':>10}{'drift':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        d = "-" if r["drift"] is None else f"{r['drift']:.4f}"
        print(
            f"{r['variant']:<10}{r['p50_ms']:>10.2f}{r['p99_ms']:>10.2f}"
            f"{r['jitter_ms']:>10.2f}{r['hz']:>10.1f}{r['peak_mb']:>10.0f}{d:>10}"
        )


def save_json(rows, path):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
