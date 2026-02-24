import argparse

import numpy as np


def _load_txt(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except ValueError:
                continue
    if not vals:
        return np.zeros((0,), dtype=np.float64)
    return np.asarray(vals, dtype=np.float64)


def _ascii_hist(samples: np.ndarray, bins: np.ndarray):
    counts, edges = np.histogram(samples, bins=bins)
    max_count = int(np.max(counts)) if counts.size > 0 else 1
    print("step_time_ms histogram")
    for i, c in enumerate(counts):
        left = edges[i]
        right = edges[i + 1]
        if right >= 1e8:
            label = f"[{left:6.1f}, inf)"
        else:
            label = f"[{left:6.1f},{right:6.1f})"
        bar_len = int(round((40.0 * c) / max(1, max_count)))
        bar = "#" * bar_len
        print(f"{label} {c:6d} {bar}")


def main():
    ap = argparse.ArgumentParser(description="Plot timing histogram from txt samples (ms per line).")
    ap.add_argument("--txt", default="/tmp/tof_step_time_ms.txt", help="Input txt path, default /tmp/tof_step_time_ms.txt")
    ap.add_argument("--png", default="", help="Optional output png path (requires matplotlib).")
    args = ap.parse_args()

    samples = _load_txt(args.txt)
    if samples.size == 0:
        print(f"No timing samples found in {args.txt}")
        return

    bins = np.asarray([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 33.3, 50.0, 100.0, 200.0, 1e9], dtype=np.float64)
    _ascii_hist(samples, bins)

    mean_ms = float(np.mean(samples))
    p50_ms = float(np.percentile(samples, 50))
    p90_ms = float(np.percentile(samples, 90))
    p99_ms = float(np.percentile(samples, 99))
    max_ms = float(np.max(samples))
    fps = (1000.0 / mean_ms) if mean_ms > 1e-9 else 0.0
    print(
        f"samples={samples.size} mean={mean_ms:.2f}ms p50={p50_ms:.2f}ms "
        f"p90={p90_ms:.2f}ms p99={p99_ms:.2f}ms max={max_ms:.2f}ms approx_fps={fps:.1f}"
    )

    if args.png:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"matplotlib unavailable, skipping PNG: {e}")
            return

        plt.figure(figsize=(8, 4.5), dpi=120)
        finite_max = max(200.0, float(np.percentile(samples, 99.5)))
        plot_bins = np.linspace(0.0, finite_max, 40)
        plt.hist(samples, bins=plot_bins, color="#1f77b4", edgecolor="white")
        plt.xlabel("Step time (ms)")
        plt.ylabel("Count")
        plt.title("ToF Ground Segmentation Step Time Histogram (from TXT)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(args.png)
        print(f"Wrote PNG: {args.png}")


if __name__ == "__main__":
    main()
