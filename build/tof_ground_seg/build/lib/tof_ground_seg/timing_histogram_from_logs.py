import argparse
import glob
import os
import re
from typing import List

import numpy as np


SAMPLE_RE = re.compile(r"step_time_ms_sample=([0-9]+(?:\.[0-9]+)?)")


def _load_samples(log_dir: str) -> np.ndarray:
    vals: List[float] = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = SAMPLE_RE.search(line)
                if m:
                    vals.append(float(m.group(1)))
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
    ap = argparse.ArgumentParser(description="Generate step-time histogram from ROS logs.")
    ap.add_argument(
        "--log-dir",
        default=os.path.expanduser("~/.ros/log/latest"),
        help="Directory containing ROS .log files (default: ~/.ros/log/latest)",
    )
    ap.add_argument(
        "--png",
        default="",
        help="Optional output PNG path for plotted histogram (requires matplotlib).",
    )
    args = ap.parse_args()

    bins = np.asarray([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 33.3, 50.0, 100.0, 200.0, 1e9], dtype=np.float64)
    samples = _load_samples(args.log_dir)
    if samples.size == 0:
        print(f"No timing samples found in {args.log_dir}.")
        print("Set parameter `timing_log_each_step:=true`, run the node, then retry.")
        return

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
        plt.title("ToF Ground Segmentation Step Time Histogram")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(args.png)
        print(f"Wrote PNG: {args.png}")


if __name__ == "__main__":
    main()
