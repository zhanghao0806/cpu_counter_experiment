#!/usr/bin/env python3
"""Summarize the two experiments; plot elapsed time and Global single-thread speedup."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

COLUMNS = ["family", "mode", "threads", "events", "samples", "median_ms",
           "min_ms", "max_ms", "speedup_over_single_thread"]


def summarize(rows):
    if not rows:
        raise ValueError("raw_results.csv is empty")
    groups = defaultdict(list)
    checksums = {}
    events_seen = set()
    repeats = set()
    allowed = {("experiment_1", "global"), ("experiment_2", "compact"),
               ("experiment_2", "padded")}
    for row in rows:
        family, mode = row["family"], row["mode"]
        if (family, mode) not in allowed:
            raise ValueError("Expected the new two-experiment CSV; legacy main/batch data are not accepted")
        t, events, rep = int(row["threads"]), int(row["events"]), int(row["repetition"])
        seconds = float(row["seconds"])
        if (t < 1 or events < t or rep < 1 or int(row["batch"]) != 1
                or int(row["correct"]) != 1 or int(row["actual"]) != events
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("Invalid or incorrect measurement in CSV")
        trial = family, mode, t, rep
        if trial in repeats:
            raise ValueError(f"Duplicate trial: {trial}")
        repeats.add(trial)
        events_seen.add(events)
        if t in checksums and checksums[t] != row["checksum"]:
            raise ValueError("Checksum mismatch at the same thread count")
        checksums[t] = row["checksum"]
        groups[family, mode, t].append((rep, seconds * 1000))
    if len(events_seen) != 1:
        raise ValueError("All trials must use the same total workload")
    global_threads = {t for family, mode, t in groups if family == "experiment_1"}
    if 1 not in global_threads:
        raise ValueError("Global with 1 thread is required for speedup")
    for mode in ("compact", "padded"):
        if {t for family, m, t in groups if m == mode} != global_threads:
            raise ValueError("Global, Compact and Padded must have matching thread counts")
    repetition_sets = {tuple(sorted(rep for rep, _ in items)) for items in groups.values()}
    if len(repetition_sets) != 1:
        raise ValueError("Every configuration must have the same repetitions; CSV may be incomplete")
    repetition_ids = next(iter(repetition_sets))
    if repetition_ids != tuple(range(1, len(repetition_ids) + 1)):
        raise ValueError("Repetitions must be consecutive, starting at 1")
    baseline = median(ms for _, ms in groups["experiment_1", "global", 1])
    summary = []
    for (family, mode, t), items in sorted(groups.items()):
        times = [ms for _, ms in items]
        mid = median(times)
        summary.append(dict(family=family, mode=mode, threads=t,
                            events=next(iter(events_seen)), samples=len(items),
                            median_ms=mid, min_ms=min(times), max_ms=max(times),
                            speedup_over_single_thread=baseline / mid
                            if family == "experiment_1" else ""))
    return summary


def draw_figures(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global_data = [r for r in summary if r["family"] == "experiment_1"]
    xs = [r["threads"] for r in global_data]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    def time_curve(ax, points, label):
        medians = [r["median_ms"] for r in points]
        ax.errorbar([r["threads"] for r in points], medians,
                    yerr=[[r["median_ms"] - r["min_ms"] for r in points],
                          [r["max_ms"] - r["median_ms"] for r in points]],
                    marker="o", capsize=4, linewidth=1.8, label=label)
        ax.set_ylabel("Elapsed time (ms), lower is better")
        ax.set_ylim(bottom=0)

    time_curve(axes[0], global_data, "Global: median, min-max range")
    axes[0].set_title("Experiment 1: Global elapsed time")
    axes[1].plot(xs, [r["speedup_over_single_thread"] for r in global_data],
                 marker="o", label="Global: t(1) / t(T)")
    axes[1].axhline(1, color="gray", linestyle="--", label="Single-thread performance")
    axes[1].set_ylabel("Speedup over Global with 1 thread")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Experiment 1: Global speedup")
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xlabel("Worker threads (one per physical core)")
        ax.grid(True, axis="y", alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "01_global_scaling.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.1))
    for mode, label in (("compact", "Compact: 8 counters per 64 B line"),
                        ("padded", "Padded: 1 counter per 64 B line")):
        points = [r for r in summary if r["mode"] == mode]
        time_curve(ax, points, label)
    ax.set_xticks(xs)
    ax.set_xlabel("Worker threads (compare layouts at the same thread count)")
    ax.set_title("Experiment 2: Layout comparison\nMedian elapsed time; error bars show min-max range")
    ax.grid(True, axis="y", alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "02_layout_comparison.png", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/raw_results.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    with Path(args.csv).open(encoding="utf-8", newline="") as file:
        summary = summarize(list(csv.DictReader(file)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for filename, subset in (
            ("summary.csv", summary),
            ("experiment_1_global.csv", [r for r in summary if r["family"] == "experiment_1"]),
            ("experiment_2_layout.csv", [r for r in summary if r["family"] == "experiment_2"])):
        with (out / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(subset)
    try:
        draw_figures(summary, out)
    except ImportError:
        print("matplotlib unavailable: summary CSVs created; figures skipped")
        return
    print(f"Saved summaries and two experiment figures to {out.resolve()}")


if __name__ == "__main__":
    main()
