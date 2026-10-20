#!/usr/bin/env python3
"""自动运行基准测试；只使用 Python 标准库。"""
import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from hardware import select_cpus, topology

HEADER = ["mode", "threads", "events", "batch", "seconds", "throughput_mops",
          "actual", "correct", "checksum", "repetition", "family"]

def build_cases(thread_counts):
    """Two experiments only; batch=1 is retained for the C++ CSV interface."""
    return ([("global", t, 1, "experiment_1") for t in thread_counts]
            + [(mode, t, 1, "experiment_2")
               for t in thread_counts for mode in ("compact", "padded")])


def call(exe, mode, threads, events, batch, cpus):
    cmd = [str(exe), mode, str(threads), str(events), str(batch),
           ",".join(map(str, cpus[:threads]))]
    output = subprocess.check_output(cmd, text=True).strip()
    vals = output.split(",")
    if len(vals) != 9:
        raise RuntimeError(f"Unexpected output: {output}")
    labels = HEADER[:9]
    row = dict(zip(labels, vals))
    if int(row["correct"]) != 1 or int(row["actual"]) != events:
        raise RuntimeError(f"Incorrect counter: {output}")
    return row


def system_report(args, exe, out, cpus, hardware):
    lines = [
        f"Captured at: {datetime.now().astimezone().isoformat()}",
        f"Platform: {platform.platform()}",
        f"Machine: {platform.machine()}",
        f"Python: {sys.version.splitlines()[0]}",
        f"Logical CPU count (os.cpu_count): {os.cpu_count()}",
        f"Benchmark executable: {exe}",
        f"Thread counts: {args.threads}",
        f"Events per trial (fixed work): {args.events}",
        f"Repeats: {args.repeat}",
        "Experiment 1: Global, elapsed time and speedup relative to Global with 1 thread",
        "Experiment 2: Compact vs Padded elapsed time at matching thread counts",
        "Measurement: steady_clock wall time, threads created before timing; median used in charts",
        "PaddedCounter: alignas(64), sizeof=64; compact: aligned groups of 8 x 8 bytes",
        f"CPU order (one worker per CPU, using first T): {cpus}",
        f"Executable SHA256: {hashlib.sha256(exe.read_bytes()).hexdigest()}",
        f"Preparation only: {args.prepare_only}",
        f"Load averages: {os.getloadavg()}",
    ]
    for cmd in (["lscpu"], ["sysctl", "-n", "hw.physicalcpu"],
                ["sysctl", "-n", "hw.logicalcpu"]):
        if shutil.which(cmd[0]):
            try:
                result = subprocess.run(cmd, text=True, capture_output=True, timeout=5)
                if result.returncode == 0:
                    lines.extend([f"\n$ {' '.join(cmd)}", result.stdout.rstrip()])
            except (OSError, subprocess.TimeoutExpired):
                pass
    for name in ("/proc/meminfo", "/proc/loadavg", "/proc/sys/kernel/perf_event_paranoid",
                 "/sys/devices/system/cpu/cpufreq/boost"):
        path = Path(name)
        if path.is_file():
            lines.extend([f"\n{path}", path.read_text().strip()])
    (out / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "topology.json").write_text(
        json.dumps({"selected_cpus": cpus, "allowed_cpus": hardware,
                    "arguments": vars(args)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    hashes = {}
    for path in Path(__file__).resolve().parent.iterdir():
        if path.suffix in (".cpp", ".py", ".sh", ".txt", ".md"):
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description="Multicore counter benchmark runner")
    p.add_argument("--exe", default="./counter_bench")
    p.add_argument("--threads", default="1,2,4,8,16,32")
    p.add_argument("--events", type=int, default=12_000_000)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--out", default="results")
    p.add_argument("--plan-only", action="store_true",
                   help="Print the two-experiment plan and exit without accessing hardware or running anything")
    p.add_argument("--cpus", default="physical",
                   help="physical: one CPU per core, grouped by L3; or ordered CPU list (0-7,8-15)")
    p.add_argument("--prepare-only", action="store_true",
                   help="Record hardware, verify layout and worker affinity, then exit WITHOUT counting")
    args = p.parse_args()
    thread_counts = [int(x) for x in args.threads.split(",")]
    if not thread_counts or any(x <= 0 for x in thread_counts) or args.repeat < 1:
        p.error("Threads and repeat must be positive")
    if args.events < max(thread_counts):
        p.error("Events must be >= max threads")
    if len(thread_counts) != len(set(thread_counts)):
        p.error("Thread counts must not contain duplicates")
    if 1 not in thread_counts:
        p.error("Include 1 thread for the Global single-thread speedup baseline")
    cases = build_cases(thread_counts)
    if args.plan_only:
        print(f"Fixed total events per trial: {args.events}; repetitions: {args.repeat}")
        for family in ("experiment_1", "experiment_2"):
            selected = [case for case in cases if case[3] == family]
            print(f"{family}: {len(selected) * args.repeat} measured trials")
            for mode, threads, _, _ in selected:
                print(f"  {mode:7s} threads={threads} repeats={args.repeat}")
        print(f"Total: {len(cases) * args.repeat} measured trials; no Batch or batch sweep.")
        print("Plan only: no preparation, warmup, benchmark or output files.")
        return
    exe = Path(args.exe).expanduser().resolve()
    if not exe.is_file():
        p.error(f"Executable not found: {exe}; compile counter_bench.cpp first")
    try:
        hardware = topology()
        cpus = select_cpus(args.cpus, hardware, max(thread_counts))
    except (ValueError, OSError, StopIteration) as error:
        p.error(f"Hardware/CPU validation failed: {error}")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "raw_results.csv").exists():
        p.error("Output already contains raw_results.csv; choose a new directory")
    system_report(args, exe, out, cpus, hardware)
    for threads in thread_counts:
        report = subprocess.check_output(
            [str(exe), "--inspect", str(threads), ",".join(map(str, cpus[:threads]))],
            text=True)
        (out / f"layout_{threads:02d}.txt").write_text(report, encoding="utf-8")
    print(f"Verified 64 B layout and per-worker affinity. CPU order: {cpus}")
    if args.prepare_only:
        print(f"Preparation saved: {out}. No counting, warmup or performance experiments executed.")
        return

    # 一次低成本预热，避免刚开始编译/进程加载影响首轮。
    print("Warming up ...")
    for mode in ("global", "compact", "padded"):
        call(exe, mode, min(2, max(thread_counts)), 200_000, 1, cpus)

    rng = random.Random(20261020)
    csv_path = out / "raw_results.csv"
    checksums = {}
    total = len(cases) * args.repeat
    count = 0
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HEADER)
        writer.writeheader()
        for rep in range(1, args.repeat + 1):
            shuffled = cases[:]
            rng.shuffle(shuffled)
            for mode, threads, batch, family in shuffled:
                result = call(exe, mode, threads, args.events, batch, cpus)
                key = threads
                if key in checksums and result["checksum"] != checksums[key]:
                    raise RuntimeError("Checksum mismatch across variants at same thread count")
                checksums.setdefault(key, result["checksum"])
                result.update({"repetition": rep, "family": family})
                writer.writerow(result)
                file.flush()
                count += 1
                print(f"[{count:3d}/{total}] {family} threads={threads:<2} "
                      f"mode={mode:<7} "
                      f"time={float(result['seconds'])*1000:8.2f}ms "
                      f"throughput={float(result['throughput_mops']):9.2f} Mops/s")
    print(f"\nAll results correct. Saved: {csv_path}")
    print(f"System info: {out / 'environment.txt'}")
    print("Next: python3 plot_results.py --csv results/raw_results.csv --out results")


if __name__ == "__main__":
    main()
