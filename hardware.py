"""Linux sysfs topology discovery. No workload or performance measurement."""
import os
from pathlib import Path


def parse_cpu_list(text):
    result = []
    for part in text.split(","):
        bounds = part.split("-")
        if len(bounds) == 1:
            result.append(int(part))
        elif len(bounds) == 2:
            start, end = map(int, bounds)
            if end < start:
                raise ValueError("CPU range must be ascending")
            result.extend(range(start, end + 1))
        else:
            raise ValueError("Invalid CPU range")
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise ValueError("CPU list must be nonempty, nonnegative and unique")
    return result


def topology():
    if not hasattr(os, "sched_getaffinity"):
        raise ValueError("Automatic topology discovery requires Linux")
    rows = []
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        read = lambda path: (base / path).read_text().strip()
        caches = []
        for index in sorted((base / "cache").glob("index*")):
            caches.append({name: (index / name).read_text().strip()
                           for name in ("level", "type", "size", "coherency_line_size",
                                        "shared_cpu_list")})
        l3 = next(c["shared_cpu_list"] for c in caches if c["level"] == "3")
        rows.append(dict(cpu=cpu, socket=int(read("topology/physical_package_id")),
                         core=int(read("topology/core_id")),
                         siblings=read("topology/thread_siblings_list"),
                         l3=l3, caches=caches,
                         governor=read("cpufreq/scaling_governor")
                         if (base / "cpufreq/scaling_governor").exists() else "unknown"))
    return rows


def select_cpus(spec, rows, required):
    if spec == "physical":
        selected, seen = [], set()
        # Group by L3, then physical core; use one available SMT sibling per core.
        for row in sorted(rows, key=lambda r: (
                r["socket"], min(parse_cpu_list(r["l3"])), r["core"], r["cpu"])):
            key = row["socket"], row["core"]
            if key not in seen:
                selected.append(row["cpu"])
                seen.add(key)
    else:
        selected = parse_cpu_list(spec)
    by_cpu = {r["cpu"]: r for r in rows}
    if len(selected) < required:
        raise ValueError(f"Need {required} CPUs, only {len(selected)} selected")
    if any(c not in by_cpu for c in selected):
        raise ValueError("Selected CPUs must be within the current process affinity")
    cores = [(by_cpu[c]["socket"], by_cpu[c]["core"]) for c in selected]
    if len(cores) != len(set(cores)):
        raise ValueError("This physical-core experiment forbids using SMT siblings together")
    if any(int(cache["coherency_line_size"]) != 64
           for c in selected for cache in by_cpu[c]["caches"]):
        raise ValueError("The executable layout requires 64-byte cache lines")
    return selected
