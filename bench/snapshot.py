"""Machine state record, written next to every measurement.

Everything here is read without sudo. Two levels:

- ``full_snapshot()``: once per session, includes a 60 s idle CPU sample.
- ``short_snapshot()``: before and after each benchmark round.

``discard_reasons(before, after)`` applies the plan's discard rules to a round.

CLI: ``python -m bench.snapshot [--idle 60] [--tag name]`` writes
``results/<timestamp>[-tag]/env.json`` and prints its path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
LLAMA_CPP = Path.home() / "Code" / "llama.cpp"

MIN_FREE_DISK_GB = 5.0
MIN_CPU_IDLE = 90.0


def _run(cmd: list[str], timeout: float = 120) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:  # noqa: BLE001 - a missing tool must not kill a bench
        return f"<error: {e}>"


def _signed64(v: int) -> int:
    return v - (1 << 64) if v >= (1 << 63) else v


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- pieces

def power() -> dict:
    batt = _run(["pmset", "-g", "batt"])
    pm = _run(["pmset", "-g"])
    src = re.search(r"Now drawing from '([^']+)'", batt)
    pct = re.search(r"(\d+)%;\s*([^;]+);", batt)
    lpm = re.search(r"lowpowermode\s+(\d)", pm)
    return {
        "source": src.group(1) if src else None,
        "on_charger": bool(src and src.group(1) == "AC Power"),
        "charge_pct": int(pct.group(1)) if pct else None,
        "battery_state": pct.group(2).strip() if pct else None,
        "low_power_mode": bool(lpm and lpm.group(1) == "1"),
    }


def thermal() -> dict:
    out = _run(["pmset", "-g", "therm"])
    lim = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    # "No CPU power status has been recorded" means no limit is applied.
    return {"cpu_speed_limit": int(lim.group(1)) if lim else 100, "raw": out.strip()}


def cpu(samples: int = 3, interval: int = 1) -> dict:
    """CPU idle % from top. The first top sample is since-boot, so it is dropped."""
    out = _run(["top", "-l", str(samples + 1), "-s", str(interval), "-n", "0"],
               timeout=samples * interval + 30)
    idles = [float(x) for x in re.findall(r"([\d.]+)% idle", out)][1:]
    load = _run(["sysctl", "-n", "vm.loadavg"]).strip().strip("{} ").split()
    return {
        "idle_mean": round(sum(idles) / len(idles), 2) if idles else None,
        "idle_min": min(idles) if idles else None,
        "busy_max": round(100 - min(idles), 2) if idles else None,
        "samples": len(idles),
        "interval_s": interval,
        "loadavg": [float(x) for x in load[:3]] if load else None,
    }


def memory() -> dict:
    mp = _run(["memory_pressure"])
    free = re.search(r"free percentage:\s*(\d+)%", mp)
    swap = _run(["sysctl", "-n", "vm.swapusage"])
    sw = dict(re.findall(r"(total|used|free) = ([\d.]+)M", swap))
    vm = _run(["vm_stat"])
    page = re.search(r"page size of (\d+) bytes", vm)
    pages = {k.strip().lower().replace(" ", "_").replace('"', ""): int(v)
             for k, v in re.findall(r"^([^:\n]+):\s+(\d+)\.", vm, re.M)}
    psize = int(page.group(1)) if page else 16384
    gb = lambda n: round(n * psize / 2**30, 2)  # noqa: E731
    return {
        "free_pct": int(free.group(1)) if free else None,
        "swap_used_mb": float(sw["used"]) if "used" in sw else None,
        "swap_total_mb": float(sw["total"]) if "total" in sw else None,
        "wired_gb": gb(pages.get("pages_wired_down", 0)),
        "compressed_gb": gb(pages.get("pages_occupied_by_compressor", 0)),
        "free_gb": gb(pages.get("pages_free", 0)),
    }


def disk() -> dict:
    u = shutil.disk_usage("/System/Volumes/Data")
    return {"free_gb": round(u.free / 2**30, 1), "total_gb": round(u.total / 2**30, 1)}


def gpu() -> dict:
    out = _run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])
    m = re.search(r'"PerformanceStatistics" = \{([^}]*)\}', out)
    if not m:
        return {}
    stats = dict(re.findall(r'"([^"]+)"=(\d+)', m.group(1)))
    keep = {
        "Device Utilization %": "device_util_pct",
        "Renderer Utilization %": "renderer_util_pct",
        "Tiler Utilization %": "tiler_util_pct",
        "In use system memory": "in_use_mem_bytes",
        "Alloc system memory": "alloc_mem_bytes",
    }
    return {v: int(stats[k]) for k, v in keep.items() if k in stats}


def battery_power() -> dict:
    out = _run(["ioreg", "-rn", "AppleSmartBattery"])
    res = {}
    for key in ("SystemLoad", "SystemPowerIn", "BatteryPower", "SystemVoltageIn", "SystemCurrentIn"):
        m = re.search(rf'"{key}"=(\d+)', out)
        if m:
            res[key] = _signed64(int(m.group(1)))
    for key in ("Voltage", "InstantAmperage", "CycleCount", "DesignCapacity", "NominalChargeCapacity"):
        m = re.search(rf'"{key}" = (\d+)', out)
        if m:
            res[key] = _signed64(int(m.group(1)))
    return res


def top_processes(n: int = 5) -> dict:
    def parse(sort_flag: str) -> list[dict]:
        out = _run(["ps", "-Ao", "pcpu,pmem,rss,comm", sort_flag]).splitlines()[1:n + 1]
        rows = []
        for line in out:
            parts = line.split(None, 3)
            if len(parts) == 4:
                rows.append({"cpu": float(parts[0]), "mem": float(parts[1]),
                             "rss_mb": round(int(parts[2]) / 1024, 1),
                             "comm": parts[3].rsplit("/", 1)[-1]})
        return rows
    return {"by_cpu": parse("-r"), "by_mem": parse("-m")}


def hardware() -> dict:
    hw = _run(["system_profiler", "SPHardwareDataType"])
    pw = _run(["system_profiler", "SPPowerDataType"])
    get = lambda pat, s: (m.group(1).strip() if (m := re.search(pat, s)) else None)  # noqa: E731
    return {
        "model_name": get(r"Model Name:\s*(.+)", hw),
        "model_id": get(r"Model Identifier:\s*(.+)", hw),
        "chip": get(r"Chip:\s*(.+)", hw),
        "cores": get(r"Total Number of Cores:\s*(.+)", hw),
        "memory": get(r"Memory:\s*(.+)", hw),
        "macos": _run(["sw_vers", "-productVersion"]).strip(),
        "macos_build": _run(["sw_vers", "-buildVersion"]).strip(),
        "battery_cycle_count": get(r"Cycle Count:\s*(\d+)", pw),
        "battery_condition": get(r"Condition:\s*(.+)", pw),
        "battery_max_capacity": get(r"Maximum Capacity:\s*(.+)", pw),
    }


def _du_gb(p: Path) -> float | None:
    if not p.exists():
        return None
    out = _run(["du", "-sk", str(p)], timeout=300).split()
    return round(int(out[0]) / 2**20, 2) if out and out[0].isdigit() else None


def software() -> dict:
    import importlib.metadata as md

    vers = {}
    for pkg in ("mlx", "mlx-lm", "mlx-metal", "numpy", "transformers"):
        try:
            vers[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            vers[pkg] = None
    llama = None
    if (LLAMA_CPP / ".git").exists():
        llama = _run(["git", "-C", str(LLAMA_CPP), "log", "-1", "--format=%H %cs %s"]).strip()
    return {"python": platform.python_version(), "packages": vers, "llama_cpp": llama}


def model_info(repo_or_path: str) -> dict:
    """Repo revision, quantisation and weight file hashes for an HF-cached or local model.

    For LFS files the HF cache blob name is the file's sha256, so no hashing is needed.
    """
    p = Path(repo_or_path)
    info: dict = {"model": repo_or_path}
    if not p.exists():
        d = HF_HUB / ("models--" + repo_or_path.replace("/", "--"))
        ref = d / "refs" / "main"
        if ref.exists():
            rev = ref.read_text().strip()
            info["revision"] = rev
            p = d / "snapshots" / rev
    if p.is_dir():
        cfg = p / "config.json"
        if cfg.exists():
            c = json.loads(cfg.read_text())
            info["quantization"] = c.get("quantization") or c.get("quantization_config")
            info["num_hidden_layers"] = c.get("num_hidden_layers")
            info["hidden_size"] = c.get("hidden_size")
            info["vocab_size"] = c.get("vocab_size")
            info["tie_word_embeddings"] = c.get("tie_word_embeddings")
        files = {}
        for f in sorted(p.glob("*.safetensors")) + sorted(p.glob("*.gguf")):
            blob = f.resolve()
            files[f.name] = {"bytes": blob.stat().st_size,
                             "sha256": blob.name if len(blob.name) == 64 else None}
        info["weight_files"] = files
    elif p.is_file():
        info["weight_files"] = {p.name: {"bytes": p.stat().st_size}}
    return info


# ---------------------------------------------------------------- snapshots

def short_snapshot() -> dict:
    return {
        "timestamp": now(),
        "power": power(),
        "thermal_limit": thermal()["cpu_speed_limit"],
        "memory": memory(),
        "cpu": cpu(samples=3, interval=1),
        "disk_free_gb": disk()["free_gb"],
    }


def full_snapshot(idle_seconds: int = 60, models: list[str] | None = None) -> dict:
    t0 = now()
    idle = cpu(samples=max(1, idle_seconds // 2), interval=2)
    return {
        "kind": "session",
        "timestamp": t0,
        "hardware": hardware(),
        "power": power(),
        "battery_telemetry": battery_power(),
        "thermal": thermal(),
        "cpu_idle_sample": idle,
        "top_processes": top_processes(),
        "memory": memory(),
        "gpu_idle": gpu(),
        "disk": {**disk(), "hf_cache_gb": _du_gb(HF_HUB.parent), "repo_gb": _du_gb(REPO)},
        "software": software(),
        "models": [model_info(m) for m in (models or [])],
    }


def discard_reasons(before: dict, after: dict | None = None) -> list[str]:
    """The plan's discard rules. Empty list means the round counts."""
    r = []
    if not before["power"]["on_charger"]:
        r.append("not on charger")
    if before["power"]["low_power_mode"]:
        r.append("low power mode on")
    idle = before["cpu"]["idle_mean"]
    if idle is not None and idle < MIN_CPU_IDLE:
        r.append(f"cpu idle {idle}% < {MIN_CPU_IDLE}% before round")
    if before["thermal_limit"] < 100:
        r.append(f"thermal limit {before['thermal_limit']} before round")
    if before["disk_free_gb"] < MIN_FREE_DISK_GB:
        r.append(f"free disk {before['disk_free_gb']} GB < {MIN_FREE_DISK_GB}")
    if after is not None:
        if after["thermal_limit"] < 100:
            r.append(f"thermal limit {after['thermal_limit']} after round")
        sb, sa = before["memory"]["swap_used_mb"], after["memory"]["swap_used_mb"]
        if sb is not None and sa is not None and sa > sb + 1.0:
            r.append(f"swap grew {sb:.0f} -> {sa:.0f} MB")
        if not after["power"]["on_charger"]:
            r.append("not on charger after round")
    return r


def wait_quiet(min_idle: float = MIN_CPU_IDLE + 2, max_wait_s: int = 1800, log=print) -> float:
    """Block until CPU idle (3 s sample) reaches min_idle, or max_wait_s passes.
    Returns seconds waited. Background daemons (e.g. mediaanalysisd on AC power) come and go."""
    import time

    t0 = time.time()
    while True:
        idle = cpu(samples=3, interval=1)["idle_mean"] or 0.0
        if idle >= min_idle or time.time() - t0 > max_wait_s:
            waited = time.time() - t0
            if waited > 5:
                log(f"waited {waited:.0f}s for a quiet machine (idle {idle}%)")
            return waited
        time.sleep(20)


def new_results_dir(tag: str = "") -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = RESULTS / (stamp + (f"-{tag}" if tag else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_env(d: Path, env: dict) -> Path:
    p = d / "env.json"
    p.write_text(json.dumps(env, indent=2))
    return p


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idle", type=int, default=60, help="idle CPU sample length, seconds")
    ap.add_argument("--tag", default="session")
    ap.add_argument("--model", action="append", default=[])
    a = ap.parse_args(argv)
    env = full_snapshot(a.idle, a.model)
    env["short"] = short_snapshot()
    env["discard_reasons_now"] = discard_reasons(env["short"])
    d = new_results_dir(a.tag)
    p = write_env(d, env)
    print(p)
    print(json.dumps({"power": env["power"], "thermal": env["thermal"]["cpu_speed_limit"],
                      "cpu_idle": env["cpu_idle_sample"], "memory": env["memory"],
                      "discard_reasons_now": env["discard_reasons_now"]}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
