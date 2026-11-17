"""Cost ledger on the volume, with a hard cap.

Each container writes its own file under <root>/ledger and rewrites it every 30 s, so a killed
container leaves at most 30 s unrecorded. The total spend is the sum over all files. Rates are
Modal's list prices (`modal billing rates`, 2026-11-17).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

CAP_USD = 40.0
GPU_RATE = {None: 0.0, "H100": 3.95, "H200": 4.54, "B200": 6.25, "A100-80GB": 2.50}  # $/GPU-hour
CPU_RATE = 0.0473  # $/core-hour
MEM_RATE = 0.008  # $/GiB-hour
START_PAD_S = 60  # container start and image pull, not visible to the code


def rate_per_s(gpu: str | None, cpu: float, mem_gib: float) -> float:
    return (GPU_RATE[gpu] + cpu * CPU_RATE + mem_gib * MEM_RATE) / 3600


def read_all(root: Path) -> list[dict]:
    d = Path(root) / "ledger"
    out = []
    if d.exists():
        for p in sorted(d.glob("*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                pass
    return out


def total(records: list[dict]) -> float:
    return sum(r["usd"] for r in records)


def by_stage(records: list[dict]) -> dict:
    out: dict = {}
    for r in records:
        s = out.setdefault(r["stage"], {"usd": 0.0, "gpu_hours": 0.0, "containers": 0})
        s["usd"] += r["usd"]
        s["gpu_hours"] += r["seconds"] / 3600 if r["gpu"] else 0.0
        s["containers"] += 1
    return out


class OverCap(RuntimeError):
    pass


def check(spent: float, estimate: float, stage: str) -> None:
    if spent + estimate > CAP_USD:
        raise OverCap(f"{stage}: spent ${spent:.2f} + estimate ${estimate:.2f} > cap ${CAP_USD:.2f}")


class Meter:
    """Records this container's running cost. ``guard()`` is the in-container hard cap: it assumes
    ``n_parallel`` containers of this stage run as long as this one."""

    def __init__(self, root: Path, stage: str, gpu: str | None, cpu: float, mem_gib: float,
                 n_parallel: int = 1, heartbeat_s: float = 30.0):
        self.root = Path(root)
        self.stage = stage
        self.gpu, self.cpu, self.mem_gib = gpu, cpu, mem_gib
        self.n_parallel = n_parallel
        self.rate = rate_per_s(gpu, cpu, mem_gib)
        self.t0 = time.time()
        # everything recorded before this container, except same-stage containers still running
        # (those are covered by n_parallel)
        self.base = total([r for r in read_all(self.root)
                           if r["stage"] != stage or r["last"] < self.t0 - 120])
        task = os.environ.get("MODAL_TASK_ID", uuid.uuid4().hex[:12])
        self.path = self.root / "ledger" / f"{stage}-{task}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self.write()
        self._thread = threading.Thread(target=self._beat, args=(heartbeat_s,), daemon=True)
        self._thread.start()

    def seconds(self) -> float:
        return time.time() - self.t0 + START_PAD_S

    def usd(self) -> float:
        return self.seconds() * self.rate

    def write(self) -> None:
        rec = {"stage": self.stage, "gpu": self.gpu, "cpu": self.cpu, "mem_gib": self.mem_gib,
               "start": self.t0, "last": time.time(), "seconds": self.seconds(), "usd": self.usd()}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, self.path)

    def _beat(self, every: float) -> None:
        while not self._stop.wait(every):
            try:
                self.write()
            except OSError:
                pass

    def projected(self) -> float:
        return self.base + self.n_parallel * self.usd()

    def guard(self) -> bool:
        """True while the projected total stays under the cap."""
        return self.projected() < CAP_USD

    def close(self) -> None:
        self._stop.set()
        self.write()
