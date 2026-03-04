from __future__ import annotations

import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _run_cmd(args: list[str]) -> str | None:
    try:
        out = subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    s = out.strip()
    return s or None


def _read_first_match(path: Path, prefix: str) -> str | None:
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
    except Exception:
        return None
    return None


def _strip_os_release_value(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        return v[1:-1]
    return v


def _linux_pretty_name() -> str | None:
    p = Path("/etc/os-release")
    if not p.exists():
        return None
    try:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("PRETTY_NAME="):
                return _strip_os_release_value(line.split("=", 1)[1])
    except Exception:
        return None
    return None


def _darwin_pretty_name() -> str | None:
    name = _run_cmd(["sw_vers", "-productName"])
    ver = _run_cmd(["sw_vers", "-productVersion"])
    build = _run_cmd(["sw_vers", "-buildVersion"])
    if name and ver and build:
        return f"{name} {ver} ({build})"
    if name and ver:
        return f"{name} {ver}"
    return None


def _windows_total_ram_bytes() -> int | None:
    try:
        import ctypes
    except Exception:
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint),
            ("dwMemoryLoad", ctypes.c_uint),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) == 0:
            return None
    except Exception:
        return None
    return int(stat.ullTotalPhys)


def _os_pretty_name(system: str) -> str | None:
    if system == "Darwin":
        return _darwin_pretty_name()
    if system == "Linux":
        return _linux_pretty_name()
    if system == "Windows":
        rel = platform.release()
        ver = platform.version()
        if rel and ver:
            return f"Windows {rel} ({ver})"
        if rel:
            return f"Windows {rel}"
    return None


def collect_machine_specs() -> dict[str, Any]:
    """
    Capture lightweight machine + runtime info for benchmark evidence (stdlib only).
    Always returns non-empty, best-effort values where possible.
    """
    system = platform.system()

    cpu_brand: str | None = None
    ram_bytes: int | None = None

    if system == "Darwin":
        cpu_brand = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if mem is not None:
            try:
                ram_bytes = int(mem)
            except ValueError:
                ram_bytes = None

    elif system == "Linux":
        cpu_brand = _read_first_match(Path("/proc/cpuinfo"), "model name")
        mem_kb = _read_first_match(Path("/proc/meminfo"), "MemTotal")
        if mem_kb is not None and mem_kb.lower().endswith("kb"):
            try:
                ram_bytes = int(mem_kb.lower().replace("kb", "").strip()) * 1024
            except ValueError:
                ram_bytes = None

    elif system == "Windows":
        cpu_brand = platform.processor() or None
        ram_bytes = _windows_total_ram_bytes()

    else:
        cpu_brand = platform.processor() or None

    cpu_brand = cpu_brand or platform.processor() or "unknown"

    platform_str = platform.platform() or "unknown"
    os_pretty = _os_pretty_name(system) or platform_str

    specs: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "system": system,
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform_str,
        "os_pretty": os_pretty,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_brand": cpu_brand,
    }

    if ram_bytes is not None:
        specs["ram_total_bytes"] = ram_bytes
        specs["ram_total_gb"] = round(ram_bytes / (1024**3), 2)

    return specs
