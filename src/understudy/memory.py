"""How much memory a model could take right now, read from the operating system.

On macOS that is the pages the kernel can hand out without pushing a running program into swap:
free, speculative, purgeable and inactive. Compressed and wired memory are not counted, because
getting them back means swapping something the person is using. On Linux it is MemAvailable.

Every number here is bytes. A reading that cannot be taken is None, never zero: "we could not
look" and "nothing is free" lead to different decisions.
"""
from __future__ import annotations

import re
import subprocess
import sys

MAC_PAGES = ("Pages free", "Pages speculative", "Pages purgeable", "Pages inactive")


def parse_vm_stat(text: str) -> int | None:
    """Available bytes from `vm_stat` output, or None when it does not look like vm_stat."""
    m = re.search(r"page size of (\d+) bytes", text)
    if not m:
        return None
    page = int(m.group(1))
    counts = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        v = v.strip().rstrip(".")
        if v.isdigit():
            counts[k.strip()] = int(v)
    if not all(k in counts for k in MAC_PAGES):
        return None
    return sum(counts[k] for k in MAC_PAGES) * page


def parse_meminfo(text: str) -> int | None:
    m = re.search(r"^MemAvailable:\s+(\d+) kB", text, re.M)
    return int(m.group(1)) * 1024 if m else None


def available(run=subprocess.run, platform: str = sys.platform) -> int | None:
    try:
        if platform == "darwin":
            r = run(["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=10)
            return parse_vm_stat(r.stdout) if r.returncode == 0 else None
        with open("/proc/meminfo") as f:
            return parse_meminfo(f.read())
    except (OSError, subprocess.SubprocessError):
        return None


def total(run=subprocess.run, platform: str = sys.platform) -> int | None:
    try:
        if platform == "darwin":
            r = run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=10)
            return int(r.stdout.strip()) if r.returncode == 0 else None
        with open("/proc/meminfo") as f:
            m = re.search(r"^MemTotal:\s+(\d+) kB", f.read(), re.M)
            return int(m.group(1)) * 1024 if m else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


GB = 1024 ** 3


def gb(n: int | None) -> float | None:
    return None if n is None else round(n / GB, 1)
