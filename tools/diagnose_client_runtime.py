#!/usr/bin/env python3
"""Stress the Python paths that failed while creating the client environment.

This script intentionally uses only the Python standard library and pip's
vendored ``packaging`` module so it can run in a partially-created Conda env.
It does not modify the environment.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import resource
import site
import sys
import urllib.parse


def _source_digest(module_name: str) -> tuple[str, str]:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return "<not found>", "<not found>"
    path = Path(spec.origin)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        digest = f"<unreadable: {exc}>"
    return str(path), digest


def _check_urlsplit(iterations: int) -> None:
    for index in range(iterations):
        url = f"https://example.invalid:443/packages/{index}?sha256={index:08x}"
        result = urllib.parse.urlsplit(url)
        if (
            result.scheme != "https"
            or result.netloc != "example.invalid:443"
            or result.path != f"/packages/{index}"
        ):
            raise AssertionError((index, result))


def _check_packaging_version(iterations: int) -> None:
    from pip._vendor.packaging.version import Version

    expected = "1.2.3+cu124"
    for index in range(iterations):
        rendered = str(Version(expected))
        if rendered != expected:
            raise AssertionError((index, rendered))


def _check_memory(memory_mib: int, rounds: int = 4) -> None:
    size = memory_mib * 1024 * 1024
    data = bytearray(size)
    for round_index in range(rounds):
        value = (0x5A + 37 * round_index) & 0xFF
        data[:] = bytes((value,)) * size
        if data.count(value) != size:
            raise AssertionError(f"memory pattern mismatch in round {round_index}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300_000)
    parser.add_argument("--memory-mib", type=int, default=256)
    args = parser.parse_args()
    if args.iterations < 1 or args.memory_mib < 1:
        parser.error("--iterations and --memory-mib must be positive")

    faulthandler.enable(all_threads=True)
    print(f"executable: {sys.executable}", flush=True)
    print(f"python: {sys.version}", flush=True)
    print(f"platform: {platform.platform()}", flush=True)
    print(f"RLIMIT_STACK: {resource.getrlimit(resource.RLIMIT_STACK)}", flush=True)
    print(f"PYTHONNOUSERSITE: {os.environ.get('PYTHONNOUSERSITE')!r}", flush=True)
    print(f"user-site enabled: {site.ENABLE_USER_SITE}", flush=True)
    print(f"pycache prefix: {sys.pycache_prefix!r}", flush=True)

    for module_name in ("urllib.parse", "pip._vendor.packaging.version"):
        path, digest = _source_digest(module_name)
        print(f"{module_name}: {path}", flush=True)
        print(f"{module_name} sha256: {digest}", flush=True)

    print(f"urlsplit stress: {args.iterations} iterations", flush=True)
    _check_urlsplit(args.iterations)
    print(f"packaging.version stress: {args.iterations} iterations", flush=True)
    _check_packaging_version(args.iterations)
    print(f"memory pattern stress: {args.memory_mib} MiB", flush=True)
    _check_memory(args.memory_mib)
    print("CLIENT_RUNTIME_DIAGNOSTIC_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
