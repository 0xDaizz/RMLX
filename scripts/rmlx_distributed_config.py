#!/usr/bin/env python3
"""Configure and generate hostfiles for distributed RMLX runs.

This script is intentionally modeled after MLX's distributed helpers:
- validates SSH reachability for each host
- collects control-plane IPs
- probes RDMA devices (ibv_devices)
- optionally runs baseline host-side setup commands
- writes a JSON hostfile for launch tools
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class HostInfo:
    ssh: str
    ip: str
    rdma_devices: List[str]


def _run_local(cmd: str, timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", cmd],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _run_remote(
    host: str,
    cmd: str,
    *,
    user: Optional[str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    if host in {"localhost", "127.0.0.1"}:
        return _run_local(cmd, timeout=timeout)

    target = f"{user}@{host}" if user else host
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            target,
            f"bash -lc {shlex.quote(cmd)}",
        ],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _must_ok(result: subprocess.CompletedProcess[str], context: str) -> str:
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"{context}: {detail}")
    return result.stdout.strip()


def _probe_control_ip(
    host: str,
    *,
    iface: str,
    user: Optional[str],
    timeout: int,
) -> str:
    out = _must_ok(
        _run_remote(
            host,
            f"ipconfig getifaddr {shlex.quote(iface)}",
            user=user,
            timeout=timeout,
        ),
        f"failed to query control IP on {host}",
    )
    if not out:
        raise RuntimeError(f"failed to query control IP on {host}: empty output")
    return out.splitlines()[0].strip()


def _probe_rdma_devices(
    host: str,
    *,
    user: Optional[str],
    timeout: int,
) -> List[str]:
    out = _must_ok(
        _run_remote(
            host,
            "ibv_devices | awk 'NR>2 && NF>0 {print $1}'",
            user=user,
            timeout=timeout,
        ),
        f"failed to probe RDMA devices on {host}",
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def _verify_ssh(host: str, *, user: Optional[str], timeout: int) -> None:
    _must_ok(
        _run_remote(host, "echo rmlx-ssh-ok", user=user, timeout=timeout),
        f"ssh probe failed on {host}",
    )


def _auto_setup_host(host: str, *, user: Optional[str], timeout: int, verbose: bool) -> None:
    # Conservative setup: disable thunderbolt bridge where present.
    cmd = (
        "set -euo pipefail; "
        "if networksetup -listallnetworkservices | grep -qx 'Thunderbolt Bridge'; then "
        "sudo networksetup -setnetworkserviceenabled 'Thunderbolt Bridge' off || true; "
        "fi; "
        "sudo ifconfig bridge0 down 2>/dev/null || true"
    )
    if verbose:
        print(f"[{host}] auto-setup: disable Thunderbolt Bridge")
    _must_ok(
        _run_remote(host, cmd, user=user, timeout=timeout),
        f"auto-setup failed on {host}",
    )


def _build_rdmamap(devs: List[str], rank: int, world_size: int) -> List[Optional[str]]:
    row: List[Optional[str]] = []
    cursor = 0
    for peer in range(world_size):
        if peer == rank:
            row.append(None)
            continue
        row.append(devs[cursor % len(devs)])
        cursor += 1
    return row


def _parse_hosts(value: str) -> List[str]:
    hosts = [h.strip() for h in value.split(",") if h.strip()]
    if not hosts:
        raise argparse.ArgumentTypeError("at least one host is required")
    return hosts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate hostfile + optional baseline setup for RMLX distributed runs."
    )
    parser.add_argument(
        "--hosts",
        required=True,
        help="Comma-separated hostnames or IPs (e.g. node1,node2).",
    )
    parser.add_argument(
        "--backend",
        choices=["rdma", "ring"],
        default="rdma",
        help="Distributed backend profile for hostfile generation.",
    )
    parser.add_argument(
        "--over",
        choices=["thunderbolt", "ethernet"],
        default="thunderbolt",
        help="Control-plane topology hint.",
    )
    parser.add_argument(
        "--control-iface",
        default="en0",
        help="Interface used to resolve host control IP (default: en0).",
    )
    parser.add_argument(
        "--auto-setup",
        action="store_true",
        help="Run baseline host setup commands remotely (requires passwordless sudo).",
    )
    parser.add_argument(
        "--no-verify-rdma",
        action="store_true",
        help="Skip RDMA device probe validation (hostfile still generated).",
    )
    parser.add_argument(
        "--output",
        default="rmlx-hosts.json",
        help="Output hostfile path (default: rmlx-hosts.json).",
    )
    parser.add_argument("--ssh-user", help="SSH user override (default: current user).")
    parser.add_argument("--timeout", type=int, default=20, help="Per-host command timeout seconds.")
    parser.add_argument("--verbose", action="store_true", help="Verbose progress logs.")
    args = parser.parse_args()

    hosts = _parse_hosts(args.hosts)
    infos: List[HostInfo] = []

    try:
        for host in hosts:
            if args.verbose:
                print(f"[{host}] probing ssh")
            _verify_ssh(host, user=args.ssh_user, timeout=args.timeout)

            if args.auto_setup and args.over == "thunderbolt":
                _auto_setup_host(
                    host,
                    user=args.ssh_user,
                    timeout=args.timeout,
                    verbose=args.verbose,
                )

            if args.verbose:
                print(f"[{host}] probing control IP via {args.control_iface}")
            ip = _probe_control_ip(
                host,
                iface=args.control_iface,
                user=args.ssh_user,
                timeout=args.timeout,
            )

            rdma_devs: List[str] = []
            if args.backend == "rdma" and not args.no_verify_rdma:
                if args.verbose:
                    print(f"[{host}] probing RDMA devices (ibv_devices)")
                rdma_devs = _probe_rdma_devices(host, user=args.ssh_user, timeout=args.timeout)
                if not rdma_devs:
                    raise RuntimeError(
                        f"no RDMA devices found on {host}; use --no-verify-rdma to bypass"
                    )

            infos.append(HostInfo(ssh=host, ip=ip, rdma_devices=rdma_devs))

        world = len(infos)
        hostfile = []
        for rank, info in enumerate(infos):
            entry = {"ssh": info.ssh, "ips": [info.ip]}
            if args.backend == "rdma":
                if not args.no_verify_rdma:
                    needed = world - 1
                    if len(info.rdma_devices) < needed:
                        raise RuntimeError(
                            f"{info.ssh}: requires >= {needed} RDMA devices for full mesh, "
                            f"found {len(info.rdma_devices)} ({info.rdma_devices})"
                        )
                entry["rdma"] = _build_rdmamap(info.rdma_devices or ["rdma_device_todo"], rank, world)
            hostfile.append(entry)

        out_path = Path(args.output)
        out_path.write_text(json.dumps(hostfile, indent=2), encoding="utf-8")
        print(f"wrote hostfile: {out_path}")
        print(f"hosts: {', '.join(hosts)}")
        print(f"backend: {args.backend}  over: {args.over}")
        print(
            "next: python3 scripts/rmlx_launch.py "
            f"--backend {args.backend} --hostfile {shlex.quote(str(out_path))} -- <command>"
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
