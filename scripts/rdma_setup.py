#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
RDMA network setup for Mac Studio Thunderbolt 5 connections.

Configures static IPs, removes interfaces from bridges,
and verifies RDMA connectivity between nodes via SSH.

Usage:
    # Basic two-node setup
    python scripts/rdma_setup.py \\
      --node hwstudio1:en5:10.254.0.5 \\
      --node hwstudio2:en5:10.254.0.6 \\
      --netmask 30

    # With SSH proxy (hwstudio2 reached via hwstudio1)
    python scripts/rdma_setup.py \\
      --node hwstudio1:en5:10.254.0.5 \\
      --node hwstudio2:en5:10.254.0.6 \\
      --netmask 30 \\
      --proxy hwstudio1:hwstudio2

    # Dry-run mode (print commands without executing)
    python scripts/rdma_setup.py \\
      --node hwstudio1:en5:10.254.0.5 \\
      --node hwstudio2:en5:10.254.0.6 \\
      --netmask 30 \\
      --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(text: str) -> str:
    return _c("32", text)


def yellow(text: str) -> str:
    return _c("33", text)


def red(text: str) -> str:
    return _c("31", text)


def bold(text: str) -> str:
    return _c("1", text)


# Status markers
OK = green("✓")
WARN = yellow("⚠")
FAIL = red("✗")

log = logging.getLogger("rdma_setup")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NodeConfig:
    """Parsed --node specification."""

    ssh_host: str
    interface: str
    rdma_ip: str
    proxy: str | None = None  # SSH proxy host, if any

    # Populated during configuration
    bridge_removed: str | None = None
    ip_set: bool = False
    rdma_enabled: bool | None = None
    ibv_devices: list[str] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: str) -> NodeConfig:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --node format: {spec!r}  "
                f"(expected ssh_host:interface:rdma_ip)"
            )
        return cls(ssh_host=parts[0], interface=parts[1], rdma_ip=parts[2])


# ---------------------------------------------------------------------------
# Netmask calculation
# ---------------------------------------------------------------------------


def cidr_to_netmask(prefix_len: int) -> str:
    """Convert CIDR prefix length to dotted-quad netmask."""
    if not 0 <= prefix_len <= 32:
        raise ValueError(f"Invalid CIDR prefix length: {prefix_len}")
    bits = (0xFFFFFFFF << (32 - prefix_len)) & 0xFFFFFFFF
    return ".".join(str((bits >> (8 * i)) & 0xFF) for i in range(3, -1, -1))


# ---------------------------------------------------------------------------
# SSH command execution
# ---------------------------------------------------------------------------


def ssh_run(
    host: str,
    command: str,
    *,
    proxy: str | None = None,
    dry_run: bool = False,
    timeout: int = 10,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute a command on a remote host via SSH.

    If *proxy* is set, the command is tunneled through the proxy host:
        ssh proxy "ssh host 'command'"

    In dry-run mode the command is printed but not executed.
    """
    # Escape single quotes inside the command for safe nesting
    escaped_cmd = command.replace("'", "'\"'\"'")

    if proxy:
        # Double-nested: ssh proxy "ssh host '<cmd>'"
        inner = f"ssh {host} '{escaped_cmd}'"
        ssh_cmd = ["ssh", proxy, inner]
    else:
        ssh_cmd = ["ssh", host, command]

    display = " ".join(ssh_cmd)

    if dry_run:
        print(f"    [dry-run] {display}")
        return subprocess.CompletedProcess(
            args=ssh_cmd, returncode=0, stdout="", stderr=""
        )

    log.debug("Running: %s", display)
    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            log.debug(
                "Command failed (rc=%d): %s\nstderr: %s",
                result.returncode,
                display,
                result.stderr.strip(),
            )
        return result
    except subprocess.TimeoutExpired:
        log.warning("Timeout (%ds) running: %s", timeout, display)
        return subprocess.CompletedProcess(
            args=ssh_cmd, returncode=-1, stdout="", stderr="timeout"
        )


# ---------------------------------------------------------------------------
# Configuration steps
# ---------------------------------------------------------------------------


def step_remove_from_bridge(node: NodeConfig, netmask_bits: int, dry_run: bool) -> None:
    """Check if the interface is in any bridge and remove it."""
    iface = node.interface

    # Scan bridge0 through bridge9
    for i in range(10):
        bridge = f"bridge{i}"
        result = ssh_run(
            node.ssh_host,
            f"ifconfig {bridge} 2>/dev/null",
            proxy=node.proxy,
            dry_run=dry_run,
        )
        if dry_run:
            continue

        if result.returncode != 0:
            continue  # bridge doesn't exist

        # Check if our interface is a member
        # macOS ifconfig shows: "member: en5 ..."
        if re.search(rf"\bmember:\s*{re.escape(iface)}\b", result.stdout):
            print(f"    Removing {iface} from {bridge}...")
            rm_result = ssh_run(
                node.ssh_host,
                f"sudo ifconfig {bridge} deletem {iface}",
                proxy=node.proxy,
                dry_run=dry_run,
            )
            if rm_result.returncode == 0:
                node.bridge_removed = bridge
                print(f"  {OK} Removed {iface} from {bridge}")
                return
            else:
                print(
                    f"  {FAIL} Failed to remove {iface} from {bridge}: "
                    f"{rm_result.stderr.strip()}"
                )
                return

    if dry_run:
        print(f"  {OK} (dry-run) Bridge check skipped")
    else:
        print(f"  {OK} {iface} not in any bridge")


def step_set_static_ip(node: NodeConfig, netmask_bits: int, dry_run: bool) -> None:
    """Assign a static IP to the interface."""
    mask = cidr_to_netmask(netmask_bits)
    cmd = f"sudo ifconfig {node.interface} {node.rdma_ip} netmask {mask}"
    result = ssh_run(
        node.ssh_host,
        cmd,
        proxy=node.proxy,
        dry_run=dry_run,
    )
    if dry_run:
        print(f"  {OK} (dry-run) Static IP: {node.rdma_ip}/{netmask_bits}")
        node.ip_set = True
        return

    if result.returncode == 0:
        node.ip_set = True
        print(f"  {OK} Static IP set: {node.rdma_ip}/{netmask_bits}")
    else:
        print(
            f"  {FAIL} Failed to set IP: {result.stderr.strip()}"
        )


def step_check_rdma(node: NodeConfig, dry_run: bool) -> None:
    """Check RDMA status via rdma_ctl."""
    result = ssh_run(
        node.ssh_host,
        "rdma_ctl status 2>&1",
        proxy=node.proxy,
        dry_run=dry_run,
    )
    if dry_run:
        print(f"  {OK} (dry-run) RDMA status check")
        return

    stdout = result.stdout.strip().lower()
    if result.returncode != 0 or "disabled" in stdout or "not" in stdout:
        node.rdma_enabled = False
        print(f"  {WARN} RDMA: disabled (enable via Recovery Mode)")
    else:
        node.rdma_enabled = True
        print(f"  {OK} RDMA: enabled")


def step_check_ibv_devices(node: NodeConfig, dry_run: bool) -> None:
    """List IBV devices via ibv_devices."""
    result = ssh_run(
        node.ssh_host,
        "ibv_devices 2>&1",
        proxy=node.proxy,
        dry_run=dry_run,
    )
    if dry_run:
        print(f"  {OK} (dry-run) IBV devices check")
        return

    if result.returncode != 0:
        print(f"  {FAIL} IBV devices: none (RDMA disabled)")
        return

    # Parse device names from ibv_devices output
    devices: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        # Skip header lines; device lines typically start with "rdma_"
        if line.startswith("rdma_"):
            devices.append(line.split()[0])

    node.ibv_devices = devices
    if devices:
        print(f"  {OK} IBV devices: {', '.join(devices)}")
    else:
        print(f"  {FAIL} IBV devices: none (RDMA disabled)")


# ---------------------------------------------------------------------------
# Connectivity test
# ---------------------------------------------------------------------------


def ping_test(
    src: NodeConfig,
    dst: NodeConfig,
    dry_run: bool,
) -> tuple[bool, str]:
    """Ping dst's RDMA IP from src. Returns (success, latency_or_error)."""
    cmd = f"ping -c 1 -W 2 {dst.rdma_ip}"
    result = ssh_run(
        src.ssh_host,
        cmd,
        proxy=src.proxy,
        dry_run=dry_run,
        timeout=15,
    )
    if dry_run:
        return True, "dry-run"

    if result.returncode != 0:
        return False, result.stderr.strip() or "no response"

    # Extract round-trip time from ping output
    # macOS: "round-trip min/avg/max/stddev = 0.123/0.123/0.123/0.000 ms"
    m = re.search(r"[\d.]+/([\d.]+)/[\d.]+/[\d.]+ ms", result.stdout)
    if m:
        return True, f"{float(m.group(1)):.2f}ms"
    return True, "ok"


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def configure_nodes(
    nodes: list[NodeConfig],
    netmask_bits: int,
    dry_run: bool,
) -> None:
    """Run full configuration sequence."""
    print()
    print(bold("RDMA Network Setup"))
    print("=" * 18)
    print()

    # Phase 1: Configure each node
    for idx, node in enumerate(nodes, 1):
        proxy_tag = f" [via {node.proxy}]" if node.proxy else ""
        print(
            bold(
                f"[{idx}/{len(nodes)}] Configuring {node.ssh_host} "
                f"({node.interface} -> {node.rdma_ip}/{netmask_bits})"
                f"{proxy_tag}"
            )
        )

        step_remove_from_bridge(node, netmask_bits, dry_run)
        step_set_static_ip(node, netmask_bits, dry_run)
        step_check_rdma(node, dry_run)
        step_check_ibv_devices(node, dry_run)
        print()

    # Phase 2: Connectivity tests (only if >1 node)
    if len(nodes) > 1:
        print(bold("Connectivity Test"))
        print("=" * 17)

        for src in nodes:
            for dst in nodes:
                if src is dst:
                    continue
                ok, latency = ping_test(src, dst, dry_run)
                status = f"{OK} {latency}" if ok else f"{FAIL} {latency}"
                print(f"  {src.ssh_host} -> {dst.ssh_host}: {status}")

        print()

    # Phase 3: Summary
    print(bold("Summary"))
    print("=" * 7)

    for node in nodes:
        rdma_str = "RDMA:enabled" if node.rdma_enabled else "RDMA:disabled"
        if node.rdma_enabled is None:
            rdma_str = "RDMA:unknown"

        if node.rdma_enabled:
            marker = OK
        elif node.rdma_enabled is False:
            marker = WARN
        else:
            marker = yellow("?")

        print(
            f"  {node.ssh_host:<14} {node.interface:<5} "
            f"{node.rdma_ip + '/' + str(netmask_bits):<18} "
            f"{rdma_str:<14} {marker}"
        )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure RDMA networking on remote Mac Studio nodes via SSH.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --node hwstudio1:en5:10.254.0.5 --node hwstudio2:en5:10.254.0.6
  %(prog)s --node hwstudio1:en5:10.254.0.5 --node hwstudio2:en5:10.254.0.6 --proxy hwstudio1:hwstudio2
  %(prog)s --node hwstudio1:en5:10.254.0.5 --node hwstudio2:en5:10.254.0.6 --dry-run
""",
    )
    parser.add_argument(
        "--node",
        action="append",
        required=True,
        metavar="SSH_HOST:IFACE:RDMA_IP",
        help="Node specification (can be repeated). Format: ssh_host:interface:rdma_ip",
    )
    parser.add_argument(
        "--proxy",
        action="append",
        default=[],
        metavar="PROXY_HOST:TARGET_HOST",
        help=(
            "SSH proxy routing (can be repeated). "
            "Format: proxy_host:target_host  "
            "Means: reach target_host by SSHing through proxy_host first."
        ),
    )
    parser.add_argument(
        "--netmask",
        type=int,
        default=30,
        metavar="BITS",
        help="CIDR prefix length for the netmask (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Parse proxy mappings: target_host -> proxy_host
    proxy_map: dict[str, str] = {}
    for spec in args.proxy:
        parts = spec.split(":")
        if len(parts) != 2:
            log.error(
                "Invalid --proxy format: %r  (expected proxy_host:target_host)",
                spec,
            )
            return 1
        proxy_host, target_host = parts
        proxy_map[target_host] = proxy_host

    # Parse node specs
    nodes: list[NodeConfig] = []
    for spec in args.node:
        try:
            node = NodeConfig.from_spec(spec)
        except ValueError as exc:
            log.error("%s", exc)
            return 1
        # Attach proxy if configured
        node.proxy = proxy_map.get(node.ssh_host)
        nodes.append(node)

    if args.netmask < 0 or args.netmask > 32:
        log.error("Netmask must be between 0 and 32, got %d", args.netmask)
        return 1

    configure_nodes(nodes, args.netmask, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
