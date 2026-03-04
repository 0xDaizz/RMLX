#!/usr/bin/env python3
"""Launch distributed RMLX commands across hosts.

Modeled after mlx.launch with a smaller feature set:
- launch one or more processes per host via SSH
- set rank/world env vars
- prefix process output with rank + host
- fail fast if any rank exits non-zero
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple


@dataclass
class Slot:
    rank: int
    world_size: int
    host: str
    local_slot: int


def _parse_env_pairs(values: List[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid --env entry: {item!r} (expected KEY=VALUE)")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"invalid --env key in: {item!r}")
        env[k] = v
    return env


def _load_hosts(hosts_csv: Optional[str], hostfile: Optional[str]) -> List[str]:
    if bool(hosts_csv) == bool(hostfile):
        raise ValueError("provide exactly one of --hosts or --hostfile")
    if hosts_csv:
        hosts = [h.strip() for h in hosts_csv.split(",") if h.strip()]
        if not hosts:
            raise ValueError("no hosts parsed from --hosts")
        return hosts

    path = Path(hostfile or "")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"hostfile must be a JSON list: {path}")
    hosts = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "ssh" not in entry:
            raise ValueError(f"hostfile entry {i} missing 'ssh'")
        hosts.append(str(entry["ssh"]))
    if not hosts:
        raise ValueError("hostfile has no hosts")
    return hosts


def _build_slots(hosts: List[str], repeat: int) -> List[Slot]:
    slots: List[Slot] = []
    world_size = len(hosts) * repeat
    rank = 0
    for host in hosts:
        for local_slot in range(repeat):
            slots.append(
                Slot(rank=rank, world_size=world_size, host=host, local_slot=local_slot)
            )
            rank += 1
    return slots


def _remote_command(
    base_cmd: str, slot: Slot, backend: str, coordinator: str, extra_env: Dict[str, str]
) -> str:
    env = {
        "RMLX_RANK": str(slot.rank),
        "RMLX_WORLD_SIZE": str(slot.world_size),
        "RMLX_LOCAL_SLOT": str(slot.local_slot),
        "RMLX_BACKEND": backend,
        "RMLX_COORDINATOR": coordinator,
    }
    env.update(extra_env)
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return f"{exports} {base_cmd}"


def _spawn_for_slot(slot: Slot, cmd: str, ssh_user: Optional[str]) -> subprocess.Popen[str]:
    if slot.host in {"localhost", "127.0.0.1"}:
        return subprocess.Popen(
            ["bash", "-lc", cmd],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    target = f"{ssh_user}@{slot.host}" if ssh_user else slot.host
    return subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", target, f"bash -lc {shlex.quote(cmd)}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _reader_thread(
    slot: Slot,
    proc: subprocess.Popen[str],
    outq: "Queue[Tuple[Slot, str]]",
) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        outq.put((slot, line.rstrip("\n")))


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch distributed RMLX jobs via SSH.")
    parser.add_argument("--hosts", help="Comma-separated hostnames/IPs.")
    parser.add_argument("--hostfile", help="JSON hostfile (entries need 'ssh').")
    parser.add_argument(
        "-n",
        "--repeat-hosts",
        type=int,
        default=1,
        help="Processes per host (default: 1).",
    )
    parser.add_argument("--backend", default="rdma", help="Backend hint exported as RMLX_BACKEND.")
    parser.add_argument("--ssh-user", help="SSH user override.")
    parser.add_argument("--env", action="append", default=[], help="Extra env KEY=VALUE (repeatable).")
    parser.add_argument(
        "--print-python",
        action="store_true",
        help="Print resolved python executable and exit.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run (prefix with --).")
    args = parser.parse_args()

    if args.print_python:
        print(sys.executable)
        return 0

    if not args.command:
        print("error: missing command. Use: -- <command>", file=sys.stderr)
        return 2

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("error: empty command after --", file=sys.stderr)
        return 2
    base_cmd = " ".join(shlex.quote(c) for c in command)

    try:
        hosts = _load_hosts(args.hosts, args.hostfile)
        if args.repeat_hosts < 1:
            raise ValueError("--repeat-hosts must be >= 1")
        extra_env = _parse_env_pairs(args.env)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    slots = _build_slots(hosts, args.repeat_hosts)
    coordinator = hosts[0]
    outq: Queue[Tuple[Slot, str]] = Queue()
    procs: List[Tuple[Slot, subprocess.Popen[str]]] = []
    threads: List[threading.Thread] = []

    try:
        for slot in slots:
            slot_cmd = _remote_command(base_cmd, slot, args.backend, coordinator, extra_env)
            proc = _spawn_for_slot(slot, slot_cmd, args.ssh_user)
            procs.append((slot, proc))
            t = threading.Thread(target=_reader_thread, args=(slot, proc, outq), daemon=True)
            t.start()
            threads.append(t)

        failures: Dict[int, int] = {}
        alive = len(procs)
        while alive > 0:
            try:
                slot, line = outq.get(timeout=0.1)
                print(f"[rank={slot.rank} host={slot.host}] {line}")
            except Empty:
                pass

            alive = 0
            for slot, proc in procs:
                rc = proc.poll()
                if rc is None:
                    alive += 1
                    continue
                if slot.rank not in failures and rc != 0:
                    failures[slot.rank] = rc

            if failures:
                # Fail fast: terminate remaining processes once first failure appears.
                for slot, proc in procs:
                    if proc.poll() is None:
                        proc.terminate()
                break

        # Drain remaining output.
        while True:
            try:
                slot, line = outq.get_nowait()
                print(f"[rank={slot.rank} host={slot.host}] {line}")
            except Empty:
                break

        codes: Dict[int, int] = {}
        for slot, proc in procs:
            rc = proc.wait()
            codes[slot.rank] = rc
            if rc != 0:
                failures.setdefault(slot.rank, rc)

        if failures:
            ordered = ", ".join(f"rank {r}: {c}" for r, c in sorted(failures.items()))
            print(f"launch failed ({ordered})", file=sys.stderr)
            return 1
        return 0
    finally:
        for _slot, proc in procs:
            if proc.poll() is None:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
