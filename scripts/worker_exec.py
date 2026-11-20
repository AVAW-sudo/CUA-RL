#!/usr/bin/env python3
"""Execute argv, without shell interpolation, on one of the two authorized workers."""
import argparse
import os
import subprocess

WORKERS = {"worker-0": "research-a-worker-0", "worker-1": "research-a-worker-1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker", choices=sorted(set(WORKERS) | set(WORKERS.values())))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after the worker name")
    node = WORKERS.get(args.worker, args.worker)
    if os.uname().nodename == node:
        os.execvp(command[0], command)
    result = subprocess.run(["/opt/kube/kubectl", "exec", "-i", node, "-c", "main", "--", *command])
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
