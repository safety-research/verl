#!/usr/bin/env python3
"""sky_to_slurm.py

Convert a Sky‑style YAML job into a Slurm job – locally **or on a remote
login node** – with

* **Parallel rsync** of `file_mounts` (cluster‑path → local‑path) to a remote
  login host (when `--ssh-host` is given).
* **.skyignore** support: any glob patterns listed in a `.skyignore` file in the
  current directory are passed to `rsync --exclude`, so temporary files or large
  caches aren’t uploaded.

Typical workflow
----------------
```bash
python sky_to_slurm.py job.yaml --ssh-host cluster1 -p gpu
```

Steps when `--ssh-host` is provided:
1. Load ignore globs from `.skyignore` (if present).
2. **Parallel rsync** every file_mount (respecting ignores) from *local* →
   `cluster1:`.
3. Copy the generated SBATCH script to `cluster1:/tmp/…` and `sbatch` it.
4. Without `--ssh-host`, submission is local and no rsync occurs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import shlex
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import List

import yaml

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _extract_num_gpus(acc: str | None) -> int:
    if not acc:
        return 1
    m = re.search(r":(\d+)$", acc)
    if m:
        return int(m.group(1))
    try:
        return int(acc)
    except ValueError:
        return 1


def _append_block(lines: List[str], label: str, block: str | None) -> None:
    if block:
        lines.append(f"# ---- {label} ----")
        lines.extend(block.strip().splitlines())

# -----------------------------------------------------------------------------
# .skyignore handling
# -----------------------------------------------------------------------------

def _load_skyignore(cwd: Path | None = None) -> List[str]:
    """Read .skyignore patterns from *cwd* (default: current working dir)."""
    cwd = cwd or Path.cwd()
    ignore_file = cwd / ".skyignore"
    if not ignore_file.exists():
        return []
    patterns: List[str] = []
    for line in ignore_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns

# -----------------------------------------------------------------------------
# Rsync mounts (parallel, with excludes)
# -----------------------------------------------------------------------------

def _rsync_single(host: str, remote_path: str, local_path: str, excludes: List[str]):
    """Sync one mount; raise CalledProcessError on failure."""
    local = Path(local_path).expanduser()
    remote_path = os.path.expandvars(remote_path)
    remote_dir = os.path.dirname(remote_path.rstrip("/")) or "/"

    # Ensure destination hierarchy exists on host
    subprocess.check_call(["ssh", host, "mkdir", "-p", remote_dir])

    base_cmd = [
        "rsync", "-a", "--compress", "--progress", "-e", "ssh",
    ]
    for pat in excludes:
        base_cmd += ["--exclude", pat]

    if local.is_dir():
        cmd = base_cmd + [
            f"{local}/", f"{host}:{remote_path.rstrip('/')}/",
        ]
    else:
        cmd = base_cmd + [
            f"{local}", f"{host}:{remote_path}",
        ]
    subprocess.check_call(cmd)


def _rsync_mounts_parallel(host: str, mounts: dict[str, str], excludes: List[str], max_workers: int = 8):
    if not mounts:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(_rsync_single, host, remote, local, excludes)
            for remote, local in mounts.items()
        ]
        for fut in concurrent.futures.as_completed(futs):
            fut.result()

# -----------------------------------------------------------------------------
# SBATCH script construction
# -----------------------------------------------------------------------------

def build_sbatch(cfg: dict, args) -> str:
    resources = cfg.get("resources", {})
    gpus_per_node = _extract_num_gpus(str(resources.get("accelerators", "gpu:1")))

    envs = cfg.get("envs", {})
    setup_block = cfg.get("setup", "")
    run_block = cfg.get("run", "")

    job_name = envs.get("EXPERIMENT_NAME", Path(args.yaml).stem)

    sb: List[str] = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --gres=gpu:{gpus_per_node}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --chdir=/home/{args.user}/sky_workdir"
    ]
    if args.partition:
        sb.append(f"#SBATCH --partition={args.partition}")
    if args.mem != "0":
        sb.append(f"#SBATCH --mem={args.mem}")
    if args.time:
        sb.append(f"#SBATCH --time={args.time}")

    sb += [
        "set -euo pipefail",
        "echo \"[slurm] Node $(hostname) – $(date)\"",
    ]

    _append_block(sb, "setup", setup_block)

    if envs:
        sb.append("# ---- export envs ----")
        for k, v in envs.items():
            sb.append(f"export {k}={shlex.quote(str(v))}")

    sb += [
        "# ---- SkyPilot cluster-level env vars ----",
        "node_ips=$(srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES hostname -I | awk '{print $1}' | paste -sd, -)",
        "export SKYPILOT_NODE_IPS=$node_ips",
        "export SKYPILOT_NUM_NODES=$SLURM_JOB_NUM_NODES",
        "export SKYPILOT_NODE_RANK=${SLURM_NODEID:-0}",
        f"export SKYPILOT_NUM_GPUS_PER_NODE={gpus_per_node}",
        "export MASTER_ADDR=${node_ips%%,*}",
    ]

    _append_block(sb, "run", run_block)
    return "\n".join(sb) + "\n"

# -----------------------------------------------------------------------------
# CLI + submission logic
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Submit a Sky YAML job to Slurm (locally or remotely).")
    p.add_argument("yaml", help="Sky-style YAML file")
    p.add_argument("--partition", "-p", help="Slurm partition")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--mem", default="0", help="e.g. 64G (0 = cluster default)")
    p.add_argument("--time", help="wall‑clock limit HH:MM:SS")
    p.add_argument("--ssh-host", help="Remote login host (e.g. cluster1)")
    p.add_argument("--user", help="User whose home directory will be used as wd", required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _remote_submit(host: str, script_path: str, mounts: dict[str, str], excludes: List[str]):
    print(f"[sky_to_slurm] Parallel rsync to {host} (excludes: {', '.join(excludes) or 'none'}) …")
    _rsync_mounts_parallel(host, mounts, excludes)

    remote_tmp = f"/tmp/{Path(script_path).name}"
    subprocess.check_call(["scp", script_path, f"{host}:{remote_tmp}"])

    print("[sky_to_slurm] Submitting batch job …")
    ssh_res = subprocess.run(["ssh", host, f"sbatch {remote_tmp}"], capture_output=True, text=True)
    if ssh_res.returncode == 0:
        print(ssh_res.stdout.strip())
    else:
        print(ssh_res.stderr.strip(), file=sys.stderr)
        sys.exit(ssh_res.returncode)


def main() -> None:
    args = parse_args()

    ignores = _load_skyignore()

    with open(args.yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sbatch_script = build_sbatch(cfg, args)

    if args.dry_run:
        print("# ---- Generated SBATCH script ----\n")
        print(sbatch_script)
        return

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as tmp:
        tmp.write(sbatch_script)
        script_path = tmp.name

    os.chmod(script_path, 0o777)

    if args.ssh_host:
        _remote_submit(args.ssh_host, script_path, cfg.get("file_mounts", {}), ignores)
    else:
        print("[sky_to_slurm] Submitting locally … (ignores are not used)")
        res = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if res.returncode == 0:
            print(res.stdout.strip())
        else:
            print(res.stderr.strip(), file=sys.stderr)
            sys.exit(res.returncode)


if __name__ == "__main__":
    main()
