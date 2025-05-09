#!/usr/bin/env python3
"""sky_to_slurm.py – v2.4 (ripgrep‑powered, **gzip‑compressed**)

Updates
=======
* **Compresses the mounts archive** with gzip (`.tar.gz`) to shrink upload
  size.
* Still enumerates files via **ripgrep `rg`** and gathers per‑mount listings in
  parallel threads.
* SBATCH script now extracts with `tar -xzf`.

Requirements: ripgrep (`rg`) must be installed on the submission host. No
changes needed on the cluster (GNU tar handles gzip by default).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# .skyignore handling
# ----------------------------------------------------------------------------

def _load_skyignore(cwd: Path | None = None) -> List[str]:
    cwd = cwd or Path.cwd()
    ignore_file = cwd / ".skyignore"
    if not ignore_file.exists():
        return []
    return [ln.strip() for ln in ignore_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


# ----------------------------------------------------------------------------
# ripgrep‑based file discovery
# ----------------------------------------------------------------------------

def _rel_arcname(remote_path: str, user: str) -> str:
    if remote_path.startswith("~/"):
        return f"home/{user}/{remote_path[2:]}"
    return remote_path.lstrip("/")


def _should_ignore(path: Path, patterns: List[str]) -> bool:
    rel = os.path.relpath(path, Path.cwd())
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


def _rg_list_files(root: Path, excludes: List[str]) -> List[Path]:
    """Return all non‑ignored files under *root* using `rg --files`."""
    cmd = ["rg", "--files", "-0"]
    for pat in excludes:
        cmd += ["--glob", f"!{pat}"]
    try:
        res = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        print("[sky_to_slurm] ERROR: ripgrep ('rg') is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)
    paths = [root / Path(p) for p in res.stdout.decode().split("\0") if p]
    return paths


def _collect_files_parallel(mounts: Dict[str, str], excludes: List[str]) -> Dict[Tuple[str, str], List[Path]]:
    results: Dict[Tuple[str, str], List[Path]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(mounts) or 1)) as ex:
        fut_to_key = {}
        for remote, local in mounts.items():
            src_root = Path(local).expanduser()
            if not src_root.exists():
                continue
            if src_root.is_dir():
                fut = ex.submit(_rg_list_files, src_root, excludes)
            else:
                fut = ex.submit(lambda p: [p] if not _should_ignore(p, excludes) else [], src_root)
            fut_to_key[fut] = (remote, local)
        for fut in concurrent.futures.as_completed(fut_to_key):
            remote, local = fut_to_key[fut]
            try:
                files = fut.result()
            except Exception as e:
                print(f"[sky_to_slurm] WARN: error listing {local}: {e}", file=sys.stderr)
                files = []
            results[(remote, local)] = files
    return results


def _create_mounts_tar(mounts: Dict[str, str], user: str, excludes: List[str]) -> str:
    """Create **gzip‑compressed** tarball of mounts; return path."""
    if not mounts:
        return ""

    file_map = _collect_files_parallel(mounts, excludes)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz")
    tmp_path = Path(tmp.name)
    tmp.close()

    with tarfile.open(tmp_path, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        for (remote, local_root), files in file_map.items():
            base_arc = _rel_arcname(remote, user)
            local_root_p = Path(local_root).expanduser()
            for f in files:
                if local_root_p.is_dir():
                    rel_sub = os.path.relpath(f, local_root_p)
                    arcname = os.path.join(base_arc, rel_sub)
                else:
                    arcname = base_arc  # single file mount
                tar.add(f, arcname=arcname, recursive=False)
    return str(tmp_path)


# ----------------------------------------------------------------------------
# SBATCH script construction
# ----------------------------------------------------------------------------

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
        f"#SBATCH --chdir=/home/{args.user}/sky_workdir",
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

    if getattr(args, "tar_filename", None):
        sb += [
            "# ---- unpack file mounts ----",
            f"tarball=/workspace/jeffg/{args.tar_filename}",
            "if [ ! -f $tarball ]; then echo 'Mount tarball missing' >&2; exit 1; fi",
            "tar -xzf $tarball -C /",
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


# ----------------------------------------------------------------------------
# CLI + submission logic
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Submit a Sky YAML job to Slurm (locally or remotely).")
    p.add_argument("yaml", help="Sky‑style YAML file")
    p.add_argument("--partition", "-p", help="Slurm partition")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--mem", default="0", help="e.g. 64G (0 = cluster default)")
    p.add_argument("--time", help="wall‑clock limit HH:MM:SS")
    p.add_argument("--ssh-host", help="Remote login host (e.g. cluster1)")
    p.add_argument("--user", required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Remote submission helpers
# ----------------------------------------------------------------------------

def _remote_submit(host: str, script_path: str, tar_local: str | None, tar_filename: str | None):
    subprocess.check_call(["ssh", host, "mkdir", "-p", "/workspace/jeffg"])
    if tar_local and tar_filename:
        print(f"[sky_to_slurm] Uploading mounts tar to {host}:/workspace/jeffg/{tar_filename} …")
        subprocess.check_call(["rsync", tar_local, f"{host}:/workspace/jeffg/{tar_filename}"])
    remote_tmp = f"/tmp/{Path(script_path).name}"
    subprocess.check_call(["rsync", script_path, f"{host}:{remote_tmp}"])
    print("[sky_to_slurm] Submitting batch job …")
    ssh_res = subprocess.run(["ssh", host, f"sbatch {remote_tmp}"], capture_output=True, text=True)
    if ssh_res.returncode == 0:
        print(ssh_res.stdout.strip())
    else:
        print(ssh_res.stderr.strip(), file=sys.stderr)
        sys.exit(ssh_res.returncode)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    with open(args.yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mounts: Dict[str, str] = cfg.get("file_mounts", {})
    ignores = _load_skyignore()

    tar_local = tar_filename = None
    if args.ssh_host and mounts:
        tar_local = _create_mounts_tar(mounts, args.user, ignores)
        tar_filename = Path(tar_local).name
        setattr(args, "tar_filename", tar_filename)

    sbatch_script = build_sbatch(cfg, args)

    if args.dry_run:
        print("# ---- Generated SBATCH script ----\n")
        print(sbatch_script)
        if tar_local:
            print(f"# ---- Mounts tar located at {tar_local} ----")
        return

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as tmp:
        tmp.write(sbatch_script)
        script_path = tmp.name

    os.chmod(script_path, 0o777)

    if args.ssh_host:
        _remote_submit(args.ssh_host, script_path, tar_local, tar_filename)
    else:
        print("[sky_to_slurm] Submitting locally … (mount tar not used)")
        res = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if res.returncode == 0:
            print(res.stdout.strip())
        else:
            print(res.stderr.strip(), file=sys.stderr)
            sys.exit(res.returncode)


if __name__ == "__main__":
    main()
