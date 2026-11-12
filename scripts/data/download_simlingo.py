#!/usr/bin/env python
"""Download the SimLingo training split (RenzKa/simlingo) one archive at a time.

For every ``data_*`` / ``dreamer_*`` archive whose name contains the split
(default ``routes_training``) the script downloads it, extracts it into
``--root`` (skipping ``lidar/``: the challenge is camera-only and the data
loader never reads it), records it in ``<root>/.done`` and deletes the
archive. Re-running resumes: finished archives are skipped and an interrupted
download is resumed by huggingface_hub. ``buckets_paths.pkl`` goes to
``<root>/buckets_paths.pkl``. The run stops before an archive would push the
project's Lustre usage past ``--max-used-gb``.

Run on the login node (compute nodes are offline):
    $PY scripts/data/download_simlingo.py --root database/simlingo --dry-run
    nohup $PY scripts/data/download_simlingo.py --root database/simlingo \
        > logs/simlingo_download.log 2>&1 &
"""
import argparse
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

from huggingface_hub import HfApi, hf_hub_download

REPO = "RenzKa/simlingo"


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def project_used_gb(project_id, mount):
    """Used space of the Lustre project in GB, from ``lfs quota``."""
    out = subprocess.run(["lfs", "quota", "-q", "-p", str(project_id), mount],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == mount:
            return int(re.sub(r"\D", "", fields[1])) / 1024 ** 2  # kbytes -> GB
        if len(fields) >= 1 and re.match(r"^\d+\*?$", fields[0]):  # wrapped line
            return int(re.sub(r"\D", "", fields[0])) / 1024 ** 2
    raise RuntimeError(f"cannot parse lfs quota output:\n{out}")


def list_files(api):
    files = {}
    for entry in api.list_repo_tree(REPO, repo_type="dataset", recursive=True):
        size = getattr(entry, "size", None)
        if size is not None:
            files[entry.path] = size
    return files


def group_name(path):
    return re.sub(r"_chunk_\d+\.tar\.gz$", "", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="data_root, e.g. database/simlingo")
    p.add_argument("--split", default="routes_training",
                   help="substring an archive name must contain")
    p.add_argument("--max-used-gb", type=float, default=900)
    p.add_argument("--project-id", default="60762")
    p.add_argument("--mount", default="/projects")
    p.add_argument("--keep-lidar", action="store_true")
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be downloaded and the total size, then exit")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    staging = os.path.join(root, ".staging")
    done_file = os.path.join(root, ".done")

    api = HfApi()
    files = list_files(api)
    archives = sorted(f for f in files
                      if f.endswith(".tar.gz") and args.split in f
                      and (f.startswith("data_") or f.startswith("dreamer_")))
    want = archives + (["buckets_paths.pkl"] if "buckets_paths.pkl" in files else [])
    total = sum(files[f] for f in want) / 1e9

    if args.dry_run:
        groups = defaultdict(lambda: [0, 0])
        for f in archives:
            g = groups[group_name(f)]
            g[0] += 1
            g[1] += files[f]
        print(f"{'archives':>8} {'GB':>8}  group")
        for name in sorted(groups):
            n, size = groups[name]
            print(f"{n:8d} {size / 1e9:8.1f}  {name}")
        print(f"\nselected: {len(want)} files, {total:.1f} GB (split '{args.split}')")
        others = defaultdict(float)
        for f, s in files.items():
            if f.endswith(".tar.gz") and f not in archives:
                others[group_name(f).split("_simlingo")[0] + " / " +
                       ("routes_validation" if "routes_validation" in f else "other")] += s
        print("not selected:")
        for k in sorted(others):
            print(f"  {others[k] / 1e9:8.1f} GB  {k}")
        other_names = sorted({group_name(f) for f in files
                              if f.endswith(".tar.gz") and f not in archives
                              and "routes_validation" not in f})
        if other_names:
            print("not selected, neither training nor validation (check these):")
            for n in other_names:
                print("  ", n)
        return

    os.makedirs(staging, exist_ok=True)
    done = set()
    if os.path.exists(done_file):
        with open(done_file) as fh:
            done = {line.strip() for line in fh if line.strip()}
    todo = [f for f in want if f not in done]
    log(f"{len(want)} files ({total:.1f} GB), {len(done)} done, {len(todo)} to go -> {root}")

    for i, name in enumerate(todo, 1):
        size_gb = files[name] / 1e9
        used = project_used_gb(args.project_id, args.mount)
        # archive + its extracted copy (~1.05x) must fit under the cap
        if used + 2.1 * size_gb > args.max_used_gb:
            log(f"STOP: project uses {used:.0f} GB; {name} ({size_gb:.1f} GB) "
                f"would exceed --max-used-gb {args.max_used_gb:.0f}")
            sys.exit(2)
        log(f"[{i}/{len(todo)}] {name} ({size_gb:.1f} GB), project at {used:.0f} GB")

        for attempt in range(1, args.retries + 1):
            try:
                path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=staging)
                break
            except Exception as exc:  # network hiccups: back off and retry
                log(f"  download failed ({attempt}/{args.retries}): {exc}")
                if attempt == args.retries:
                    sys.exit(1)
                time.sleep(60 * attempt)

        if name.endswith(".tar.gz"):
            cmd = ["tar", "-xzf", path, "-C", root]
            if not args.keep_lidar:
                cmd += ["--exclude=*/lidar", "--exclude=*/lidar/*"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                log(f"  tar failed, archive kept for inspection: {res.stderr.strip()[:500]}")
                sys.exit(1)
            os.remove(path)
        else:
            os.replace(path, os.path.join(root, os.path.basename(name)))

        with open(done_file, "a") as fh:
            fh.write(name + "\n")
        log("  ok")

    log("all done")


if __name__ == "__main__":
    main()
