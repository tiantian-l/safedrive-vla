"""Training-data volume and driving-mode distribution of SafeDriveVLA (read-only).

    $PY scripts/data/mode_stats.py --root database/simlingo \
        [--config safedrive_vla/configs/experiment/safedrive_vla.yaml] \
        [--draws 6000] [--workers 8] [--seed 0]

Part 1 (volume): indexes the training frames exactly like SimLingoBase
(train split = first 99 % of the hashed route list for driving frames,
routes_training for action-dreaming frames; clean routes only; frames
10 .. n-14; bucket membership from buckets_paths.pkl; data_scale = 1),
then reports per bucket the frame count, the sampling weight, the number
of samples per epoch that SafeDriveDataModule draws (min over buckets of
len / weight) and how often each bucket's frames are repeated per epoch.

Part 2 (modes): draws samples with the same weights (bucket by weight,
frame uniformly inside the bucket) and labels them like DrivingDataset /
ActionDreamingDataset (incl. the 50 % shifted-camera augmentation of
driving frames and the random alternative instruction of dreaming
frames). Only measurements and dreamer json files are read, no images.

The label code below copies dataset.py (_dream_label) and simlingo_base.py
(get_waypoints, _route_is_clean, bucket loading) so that the script does not
need torch / imgaug; driving_mode.py and constants.py are imported as is.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from safedrive_vla.constants import COMMAND_TO_META_COMMAND  # noqa: E402
from safedrive_vla.data.driving_mode import (  # noqa: E402
    ACCEL_STRICT, CRUISE_SPEED, HORIZON_S, LATERAL_DEPARTURE_M, WAYPOINT_DT,
    classify_driving_mode, classify_speed_instruction)

# --- copied from safedrive_vla/data/simlingo_base.py -------------------------
PRED_LEN = 11
SKIP_FIRST_N_FRAMES = 10
TRAIN_SPLIT_FRACTION = 0.99
IMG_SHIFT_AUGMENTATION_PROB = 0.5
BUCKET_ALIASES = {
    "acceleration_negative_5": ["acceleration_-5"],
    "acceleration_negative_1": ["acceleration_-1"],
    "acceleration_positive_1": ["acceleration_5"],
    "acceleration_positive_5": ["acceleration_20"],
    "lateral_control_1_2": ["lateral_control_1", "lateral_control_2"],
    "lateral_control_higher_5": ["lateral_control_5", "lateral_control_1000000"],
}
BUCKET_PATH_PREFIX = "database/simlingo_v2_2025_01_10"
# --- copied from safedrive_vla/data/dataset.py --------------------------------
SPEED_DREAM_CATEGORIES = frozenset({"target_speed", "faster", "faster_factor", "slower", "slower_factor", "stop"})

META_SHORT = {"follow_road": "follow_road", "go_straight_intersection": "straight", "turn_left": "turn_L",
              "turn_right": "turn_R", "lane_change_left": "lc_L", "lane_change_right": "lc_R", None: "none"}
MODES = ("strict", "cautious", "fallback", None)


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def route_is_clean(route_dir: str) -> bool:
    f = f"{route_dir}/results.json.gz"
    if not os.path.isfile(f):
        return False
    try:
        with gzip.open(f, "rt") as fh:
            r = json.load(fh)
    except Exception:
        return False
    if r["scores"]["score_composed"] < 100.0:
        inf = r["infractions"]
        only_min = r["num_infractions"] == len(inf["min_speed_infractions"]) + len(inf["outside_route_lanes"])
        if not (r["scores"]["score_route"] > 94.0 and only_min):
            return False
    return True


def to_dreamer(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    return os.path.join(root, "dreamer", os.path.relpath(rel, "data"))


def get_waypoints(measurements, y_aug=0.0, yaw_aug=0.0):
    origin = np.array(measurements[0]["ego_matrix"])[:3]
    t0, r0 = origin[:, 3:4], origin[:, :3]
    a = np.deg2rad(yaw_aug)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    tr = np.array([[0.0], [y_aug]])
    out = []
    for m in measurements:
        wp = np.array(m["ego_matrix"])[:3, 3:4]
        ego = r0.T @ (wp - t0)
        out.append(np.squeeze(rot.T @ (np.expand_dims(ego[:2, 0], axis=1) - tr)))
    return out


def dream_label(option, current, waypoints, safe):
    base_meta = COMMAND_TO_META_COMMAND.get(int(current.get("command", 0)))
    if base_meta is None:
        return None, None
    category = str(option.get("mode", ""))
    meta_name = base_meta
    if category == "lane_change":
        wp = np.asarray(waypoints, dtype=np.float32).reshape(-1, 2)
        end_y = float(wp[-1, 1]) if wp.size else 0.0
        meta_name = "lane_change_right" if end_y > 0 else "lane_change_left"
    if not safe:
        return meta_name, "fallback"
    speed = float(current["speed"])
    wp = np.asarray(waypoints, dtype=np.float32).reshape(-1, 2)
    if category == "lane_change":
        mode = classify_driving_mode(wp, meta_name, speed, float(current.get("speed_limit", 0.0)))
    elif category in SPEED_DREAM_CATEGORIES:
        info = option.get("info", {}) or {}
        target = float(info.get("target_speed", info.get("final_speed", speed)))
        mode = classify_speed_instruction(wp, speed, target)
    else:
        mode = "cautious"
    return meta_name, mode


def cruise_reasons(wp, speed):
    """Why a follow_road / go-straight frame is not STRICT (same tests as classify_cruise)."""
    wp = np.asarray(wp, dtype=np.float32).reshape(-1, 2)
    pts = np.vstack([np.zeros((1, 2), dtype=np.float32), wp])
    sp = np.linalg.norm(np.diff(pts, axis=0), axis=1) / WAYPOINT_DT
    accel = (float(sp[-2:].mean()) - speed) / HORIZON_S
    return dict(slow=speed < CRUISE_SPEED, braking=accel < ACCEL_STRICT,
                lateral=float(np.abs(wp[:, 1]).max()) > LATERAL_DEPARTURE_M)


def route_geometry(route, wp):
    """Split a lateral departure into 'the road itself bends' vs 'the expert left the route line'.
    route: current['route'] (ego frame, the planned lane centre ahead); wp: un-augmented expert waypoints.
    Returns (road_lat, off_route): road_lat = largest |y| of the route within the distance the expert drove
    (> 1.5 m = a curve the expert simply followed); off_route = largest distance of a waypoint to the route
    polyline (> 1.0 m = swerve / lane change / leaving the route)."""
    r = np.asarray(route, dtype=np.float32).reshape(-1, 2)
    wp = np.asarray(wp, dtype=np.float32).reshape(-1, 2)
    if r.shape[0] < 2:
        return float("nan"), float("nan")
    seg = np.linalg.norm(np.diff(r, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    driven = float(np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), wp]), axis=0), axis=1).sum())
    road_lat = float(np.abs(r[cum <= max(driven, 1.0), 1]).max())
    a, b = r[:-1], r[1:]
    d = b - a
    l2 = np.maximum((d ** 2).sum(1), 1e-9)
    dist = []
    for q in wp:
        if q[0] > r[-1, 0] + 1.0:  # beyond the stored route: cannot judge
            continue
        t = np.clip(((q - a) * d).sum(1) / l2, 0, 1)
        dist.append(float(np.min(np.linalg.norm(a + t[:, None] * d - q, axis=1))))
    return road_lat, (max(dist) if dist else float("nan"))


# --- indexing -------------------------------------------------------------------
def index_routes(root: str):
    """(driving_routes, dreaming_routes) with frame lists, like SimLingoBase.__init__."""
    routes = glob.glob(f"{root}/data/simlingo/*/*/*/Town*")
    routes = sorted(routes, key=lambda d: stable_hash(f"route_split:{os.path.relpath(d, root)}"))
    n_train = int(TRAIN_SPLIT_FRACTION * len(routes))
    drive_routes = routes[:n_train]
    dream_routes = [d for d in routes if "routes_training" in d]
    clean_cache, frames_cache = {}, {}

    def frames(route):
        if route not in frames_cache:
            num_seq = len(os.listdir(f"{route}/rgb"))
            frames_cache[route] = list(range(SKIP_FIRST_N_FRAMES, num_seq - PRED_LEN - 2))
        return frames_cache[route]

    def clean(route):
        if route not in clean_cache:
            clean_cache[route] = route_is_clean(route)
        return clean_cache[route]

    t0 = time.time()
    drive = {}
    for i, r in enumerate(drive_routes):
        if clean(r):
            drive[r] = frames(r)
        if i % 2000 == 0:
            print(f"  [index driving] {i}/{len(drive_routes)} routes, {time.time() - t0:.0f}s", flush=True)
    dream = {}
    for i, r in enumerate(dream_routes):
        ddir = to_dreamer(root, f"{r}/dreamer")
        if not os.path.isdir(os.path.dirname(ddir)) or not clean(r):
            continue
        have = set(os.listdir(ddir)) if os.path.isdir(ddir) else set()
        fr = [s for s in frames(r) if f"{s:04}.json.gz" in have]
        if fr:
            dream[r] = fr
        if i % 2000 == 0:
            print(f"  [index dreaming] {i}/{len(dream_routes)} routes, {time.time() - t0:.0f}s", flush=True)
    stats = dict(total_route_folders=len(routes), driving_train_routes=len(drive_routes),
                 driving_clean=len(drive), dreaming_candidate_routes=len(dream_routes), dreaming_used=len(dream))
    return drive, dream, stats


def load_bucket(root: str, bucket_dict, name: str):
    files = collections.defaultdict(set)
    for key in BUCKET_ALIASES.get(name, [name]):
        if key not in bucket_dict:
            raise KeyError(f"bucket {name} ({key}) not in buckets_paths.pkl")
        for p in bucket_dict[key]:
            p = p.replace(BUCKET_PATH_PREFIX, root)
            files[os.path.dirname(p)].add(os.path.basename(p))
    return files


# --- labelling (worker) --------------------------------------------------------------
def load_meas(route, start):
    out = []
    for i in range(PRED_LEN + 1):
        try:
            with gzip.open(f"{route}/measurements/{start + i:04}.json.gz", "rt") as f:
                out.append(json.load(f))
        except FileNotFoundError:
            out.append(out[-1])
    return out


def label(job):
    kind, bucket, route, start, root, seed = job
    rng = random.Random(seed)
    try:
        meas = load_meas(route, start)
        cur = meas[0]
        speed = float(cur["speed"])
        rec = dict(kind=kind, bucket=bucket, speed=speed, stopped=speed < 1.0)
        if kind == "driving":
            shifted = rng.random() <= IMG_SHIFT_AUGMENTATION_PROB
            y_aug = cur.get("augmentation_translation", 0.0) if shifted else 0.0
            yaw_aug = cur.get("augmentation_rotation", 0.0) if shifted else 0.0
            wp = np.array(get_waypoints(meas, y_aug, yaw_aug)[1:-1])
            meta = COMMAND_TO_META_COMMAND.get(int(cur.get("command", 0)))
            mode = classify_driving_mode(wp, meta, speed, float(cur.get("speed_limit", 0.0))) if meta else None
            rec.update(meta=meta, mode=mode, cat="expert", safe=True)
            # Part 3: the same frame labelled from the un-augmented expert trajectory
            wp_org = np.array(get_waypoints(meas)[1:-1])
            rec.update(shifted=shifted, y_aug=float(y_aug), yaw_aug=float(yaw_aug),
                       mode_org=classify_driving_mode(wp_org, meta, speed, float(cur.get("speed_limit", 0.0))) if meta else None)
            if meta in ("follow_road", "go_straight_intersection"):
                rec["why"] = cruise_reasons(wp, speed)
                rec["why_org"] = cruise_reasons(wp_org, speed)
                if "route" in cur:
                    rec["road_lat"], rec["off_route"] = route_geometry(cur["route"], wp_org)
            if seed % 997 == 0:
                rec["mkeys"] = sorted(cur.keys())
        else:
            wp_org = np.array(get_waypoints(meas)[1:-1])
            with gzip.open(to_dreamer(root, f"{route}/dreamer/{start:04}.json.gz"), "rt") as f:
                alts = json.load(f)
            options = [o for k, os_ in alts.items() if "factor" not in k for o in os_]
            opt = rng.choice(options)
            safe = opt.get("safe_to_execute", True) is not False
            wp = wp_org if (not safe or opt["waypoints"] == "org") else np.array(opt["waypoints"])
            meta, mode = dream_label(opt, cur, wp, safe)
            rec.update(meta=meta, mode=mode, cat=str(opt.get("mode", "")), safe=safe)
        # speeds 2.5 s later: does a stopped car start moving?
        end_speed = float(np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), wp]), axis=0)[-2:], axis=1).mean() / 0.25)
        rec["starts_moving"] = rec["stopped"] and end_speed > 1.0
        return rec
    except Exception as e:  # noqa: BLE001
        return dict(kind=kind, bucket=bucket, error=f"{type(e).__name__}: {e}")


# --- reporting ------------------------------------------------------------------------
def pct_table(recs, key, title):
    groups = collections.defaultdict(collections.Counter)
    for r in recs:
        groups[r[key]][r["mode"]] += 1
    print(f"\n  {title}")
    print(f"  {'':38s} {'n':>6s} {'share':>6s} | {'strict':>7s} {'cautious':>8s} {'fallback':>8s} {'none':>5s}")
    total = len(recs)
    for g, c in sorted(groups.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        cells = " ".join(f"{100 * c[m] / n:{w}.1f}" for m, w in zip(MODES, (7, 8, 8, 5)))
        print(f"  {str(g):38s} {n:6d} {100 * n / total:5.1f}% | {cells}")


def part3(rk):
    """Is CAUTIOUS on straight / follow-road frames caused by the shifted-camera augmentation?"""
    print("\n== Part 3: CAUTIOUS on follow_road / go_straight driving frames ==")
    cr = [r for r in rk if r.get("meta") in ("follow_road", "go_straight_intersection")]
    if not cr:
        print("  no samples"); return
    for name, grp in (("shifted camera", [r for r in cr if r["shifted"]]), ("original camera", [r for r in cr if not r["shifted"]])):
        n = len(grp) or 1
        c = collections.Counter(r["mode"] for r in grp)
        print(f"  {name:16s} n={len(grp):5d}: " + ", ".join(f"{m} {100 * c[m] / n:.1f}%" for m in MODES[:3]))
    sh = [r for r in cr if r["shifted"]]
    sc = [r for r in sh if r["mode"] == "cautious"]
    if sc:
        t = collections.Counter(r["mode_org"] for r in sc)
        print(f"  shifted & CAUTIOUS: {len(sc)}; labelled from the un-augmented trajectory they would be: "
              + ", ".join(f"{m} {100 * t[m] / len(sc):.1f}%" for m in MODES[:3]))
    for name, key, grp in (("CAUTIOUS (as trained)", "why", [r for r in cr if r["mode"] == "cautious"]),
                           ("CAUTIOUS, original camera only", "why", [r for r in cr if r["mode"] == "cautious" and not r["shifted"]]),
                           ("CAUTIOUS, shifted camera only", "why", sc)):
        n = len(grp) or 1
        cnt = collections.Counter()
        for r in grp:
            w = r[key]
            cnt["slow (<6 m/s)"] += w["slow"]; cnt["braking (<-0.5 m/s2)"] += w["braking"]; cnt["lateral (>1.5 m)"] += w["lateral"]
            cnt["lateral ONLY (fast, not braking)"] += w["lateral"] and not w["slow"] and not w["braking"]
        print(f"  reasons, {name} (n={len(grp)}; a frame can have several): "
              + ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in cnt.items()))
    lat = [r for r in cr if r["mode"] == "cautious" and r["why_org"]["lateral"] and not r["why_org"]["slow"]
           and not r["why_org"]["braking"] and "road_lat" in r]
    if lat:
        curve = sum(1 for r in lat if r["road_lat"] > 1.5 and not r["off_route"] > 1.0)
        off = sum(1 for r in lat if r["off_route"] > 1.0)
        print(f"  CAUTIOUS only because of lateral departure (un-augmented trajectory, fast, not braking): n={len(lat)}; "
              f"road itself curves and expert stays on route {100 * curve / len(lat):.1f}%, "
              f"expert leaves the route line (>1 m) {100 * off / len(lat):.1f}%, other {100 * (len(lat) - curve - off) / len(lat):.1f}%")
    keys = sorted({k for r in rk for k in r.get("mkeys", [])})
    if keys:
        print("  measurement keys (for smarter labels, e.g. hazard flags):", ", ".join(keys))
    for key, lab in (("y_aug", "|translation| m"), ("yaw_aug", "|rotation| deg")):
        v = np.abs(np.array([r[key] for r in sh])) if sh else np.zeros(1)
        print(f"  shifted frames {lab}: median {np.median(v):.2f}, p90 {np.percentile(v, 90):.2f}, max {v.max():.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="database/simlingo")
    ap.add_argument("--config", default="safedrive_vla/configs/experiment/safedrive_vla.yaml")
    ap.add_argument("--draws", type=int, default=6000, help="samples per kind (driving / dreaming)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    parts = [dict(cfg["data"]["train_partitions"]), dict(cfg["data"]["train_partitions_action_dreaming"])]
    gpus, bs, epochs = int(cfg.get("gpus", 8)), int(cfg["data"]["batch_size"]), int(cfg.get("max_epochs", 14))
    if float(cfg["data"].get("data_scale", 1.0)) < 1.0:
        print("WARNING: config data_scale < 1; this script counts data_scale = 1")
    t0 = time.time()

    print("== Part 1: volume ==")
    drive, dream, st = index_routes(root)
    for k, v in st.items():
        print(f"  {k}: {v}")
    with open(f"{root}/buckets_paths.pkl", "rb") as f:
        bucket_dict = pickle.load(f)

    buckets = []  # (kind, name, weight, [(route, frame)])
    for kind, routes, partition in (("driving", drive, parts[0]), ("dreaming", dream, parts[1])):
        total_w = sum(partition.values())
        for name, w in partition.items():
            if name == "all":
                frames = [(r, s) for r, fr in routes.items() for s in fr]
            else:
                files = load_bucket(root, bucket_dict, name)
                frames = [(r, s) for r, fr in routes.items()
                          for s in fr if f"{s:04}.json.gz" in files.get(f"{r}/measurements", ())]
            buckets.append((kind, name, float(w) / total_w * 0.5, frames))
    del bucket_dict

    nonempty = [b for b in buckets if b[3]]
    per_epoch = int(min(len(b[3]) / max(b[2], 1e-9) for b in nonempty))
    tot_w = sum(b[2] for b in nonempty)  # WeightedRandomSampler normalises the weights
    uniq_drive = sum(len(fr) for fr in drive.values())
    uniq_dream = sum(len(fr) for fr in dream.values())
    print(f"\n  unique driving frames {uniq_drive:,}   unique dreaming frames {uniq_dream:,}")
    print(f"\n  {'kind':9s} {'bucket':34s} {'frames':>10s} {'weight':>7s} {'draws/epoch':>12s} {'repeats':>8s}")
    for kind, name, w, frames in buckets:
        draws = per_epoch * w / tot_w if frames else 0
        rep = draws / len(frames) if frames else float("nan")
        print(f"  {kind:9s} {name:34s} {len(frames):10,d} {w:7.3f} {draws:12,.0f} {rep:8.2f}")
    limiting = min(nonempty, key=lambda b: len(b[3]) / b[2])
    steps = per_epoch / (gpus * bs)
    print(f"\n  samples per epoch: {per_epoch:,} (limited by bucket {limiting[0]}/{limiting[1]})")
    print(f"  = {steps:,.0f} optimizer steps/epoch at {gpus} GPUs x batch {bs} (no grad accumulation assumed);"
          f" {epochs} epochs = {per_epoch * epochs:,} samples")
    print(f"  paper cost ~190 GPU h for {epochs} epochs -> ~{per_epoch * epochs / 190:,.0f} samples per GPU hour")
    print(f"  ({time.time() - t0:.0f}s)")

    print("\n== Part 2: driving modes (sampled with the training weights) ==")
    rng = random.Random(a.seed)
    jobs = []
    for kind in ("driving", "dreaming"):
        bk = [b for b in nonempty if b[0] == kind]
        weights = [b[2] for b in bk]
        for i in range(a.draws):
            b = rng.choices(bk, weights=weights)[0]
            r, s = rng.choice(b[3])
            jobs.append((kind, b[1], r, s, root, rng.randrange(1 << 30)))
    with Pool(a.workers) as pool:
        recs = pool.map(label, jobs, chunksize=32)
    errs = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    if errs:
        print(f"  {len(errs)} samples failed, e.g. {collections.Counter(e['error'] for e in errs).most_common(3)}")

    allc = collections.Counter(r["mode"] for r in recs)
    print(f"\n  ALL ({len(recs)} samples): " + ", ".join(f"{m} {100 * allc[m] / len(recs):.1f}%" for m in MODES if allc[m]))
    for kind in ("driving", "dreaming"):
        rk = [r for r in recs if r["kind"] == kind]
        c = collections.Counter(r["mode"] for r in rk)
        print(f"\n -- {kind} ({len(rk)} samples): " + ", ".join(f"{m} {100 * c[m] / len(rk):.1f}%" for m in MODES if c[m]))
        for r in rk:
            r["meta_s"] = META_SHORT.get(r["meta"], str(r["meta"]))
        pct_table(rk, "meta_s", "by meta-command")
        if kind == "driving":
            pct_table(rk, "bucket", "by bucket")
        else:
            for r in rk:
                r["cat_safe"] = f"{r['cat']}{'' if r['safe'] else ' (UNSAFE)'}"
            pct_table(rk, "cat_safe", "by instruction category (UNSAFE = refuse, expert trajectory kept)")
            u = sum(1 for r in rk if not r["safe"])
            print(f"  unsafe instructions: {u} / {len(rk)} = {100 * u / len(rk):.1f}%")
        stopped = [r for r in rk if r["stopped"]]
        if stopped:
            sm = sum(1 for r in stopped if r["starts_moving"])
            cs = collections.Counter(r["mode"] for r in stopped)
            print(f"  stopped now (<1 m/s): {len(stopped)} ({100 * len(stopped) / len(rk):.1f}%), "
                  f"of which start moving within 2.5 s: {sm} ({100 * sm / len(stopped):.1f}%); modes "
                  + ", ".join(f"{m} {cs[m]}" for m in MODES if cs[m]))
    part3([r for r in recs if r["kind"] == "driving"])
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
