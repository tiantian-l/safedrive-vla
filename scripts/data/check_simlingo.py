"""Check a SimLingo download against what the training code expects.

    $PY scripts/data/check_simlingo.py --root database/simlingo [--bucket-sample 20000]

Reports, per route group (data/simlingo/<split>/<group>): route folders, routes
the training code keeps (results.json.gz clean, as SimLingoBase._route_is_clean),
frames, routes with rgb_augmented / measurements / boxes, routes with a dreamer
folder and dreamer frames, and leftover lidar folders. Then loads
buckets_paths.pkl, rewrites the released prefix as the training code does and
checks that a random sample of its paths exists, split by training / validation
(validation was not downloaded, so it should be missing). Read-only.
"""
import argparse, collections, glob, gzip, json, os, pickle, random, sys, time

PREFIX = "database/simlingo_v2_2025_01_10"   # as in safedrive_vla/data/simlingo_base.py

def clean(route):
    f = f"{route}/results.json.gz"
    if not os.path.isfile(f): return False
    try:
        with gzip.open(f, "rt") as fh: r = json.load(fh)
    except Exception: return False
    if r["scores"]["score_composed"] < 100.0:
        inf = r["infractions"]
        only_min = r["num_infractions"] == len(inf["min_speed_infractions"]) + len(inf["outside_route_lanes"])
        if not (r["scores"]["score_route"] > 94.0 and only_min): return False
    return True

def n(path):
    try: return len(os.listdir(path))
    except FileNotFoundError: return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="database/simlingo")
    ap.add_argument("--bucket-sample", type=int, default=20000)
    a = ap.parse_args()
    root = os.path.abspath(a.root); t0 = time.time()
    routes = sorted(glob.glob(f"{root}/data/simlingo/*/*/*/Town*"))
    print(f"{len(routes)} route folders under {root}/data/simlingo/*/*/*/Town*")
    S = collections.defaultdict(collections.Counter)
    for i, r in enumerate(routes):
        g = "/".join(os.path.relpath(r, f"{root}/data/simlingo").split("/")[:3])
        s = S[g]; s["routes"] += 1
        frames = n(f"{r}/rgb"); s["frames"] += frames
        ok = clean(r); s["clean"] += ok
        if ok: s["clean_frames"] += frames
        for sub in ("rgb_augmented", "measurements", "boxes"):
            s[f"has_{sub}"] += os.path.isdir(f"{r}/{sub}")
        if n(f"{r}/measurements") != frames: s["meas!=rgb"] += 1
        s["lidar_left"] += os.path.isdir(f"{r}/lidar")
        d = os.path.join(root, "dreamer", os.path.relpath(r, f"{root}/data"))
        if os.path.isdir(d):
            s["dreamer_routes"] += 1; s["dreamer_frames"] += n(f"{d}/dreamer")
            if ok: s["clean_dreamer_routes"] += 1
        if (i + 1) % 500 == 0: print(f"  ... {i + 1}/{len(routes)} ({time.time() - t0:.0f}s)", file=sys.stderr)
    keys = ["routes", "clean", "frames", "clean_frames", "has_rgb_augmented", "has_measurements", "has_boxes",
            "meas!=rgb", "lidar_left", "dreamer_routes", "clean_dreamer_routes", "dreamer_frames"]
    print("\n" + "group".ljust(80) + " ".join(k[:14].rjust(14) for k in keys))
    tot = collections.Counter()
    for g in sorted(S):
        tot.update(S[g]); print(g.ljust(80) + " ".join(str(S[g][k]).rjust(14) for k in keys))
    print("TOTAL".ljust(80) + " ".join(str(tot[k]).rjust(14) for k in keys))
    dd = sorted(glob.glob(f"{root}/dreamer/simlingo/*/*/*/Town*"))
    orphan = [x for x in dd if not os.path.isdir(os.path.join(root, "data", os.path.relpath(x, f"{root}/dreamer")))]
    print(f"\ndreamer route folders: {len(dd)}, without a matching data folder: {len(orphan)}")
    for x in orphan[:5]: print("  ", os.path.relpath(x, root))

    pk = f"{root}/buckets_paths.pkl"
    if not os.path.isfile(pk): print("\nbuckets_paths.pkl missing"); return
    with open(pk, "rb") as f: b = pickle.load(f)
    print(f"\nbuckets_paths.pkl: {len(b)} buckets: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(b.items())))
    allp = [p for v in b.values() for p in v]
    print(f"  {len(allp)} entries; prefix ok: {sum(p.startswith(PREFIX) for p in allp[:100000])}/{min(len(allp), 100000)} of the first 100k")
    random.seed(0); sample = random.sample(allp, min(a.bucket_sample, len(allp)))
    res = collections.Counter()
    miss_ex = []
    for p in sample:
        split = "validation" if "validation" in p else "training"
        ex = os.path.exists(p.replace(PREFIX, root)); res[(split, ex)] += 1
        if split == "training" and not ex and len(miss_ex) < 5: miss_ex.append(p)
    for sp in ("training", "validation"):
        y, m = res[(sp, True)], res[(sp, False)]
        if y + m: print(f"  sample {sp:10s}: {y}/{y + m} exist ({100 * y / (y + m):.1f}%)")
    for p in miss_ex: print("   missing (training):", p)
    print(f"\ndone in {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
