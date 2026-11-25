"""E19a [28]: red-light action-dreaming samples from the E17-red routes (database/e17_red, training towns,
normal traffic-light cycle; collected with collect_pairs.py --no-all-green).

Frames where the PDM-Lite expert is held by a red light (measurements light_hazard):
  approach  speed > 0.5 m/s  (the expert is braking towards the stop line)   every frame
  wait      speed <= 0.5 m/s (standing at the red light)                     every --wait-every-th frame
get a dream file in SimLingo's layout (<root>/dreamer/simlingo/.../<route>/dreamer/NNNN.json.gz) with one
unsafe option: a "do not stop at the light" instruction (safedrive_vla/assets/red_light_sentences.json, generic +
the route's manoeuvre, {v} = 6/8/10/12 m/s), safe_to_execute = false -> ActionDreamingDataset supervises the
expert's trajectory (slowing down / waiting) with <FALLBACK>; prompt_style = command -> the evaluation prompt
format "Command: <instruction>. Predict the actions.". The frame range matches BaseDataset's indexing.
The dreams form their own training bucket (configs: data.extra_dreaming); the SimLingo buckets, including
leading_object_traffic.traffic_light, are unchanged.

    python3 scripts/data/build_red_dreams.py --root database/e17_red [--dry]
"""
import argparse, collections, glob, gzip, json, os, random, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKIP_FIRST_N_FRAMES = 10   # safedrive_vla/data/simlingo_base.py
PRED_LEN = 11              # safedrive_vla/data/simlingo_base.py
SPEEDS = [6, 8, 10, 12]
ROUTE_RE = re.compile(r"_Rep\d+_(?P<route>.+)_route\d+_[\d_]+$")


def route_is_clean(route_dir):
    """As BaseDataset._route_is_clean."""
    f = f"{route_dir}/results.json.gz"
    if not os.path.isfile(f):
        return False
    try:
        with gzip.open(f, "rt") as fh:
            r = json.load(fh)
    except Exception:  # noqa: BLE001
        return False
    if r["scores"]["score_composed"] < 100.0:
        inf = r["infractions"]
        only_min_speed = r["num_infractions"] == len(inf["min_speed_infractions"]) + len(inf["outside_route_lanes"])
        if not (r["scores"]["score_route"] > 94.0 and only_min_speed):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--wait-every", type=int, default=4, help="keep every n-th waiting frame (4 = 1 s)")
    ap.add_argument("--n-instr", type=int, default=6, help="instructions per dream file (one is drawn per sample)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    root = Path(os.path.abspath(a.root))
    lib = json.load(open(REPO / "safedrive_vla" / "assets" / "red_light_sentences.json"))
    routes = sorted(glob.glob(f"{root}/data/simlingo/*/*/*/Town*"))
    stats = collections.Counter()
    per_man = collections.Counter()
    for rd in routes:
        if not route_is_clean(rd):
            stats["routes_unclean"] += 1
            continue
        mfiles = sorted(glob.glob(f"{rd}/measurements/*.json.gz"))
        rgb = os.listdir(f"{rd}/rgb") if os.path.isdir(f"{rd}/rgb") else []
        n_seq = len(rgb) or len(mfiles)   # as BaseDataset (rgb); measurements only if no images synced
        m = ROUTE_RE.search(os.path.basename(rd))
        man = m.group("route")[-1] if m and m.group("route")[-2] == "_" else "?"
        rng = random.Random(os.path.basename(rd))
        n_wait = 0
        kept = 0
        for seq in range(SKIP_FIRST_N_FRAMES, n_seq - PRED_LEN - 2):
            with gzip.open(f"{rd}/measurements/{seq:04}.json.gz", "rt") as fh:
                cur = json.load(fh)
            if not cur.get("light_hazard"):
                continue
            if cur["speed"] > 0.5:
                kind = "approach"
            else:
                n_wait += 1
                if (n_wait - 1) % a.wait_every:
                    continue
                kind = "wait"
            pool = list(lib["generic"]) + list(lib.get(man, []))
            sents = [rng.choice(pool).format(v=rng.choice(SPEEDS)) for _ in range(a.n_instr)]
            dream = {"red_light": [{"dreamer_instruction": sents, "safe_to_execute": False, "mode": "red_light",
                                    "prompt_style": "command", "waypoints": "org", "route": "org",
                                    "info": {"kind": kind, "speed": round(cur["speed"], 2)}}]}
            stats[kind] += 1
            kept += 1
            if not a.dry:
                rel = os.path.relpath(f"{rd}/dreamer/{seq:04}.json.gz", root)            # data/simlingo/...
                out = root / "dreamer" / os.path.relpath(rel, "data")                     # dreamer/simlingo/...
                out.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(out, "wt") as fh:
                    json.dump(dream, fh)
        stats["routes_used" if kept else "routes_without_red"] += 1
        per_man[man] += kept
    lines = [f"root {root}: {len(routes)} routes", f"  {dict(stats)}", f"  frames per route manoeuvre {dict(per_man)}",
             f"  {'(dry run, nothing written)' if a.dry else 'written under ' + str(root / 'dreamer')}"]
    print("\n".join(lines))
    if not a.dry:
        (root / "red_dreams_report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
