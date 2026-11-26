"""E19b (11/25): training-window index and held-out split of the E17 paired junction data ([26] / [27]). CPU only.

E17 routes (scripts/data/collect_pairs.py, database/e17_pairs/data/simlingo/e17/routes_training/{paired,unpaired}/
<Town>_Rep0_<Town>_j<junction>_r<road>_l<lane>_<L|S|R>_route0_<date>/) start ~60 m before one junction; the
routes of a group share the start lane and drive identically until the junction. The expert's left / straight /
right trajectories only separate 0-10 m before the entry (E17 probe 11/25: 10-15 m differ by ~1 deg), so the
anchor dropout of E19b is applied in this window only:
  before the entry: frames whose driven distance to the first junction frame is <= --win-m (10 m)
  inside the junction: every frame of a straight route; turn routes while the heading changed < --turn-deg (30)
                       since the frame before the entry (the expert's anchor switch happens at 27-65 deg)
Frames outside [SKIP_FIRST_N_FRAMES, n - PRED_LEN - 2) are dropped (no 2.5 s future label).

Split (by junction, never by route, so the routes of a group stay together):
  holdout_town  every old-town route (Town01-05, Town10HD): cross-town test, never trained on
  holdout       Town12 junctions with sha1("e17_split:<Town>_<junction>") % 100 < --holdout-pct (10)
  train         the rest
Output (--out, default <root>/e17_index.json), relabel-index compatible (safedrive_vla/data/relabel.py reads
entry[0], entry[1]):
  {"meta": {...}, "routes": {route dir relative to root: {frame: [manoeuvre, metres to entry, heading change deg]}},
   "split": {route dir relative to root: train | holdout | holdout_town}}
and e17_index.txt with the counts.

    $PY scripts/data/build_e17_index.py --root database/e17_pairs
"""
import argparse, collections, glob, hashlib, json, math, os, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "data"))
from build_relabel_index import load, wrap   # noqa: E402  (measurements -> per-frame c / j / v / yaw / p)

SKIP_FIRST_N_FRAMES, PRED_LEN = 10, 11        # safedrive_vla/data/simlingo_base.py
ROUTE_RE = re.compile(r"_Rep\d+_(?P<route>(?P<town>Town\w+?)_j(?P<j>\d+)_r-?\d+_l-?\d+_(?P<man>[LSR]))_route\d+_[\d_]+$")
TRAIN_TOWNS_HELD_IN = ("Town12",)


def split_of(town, junction, pct):
    if town not in TRAIN_TOWNS_HELD_IN:
        return "holdout_town"
    h = int(hashlib.sha1(f"e17_split:{town}_j{junction}".encode()).hexdigest(), 16) % 100
    return "holdout" if h < pct else "train"


def window(route_dir, man, win_m, turn_deg):
    """{frame: [man, metres to entry, heading change since the frame before the entry]} and a skip reason."""
    ms = load(route_dir)
    n = len(ms)
    n_rgb = len(glob.glob(f"{route_dir}/rgb/*.jpg"))
    e = next((t for t in range(1, n) if ms[t] is not None and ms[t - 1] is not None and ms[t]["j"] and not ms[t - 1]["j"]), None)
    if e is None:
        return {}, "no junction entry"
    yaw0 = ms[e - 1]["yaw"]
    lab = {}
    dist = 0.0
    for t in range(e - 1, -1, -1):
        if ms[t] is None or ms[t + 1] is None or ms[t]["j"]:
            break
        dist += math.dist(ms[t]["p"], ms[t + 1]["p"])
        if dist > win_m:
            break
        lab[t] = [man, round(dist, 2), round(wrap(ms[t]["yaw"] - yaw0), 1)]
    for t in range(e, n):
        if ms[t] is None or not ms[t]["j"]:
            break
        d = wrap(ms[t]["yaw"] - yaw0)
        if man != "S" and abs(d) >= turn_deg:
            break
        lab[t] = [man, 0.0, round(d, 1)]
    lab = {t: v for t, v in lab.items() if SKIP_FIRST_N_FRAMES <= t < n_rgb - PRED_LEN - 2}
    return lab, ("" if lab else "window outside the valid frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="database/e17_pairs"); ap.add_argument("--out")
    ap.add_argument("--win-m", type=float, default=10.0); ap.add_argument("--turn-deg", type=float, default=30.0)
    ap.add_argument("--holdout-pct", type=int, default=10)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    route_dirs = sorted(glob.glob(f"{root}/data/simlingo/*/*/*/Town*"))
    routes, split, skip = {}, {}, collections.Counter()
    frames = collections.Counter(); nroutes = collections.Counter(); junctions = collections.defaultdict(set)
    for rd in route_dirs:
        m = ROUTE_RE.search(os.path.basename(rd))
        if m is None:
            skip["name not parsed"] += 1; continue
        rel = os.path.relpath(rd, root)
        sp = split_of(m["town"], m["j"], a.holdout_pct)
        lab, why = window(rd, m["man"], a.win_m, a.turn_deg)
        if not lab:
            skip[why] += 1; continue
        routes[rel] = {str(t): v for t, v in sorted(lab.items())}
        split[rel] = sp
        kind = Path(rd).parent.name
        nroutes[(sp, kind)] += 1; junctions[sp].add(f"{m['town']}_j{m['j']}")
        for v in lab.values():
            frames[(sp, v[0], "junction" if v[1] == 0.0 else "approach")] += 1
    out = Path(a.out or Path(root) / "e17_index.json")
    meta = dict(win_m=a.win_m, turn_deg=a.turn_deg, holdout_pct=a.holdout_pct, routes=len(route_dirs))
    json.dump(dict(meta=meta, routes=routes, split=split), open(out, "w"))
    L = [f"{len(route_dirs)} route dirs, {len(routes)} with window frames; skipped {dict(skip)}",
         f"window: <= {a.win_m} m before the entry + inside the junction (turns while |heading change| < {a.turn_deg} deg)"]
    for sp in ("train", "holdout", "holdout_town"):
        L.append(f"{sp:13s} junctions {len(junctions[sp]):4d}  routes paired {nroutes[(sp, 'paired')]:5d} unpaired {nroutes[(sp, 'unpaired')]:3d}  "
                 + "  ".join(f"{mv}: {frames[(sp, mv, 'approach')]}+{frames[(sp, mv, 'junction')]}" for mv in "LSR")
                 + "  (frames approach+junction)")
    txt = "\n".join(L); print(txt); out.with_suffix(".txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
