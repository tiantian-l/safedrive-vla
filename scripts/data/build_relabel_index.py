"""Build the junction relabelling index for fine-tuning ([17], 11/20). Offline, CPU only, training data only.

For every junction entry of every SimLingo route (measurements/*.json.gz, 4 Hz):
  manoeuvre = the planner's own command 1 frame before entry (L / R / S), kept only if the expert's heading
              change from 1 frame before entry to 2 frames after exit agrees (|d| <= 30 deg = S); a
              "follow the road" command there counts as S when the expert goes straight; other cases skipped.
  skipped:    heading change > 135 deg, > 15 s inside the junction (roundabouts, long waits), disagreements.
  window:     frames from `--max-m` metres (driven distance) before entry, not crossing the previous
              junction, through the last frame inside the junction (distance 0 there).
Outputs (in --out-dir, normally the SimLingo root so the dataset finds the buckets):
  relabel_index.json   {"meta": ..., "routes": {route dir relative to --root: {frame: [manoeuvre, metres to entry]}}}
  buckets_relabel.pkl  {"relabel_straight": [...], "relabel_turn": [...]} measurement paths of window frames
                       within --bucket-m and moving > 1 m/s (same format as buckets_paths.pkl)
  relabel_stats.txt    counts and the check below
Check: at 10-15 m before entry, does the command (original vs relabelled) agree with the expert's next 2.5 s?

    $PY scripts/data/build_relabel_index.py --root database/simlingo --out-dir database/simlingo --workers 32
    $PY scripts/data/build_relabel_index.py --check-sentences      # library vs evaluation sentences only
"""
import argparse, collections, glob, gzip, json, math, os, pickle, random, re, sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CMD = {1: "L", 2: "R", 3: "S", 4: "F", 5: "lcL", 6: "lcR"}
HZ, FUT, TURN, LOOK = 4, 10, 30.0, 60


def wrap(a): return (a + 180.0) % 360.0 - 180.0
def man(d): return "L" if d < -TURN else "R" if d > TURN else "S"     # CARLA yaw grows clockwise


def load(route):
    out = {}
    for f in glob.glob(f"{route}/measurements/*.json.gz"):
        try:
            fr = int(os.path.basename(f).split(".")[0])
            with gzip.open(f, "rt") as fh: me = json.load(fh)
            em = me["ego_matrix"]
            out[fr] = dict(c=CMD.get(int(me.get("command", 0)), "?"), j=bool(me.get("junction", False)),
                           v=float(me.get("speed", 0.0)), yaw=math.degrees(math.atan2(em[1][0], em[0][0])),
                           p=(em[0][3], em[1][3]))
        except Exception:
            pass
    if not out: return []
    n = max(out) + 1
    return [out.get(i) for i in range(n)]


def analyse(args):
    route, max_m = args
    ms = load(route)
    n = len(ms)
    lab, skip = {}, collections.Counter()
    last_exit = -1
    for e in range(1, n):
        if ms[e] is None or ms[e - 1] is None or not ms[e]["j"] or ms[e - 1]["j"]: continue
        x = next((t for t in range(e, min(n, e + LOOK + 1)) if ms[t] is not None and not ms[t]["j"]), None)
        if x is None or x + 2 >= n or ms[x + 2] is None:
            skip["no exit / too long in junction"] += 1; continue
        d = wrap(ms[x + 2]["yaw"] - ms[e - 1]["yaw"]); mv = man(d); c0 = ms[e - 1]["c"]
        prev_exit, last_exit = last_exit, x
        if abs(d) > 135: skip["heading change > 135"] += 1; continue
        if c0 in ("L", "R", "S"):
            if c0 != mv: skip[f"planner {c0} vs expert {mv}"] += 1; continue
        elif c0 == "F":
            if mv != "S": skip[f"planner follow vs expert {mv}"] += 1; continue
        else:
            skip[f"planner {c0} at entry"] += 1; continue
        for t in range(e, x):
            if ms[t] is not None: lab[t] = [mv, 0.0]
        dist = 0.0
        for t in range(e - 1, max(prev_exit, -1), -1):
            if ms[t] is None or ms[t + 1] is None or ms[t]["j"]: break
            dist += math.dist(ms[t]["p"], ms[t + 1]["p"])
            if dist > max_m: break
            lab[t] = [mv, round(dist, 1)]
    # check: frames 10-15 m before entry, moving, expert future known
    chk = collections.Counter()
    for t, (mv, dist) in lab.items():
        if not 10.0 <= dist <= 15.0 or t + FUT >= n or ms[t] is None or ms[t + FUT] is None or ms[t]["v"] < 1.0: continue
        fut = man(wrap(ms[t + FUT]["yaw"] - ms[t]["yaw"]))
        orig = {"L": "L", "R": "R", "S": "S", "F": "S"}.get(ms[t]["c"], "?")
        chk[(fut, "orig_ok" if orig == fut else "orig_bad")] += 1
        chk[(fut, "new_ok" if mv == fut else "new_bad")] += 1
    moving = {t for t in lab if ms[t] is not None and ms[t]["v"] > 1.0}
    return route, {str(t): v for t, v in sorted(lab.items())}, moving, skip, chk


def check_sentences():
    lib = json.load(open(REPO / "safedrive_vla" / "assets" / "relabel_sentences.json"))
    test = json.load(open(REPO / "team_code" / "instruction_meta_commands.json"))
    norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).strip()
    t = {norm(k) for k in test}
    bad = [s for k in ("left", "right", "straight", "lane") for s in lib[k] if norm(s) in t]
    dup = [s for k in ("left", "right", "straight", "lane") for s in lib[k] if lib[k].count(s) > 1]
    print(f"sentence library: {sum(len(lib[k]) for k in ('left', 'right', 'straight', 'lane'))} sentences, "
          f"{len(t)} evaluation sentences; identical to evaluation: {bad or 'none'}; duplicates: {dup or 'none'}")
    if bad: sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="database/simlingo"); ap.add_argument("--out-dir")
    ap.add_argument("--routes", type=int, default=0, help="0 = all routes")
    ap.add_argument("--max-m", type=float, default=40.0); ap.add_argument("--bucket-m", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--check-sentences", action="store_true")
    a = ap.parse_args()
    check_sentences()
    if a.check_sentences: return
    root = os.path.abspath(a.root)
    routes = sorted(glob.glob(f"{root}/data/simlingo/*/*/*/Town*"))
    if a.routes: random.Random(0).shuffle(routes); routes = routes[:a.routes]
    print(f"{len(routes)} routes", flush=True)
    index, buckets = {}, {"relabel_straight": [], "relabel_turn": []}
    skip, chk, mvs = collections.Counter(), collections.Counter(), collections.Counter()
    with Pool(a.workers) as p:
        for i, (r, lab, moving, sk, ck) in enumerate(p.imap_unordered(analyse, [(r, a.max_m) for r in routes], chunksize=4)):
            if lab: index[os.path.relpath(r, root)] = lab
            skip.update(sk); chk.update(ck)
            for t, (mv, dist) in lab.items():
                if dist == 0.0 and int(t) - 1 >= 0 and str(int(t) - 1) in lab and lab[str(int(t) - 1)][1] > 0: mvs[mv] += 1
                if int(t) in moving and dist <= a.bucket_m:
                    buckets["relabel_straight" if mv == "S" else "relabel_turn"].append(f"{r}/measurements/{int(t):04d}.json.gz")
            if (i + 1) % 500 == 0: print(f"  {i + 1} routes", flush=True)
    out = Path(a.out_dir or root); out.mkdir(parents=True, exist_ok=True)
    nf = sum(len(v) for v in index.values())
    json.dump(dict(meta=dict(routes=len(routes), max_m=a.max_m, turn_deg=TURN), routes=index), open(out / "relabel_index.json", "w"))
    pickle.dump(buckets, open(out / "buckets_relabel.pkl", "wb"))
    L = [f"{len(routes)} routes, {len(index)} with relabelled frames, {nf} frames in the index (<= {a.max_m} m)",
         f"junction entries kept: {dict(mvs)}; skipped: {dict(skip.most_common())}",
         f"buckets (moving, <= {a.bucket_m} m): relabel_straight {len(buckets['relabel_straight'])}, relabel_turn {len(buckets['relabel_turn'])}",
         "check, frames 10-15 m before entry: command agrees with the expert's next 2.5 s (original -> relabelled)"]
    for fut in ("L", "R", "S"):
        o, ob = chk[(fut, "orig_ok")], chk[(fut, "orig_bad")]; nw, nb = chk[(fut, "new_ok")], chk[(fut, "new_bad")]
        if o + ob: L.append(f"  expert future {fut}: n={o + ob:6d}   original {100 * o / (o + ob):5.1f}%  ->  relabelled {100 * nw / (nw + nb):5.1f}%")
    txt = "\n".join(L); print(txt); (out / "relabel_stats.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
