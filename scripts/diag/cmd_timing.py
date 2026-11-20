"""[17] Command timing in the SimLingo training data (11/20). DIAGNOSTIC, offline, CPU only.

Question: near a junction, does the route-planner command of a frame (which sets both the training prompt's
navigation sentence and the world-model anchor) already say what the expert does in the next 2.5 s?
If the planner switches to "turn" only very close to the junction, frames 10-15 m before it carry
"follow lane" while the expert's future trajectory already turns -> the model learns to decide from the image.

Reads only measurements/*.json.gz (4 Hz): command, next_command, junction, speed, ego_matrix, target_point.
No map, no simulator.

Part A (frames): ego not in a junction, speed > 1 m/s, the expert's heading changes > 30 deg within 2.5 s
         (10 frames) -> what command did that frame carry? Binned by time until the ego enters the junction.
Part B (approaches): every junction entry; manoeuvre = heading change from 1 frame before entry to 2 frames
         after exit (L / S / R, 30 deg). Command at 1 / 2 / 3 / 4 s before entry, and how long before entry the
         command first matched the manoeuvre (L=1, R=2, S=3) without interruption.

    $PY scripts/diag/cmd_timing.py --root database/simlingo --routes 600 --workers 16 --out logs/cmd_timing.json
"""
import argparse, collections, glob, gzip, json, math, random, sys
from multiprocessing import Pool
from pathlib import Path

CMD = {1: "L", 2: "R", 3: "S", 4: "follow", 5: "lcL", 6: "lcR"}
MATCH = {"L": "L", "R": "R", "S": "S"}
HZ = 4
FUT = 10           # 2.5 s
TURN_DEG = 30.0
LOOK = 40          # look up to 10 s ahead / back


def wrap(a): return (a + 180.0) % 360.0 - 180.0


def yaw(m):
    em = m["ego_matrix"]; return math.degrees(math.atan2(em[1][0], em[0][0]))


def pos(m):
    em = m["ego_matrix"]; return em[0][3], em[1][3]


def man(dy): return "L" if dy < -TURN_DEG else "R" if dy > TURN_DEG else "S"   # CARLA: yaw grows clockwise


def load_route(r):
    out = []
    for f in sorted(glob.glob(f"{r}/measurements/*.json.gz")):
        try:
            with gzip.open(f, "rt") as fh: me = json.load(fh)
            out.append(dict(cmd=CMD.get(int(me.get("command", 0)), "?"), nxt=CMD.get(int(me.get("next_command", 0) or 0), "?"),
                            j=bool(me.get("junction", False)), v=float(me.get("speed", 0.0)), yaw=yaw(me), p=pos(me),
                            tp=float(math.hypot(*me["target_point"][:2])) if me.get("target_point") is not None else None))
        except Exception:
            out.append(None)
    return out


def analyse(r):
    ms = load_route(r)
    A, B = [], []
    n = len(ms)
    nxt_j = [None] * n; k = None    # frames until the ego is next inside a junction
    for i in range(n - 1, -1, -1):
        if ms[i] is not None and ms[i]["j"]: k = i
        nxt_j[i] = None if k is None else k - i
    for i in range(n - FUT):
        a, b = ms[i], ms[i + FUT]
        if a is None or b is None or a["j"] or a["v"] < 1.0 or a["cmd"] in ("lcL", "lcR"): continue
        fm = man(wrap(b["yaw"] - a["yaw"]))
        tj = nxt_j[i]
        A.append(dict(fut=fm, cmd=a["cmd"], nxt=a["nxt"], tj=None if tj is None or tj > LOOK else tj / HZ, tp=a["tp"]))
    for e in range(1, n):
        if ms[e] is None or ms[e - 1] is None or not ms[e]["j"] or ms[e - 1]["j"]: continue
        x = next((t for t in range(e, min(n, e + LOOK)) if ms[t] is not None and not ms[t]["j"]), None)
        if x is None or x + 2 >= n or ms[x + 2] is None: continue
        mv = man(wrap(ms[x + 2]["yaw"] - ms[e - 1]["yaw"]))
        before = {s: (ms[e - s * HZ]["cmd"] if e - s * HZ >= 0 and ms[e - s * HZ] is not None else None) for s in (1, 2, 3, 4)}
        lead, dist = 0, 0.0          # frames before entry the command already matched the manoeuvre, uninterrupted
        for t in range(e - 1, max(-1, e - 1 - LOOK), -1):
            if ms[t] is None or ms[t]["cmd"] != MATCH[mv]: break
            lead += 1
            if ms[t + 1] is not None: dist += math.dist(ms[t]["p"], ms[t + 1]["p"])
        B.append(dict(man=mv, before=before, lead_s=lead / HZ, lead_m=round(dist, 1), cmd_entry=ms[e - 1]["cmd"]))
    return A, B


def pct(c, k, tot): return f"{100.0 * c.get(k, 0) / tot:5.1f}%" if tot else "    -"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="database/simlingo"); ap.add_argument("--routes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    routes = glob.glob(f"{a.root}/data/simlingo/*/*/*/Town*")
    random.Random(a.seed).shuffle(routes); routes = routes[:a.routes]
    print(f"{len(routes)} routes", flush=True)
    A, B = [], []
    with Pool(a.workers) as p:
        for i, (ra, rb) in enumerate(p.imap_unordered(analyse, routes, chunksize=4)):
            A += ra; B += rb
            if (i + 1) % 100 == 0: print(f"  {i + 1} routes, {len(A)} frames, {len(B)} junction entries", flush=True)
    L = []
    cmds = ["follow", "S", "L", "R", "?"]
    L.append(f"\n== Part A: {len(A)} frames (not in junction, >1 m/s). Command carried vs. what the expert does in the next 2.5 s")
    for fm in ("L", "R", "S"):
        rows = [x for x in A if x["fut"] == fm]
        L.append(f"\n  expert future = {fm}  ({len(rows)} frames)   command: " + "  ".join(f"{c:>7s}" for c in cmds) + "   next_command=" + fm)
        bins = [("enters junction in 0-1 s", 0, 1), ("1-2 s", 1, 2), ("2-3 s", 2, 3), ("3-5 s", 3, 5), ("5-10 s", 5, 10.01), ("no junction within 10 s", None, None)]
        for name, lo, hi in bins:
            rr = [x for x in rows if (x["tj"] is None if lo is None else x["tj"] is not None and lo <= x["tj"] < hi)]
            c = collections.Counter(x["cmd"] for x in rr); cn = collections.Counter(x["nxt"] for x in rr)
            L.append(f"    {name:26s} n={len(rr):6d}   " + "  ".join(pct(c, k, len(rr)) for k in cmds) + f"   {pct(cn, fm, len(rr))}")
    L.append(f"\n== Part B: {len(B)} junction entries. Command 1/2/3/4 s before entering, by the manoeuvre the expert made")
    for mv in ("L", "R", "S"):
        rows = [x for x in B if x["man"] == mv]
        if not rows: continue
        L.append(f"\n  manoeuvre {mv}  ({len(rows)} entries)        " + "  ".join(f"{c:>7s}" for c in cmds))
        for s in (4, 3, 2, 1):
            c = collections.Counter(x["before"][s] for x in rows if x["before"][s] is not None); tot = sum(c.values())
            L.append(f"    {s} s before entry   n={tot:6d}   " + "  ".join(pct(c, k, tot) for k in cmds))
        ls = sorted(x["lead_s"] for x in rows); lm = sorted(x["lead_m"] for x in rows)
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]
        L.append(f"    command = {MATCH[mv]} uninterrupted before entry: median {q(ls, .5):.2f} s / {q(lm, .5):.1f} m;"
                 f" p10 {q(ls, .1):.2f} s / {q(lm, .1):.1f} m; p90 {q(ls, .9):.2f} s / {q(lm, .9):.1f} m;"
                 f" never matched {100.0 * sum(1 for x in rows if x['lead_s'] == 0) / len(rows):.1f}%")
    txt = "\n".join(L); print(txt)
    json.dump(dict(routes=len(routes), A=A, B=B), open(a.out, "w"))
    Path(a.out).with_suffix(".txt").write_text(txt)


if __name__ == "__main__":
    main()
