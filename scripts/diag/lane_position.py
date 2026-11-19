"""Lane position at junctions: is the "leftmost lane -> turn left" shortcut in the data? (11/19)

Offline: needs only the ``carla`` package and the OpenDRIVE files (CARLA_ROOT); no simulator.

Part "test":  every turn / go-straight / lane-follow instruction of the CARLA-F routes. For the GT, the
              lane 5 m before it enters the first junction of the instruction window: index from the left
              among same-direction driving lanes, number of lanes, manoeuvres that lane can reach (L/S/R).
              Writes JSON (one record per instruction) to join with run outcomes.
Part "train": SimLingo frames approaching a junction (planner command left / right / straight, ego not yet
              in the junction). Same lane features from the ego position; tabulates the expert's command
              by lane position, per approach (consecutive frames of one approach count once).

    CARLA_ROOT=$P/carla0915 $PY scripts/diag/lane_position.py test  --routes-dir ../track3/routes/carla_f --out logs/lane_test.json
    CARLA_ROOT=$P/carla0915 $PY scripts/diag/lane_position.py train --root database/simlingo --routes 3000 --out logs/lane_train.json
"""
import argparse, collections, glob, gzip, json, math, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark" / "metrics"))
import carla  # noqa: E402
import carla_f_metrics as M  # noqa: E402

CMD = {1: "L", 2: "R", 3: "S", 4: "follow", 5: "lcL", 6: "lcR"}


def wrap(a): return (a + 180) % 360 - 180


def same_dir_lanes(wp):
    """(index from left, number of same-direction driving lanes)."""
    def walk(w, left):
        n, seen = 0, set()
        while True:
            w = w.get_left_lane() if left else w.get_right_lane()
            if w is None or (w.road_id, w.lane_id) in seen: return n
            seen.add((w.road_id, w.lane_id))
            if w.lane_id * wp.lane_id <= 0 or w.lane_type != carla.LaneType.Driving: return n
            n += 1
    l, r = walk(wp, True), walk(wp, False)
    return l, l + r + 1


def manoeuvres(wp):
    """Turn directions reachable from this lane through the next junction: subset of L/S/R.
    Walk along the lane to the last waypoint before the junction; its next() returns one waypoint per
    connecting lane into the junction; follow each to the junction exit and compare headings."""
    w = wp
    for _ in range(40):
        nx = w.next(2.0)
        if not nx: return ""
        if any(n.is_junction for n in nx): break
        w = nx[0]
    else:
        return ""
    y0, out = w.transform.rotation.yaw, set()
    for b in nx:
        e = b
        for _ in range(60):  # through the junction
            if not e.is_junction: break
            n2 = e.next(2.0)
            if not n2: break
            e = n2[0]
        d = wrap(e.transform.rotation.yaw - y0)
        out.add("L" if d < -30 else "R" if d > 30 else "S")
    return "".join(sorted(out))


def position(idx, n):
    return "single" if n == 1 else "leftmost" if idx == 0 else "rightmost" if idx == n - 1 else "middle"


def features(cmap, x, y, z):
    wp = cmap.get_waypoint(carla.Location(x=x, y=y, z=z), project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None: return None
    idx, n = same_dir_lanes(wp)
    return dict(road=wp.road_id, lane=wp.lane_id, lane_idx=idx, n=n, pos=position(idx, n), man=manoeuvres(wp), in_junction=wp.is_junction)


def densify(pts, step=1.0):
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        d = math.dist(a[:2], b[:2]); k = max(1, int(d / step))
        out += [tuple(a[i] + (b[i] - a[i]) * j / k for i in range(3)) for j in range(1, k + 1)]
    return out


def part_test(a, maps):
    recs = []
    for xml in sorted(Path(a.routes_dir).glob("*.xml")):
        route, gt, ins = M.parse_route(xml)
        cmap = maps.get(route.attrib["town"]); g = densify(gt); gc = M.cumulative_distances(g)
        texts = [e.findtext("text") for e in route.find("instructions")]
        for i, x in enumerate(ins):
            b = x["expected_behavior"]
            if b.get("type") not in ("turn", "lane_follow"): continue
            s0 = M.trigger_distance(x) or 0.0
            s1 = M.trigger_distance(ins[i + 1]) if i + 1 < len(ins) else gc[-1]
            idx = [j for j, s in enumerate(gc) if s0 <= s <= s1]
            jin = next((j for j in idx if (w := cmap.get_waypoint(carla.Location(*g[j]), project_to_road=True, lane_type=carla.LaneType.Driving)) and w.is_junction), None)
            rec = dict(route=xml.stem, town=route.attrib["town"], idx=i, type=b["type"], dir=b.get("direction", ""), text=texts[i], window=[round(s0, 1), round(s1, 1)])
            if jin is None:
                rec["junction"] = False; recs.append(rec); continue
            jb = max(idx[0], jin - 5)
            f = features(cmap, *g[jb]); jout = next((j for j in range(jin, len(g)) if not cmap.get_waypoint(carla.Location(*g[j]), project_to_road=True, lane_type=carla.LaneType.Driving).is_junction), len(g) - 1)
            ya = math.degrees(math.atan2(g[jb + 1][1] - g[jb][1], g[jb + 1][0] - g[jb][0]))
            k = min(len(g) - 2, jout + 3); yb = math.degrees(math.atan2(g[k + 1][1] - g[k][1], g[k + 1][0] - g[k][0]))
            d = wrap(yb - ya)
            rec.update(junction=True, junction_at_m=round(gc[jin], 1), gt_man="L" if d < -30 else "R" if d > 30 else "S", **(f or {}))
            recs.append(rec)
    json.dump(recs, open(a.out, "w"), indent=1)
    print(f"{len(recs)} instructions -> {a.out}")
    for r in recs:
        if r.get("junction"):
            print(f"  {r['route'][12:]:40s} #{r['idx']} {r['type']:11s} {r['dir']:8s} GT {r['gt_man']}  lane {r.get('pos','?'):9s} (index {r.get('lane_idx')} of {r.get('n')})  reachable {r.get('man')}")


def part_train(a, maps):
    routes = glob.glob(f"{a.root}/data/simlingo/*/*/*/Town*")
    random.Random(a.seed).shuffle(routes); routes = routes[:a.routes]
    approaches = []  # one record per approach
    for ri, r in enumerate(routes):
        m = re.search(r"Town\d+", Path(r).name)
        if not m: continue
        try: cmap = maps.get(m.group(0))
        except FileNotFoundError: continue
        files = sorted(glob.glob(f"{r}/measurements/*.json.gz"))
        prev_key = None
        for fpath in files[::a.step]:
            try:
                with gzip.open(fpath, "rt") as fh: me = json.load(fh)
            except Exception: continue
            cmd = CMD.get(int(me.get("command", 0)))
            if cmd not in ("L", "R", "S") or me.get("junction") or me.get("speed", 0) < 1.0:
                prev_key = None; continue
            em = me["ego_matrix"]; f = features(cmap, em[0][3], em[1][3], em[2][3])
            if f is None or f["in_junction"]: continue
            key = (cmd, Path(r).name)
            rec = dict(town=m.group(0), route=Path(r).name, cmd=cmd, **f)
            if key == prev_key: approaches[-1] = rec  # same approach: keep the frame closest to the junction
            else: approaches.append(rec)
            prev_key = key
        if ri % 200 == 0: print(f"  [{ri}/{len(routes)}] approaches {len(approaches)}", flush=True)
    json.dump(approaches, open(a.out, "w"))
    print(f"\n{len(approaches)} junction approaches from {len(routes)} routes -> {a.out}")
    def table(rows, title):
        c = collections.Counter((x["pos"], x["cmd"]) for x in rows)
        print(f"\n{title}  (rows: lane position, %: expert command among approaches)")
        for p in ("single", "leftmost", "middle", "rightmost"):
            n = sum(c[(p, k)] for k in "LSR")
            if n: print(f"  {p:9s} n={n:6d}  " + "  ".join(f"{k} {100 * c[(p, k)] / n:5.1f}%" for k in "LSR"))
    table(approaches, "ALL approaches")
    table([x for x in approaches if "S" in x["man"] and "L" in x["man"]], "lanes that can go both straight AND left")
    table([x for x in approaches if x["town"] == "Town12"], "Town12 only")


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("part", choices=["test", "train"])
ap.add_argument("--routes-dir"); ap.add_argument("--root", default="database/simlingo")
ap.add_argument("--routes", type=int, default=3000); ap.add_argument("--step", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args(); maps = M.CarlaMaps()
part_test(a, maps) if a.part == "test" else part_train(a, maps)
