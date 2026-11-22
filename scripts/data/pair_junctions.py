"""[26] Paired junction routes for PDM-Lite data collection (11/22). Offline, CPU; needs the carla package and the
OpenDRIVE files ($CARLA_ROOT), no simulator. Training towns only; never uses a benchmark route except to EXCLUDE
the junctions near it.

Goal: training samples where the same approach (same images up to the junction) continues straight in one
recording and turns in another, so that only the instruction / anchor tells them apart (a junction version of
SimLingo's action dreaming). For every junction of the selected towns:
  connections   junction.get_waypoints(Driving): (start, end) of each lane through the junction, grouped by the
                incoming lane (road_id, lane_id) of start.previous(); manoeuvre from the heading change start->end
                (|d| <= 30 deg S, d < -30 L, d > 30 R; |d| > 150 skipped, U-turn).
  candidates    incoming lanes whose connections include S and at least one turn; lane position (leftmost /
                middle / rightmost / single among same-direction driving lanes) of the incoming lane.
  excluded      junctions within --exclude-m of any waypoint (densified at 2 m) of the benchmark routes in
                --exclude-dirs (CARLA-F, B2D-C, b2d_adv, Bench2Drive), and junctions inside a roundabout-like
                junction with > --max-connections connections.
For each kept incoming lane: one route per manoeuvre (S and each available turn), identical start --approach-m
before the junction on the same lane (shorter if the previous junction is closer; >= --min-approach-m), through
the middle of that connection, --exit-m after the junction. Waypoints: start, every 20 m of the approach, the
lane end before the junction, the connection middle, its end, the exit end (the leaderboard densifies them with
the global route planner). One weather per incoming lane (same for its routes), drawn from a fixed list.

Outputs (--out): routes/<Town>/<town>_j<junction>_r<road>_l<lane>_<M>.xml (leaderboard route XML, no scenarios),
manifest.json (one record per route), summary.txt.

    CARLA_ROOT=$P/carla0915 $PY scripts/data/pair_junctions.py --out work_dirs/pair_junctions \
        --towns Town12 Town01 Town02 Town03 Town04 Town05 Town10HD
"""
import argparse, collections, json, math, random, sys, time

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark" / "metrics"))

WEATHERS = [  # (cloudiness, precipitation, precipitation_deposits, wetness, wind, fog_density, sun_altitude, sun_azimuth)
    (5, 0, 0, 0, 10, 2, 45, -1), (30, 0, 0, 0, 20, 3, 60, 90), (80, 0, 0, 0, 30, 5, 30, 180),
    (60, 40, 40, 50, 40, 10, 40, 45), (90, 80, 80, 90, 80, 20, 20, 270), (10, 0, 0, 0, 10, 2, 10, 120),
    (20, 0, 0, 0, 10, 60, 30, 0), (5, 0, 0, 0, 10, 2, -30, 0), (70, 50, 50, 60, 50, 15, -40, 90),
    (40, 10, 20, 30, 20, 5, 70, 300)]


def wrap(a): return (a + 180.0) % 360.0 - 180.0


def manoeuvre(d):
    if abs(d) > 150: return None
    return "L" if d < -30 else "R" if d > 30 else "S"


def bench_points(dirs, step=2.0):
    """Densified waypoints of the benchmark route XMLs, per town."""
    pts = collections.defaultdict(list)
    for d in dirs:
        for xml in sorted(Path(d).glob("*.xml")):
            for route in ET.parse(xml).getroot().iter("route"):
                town = route.attrib.get("town", "")
                wps = route.find("waypoints")
                if wps is None: continue
                p = [(float(e.attrib["x"]), float(e.attrib["y"])) for e in wps.findall("position")]
                for a, b in zip(p, p[1:] + p[-1:]):
                    n = max(1, int(math.dist(a, b) / step))
                    pts[town] += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n)]
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--towns", nargs="+", default=["Town12", "Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"])
    ap.add_argument("--exclude-dirs", nargs="+", default=[str(REPO / "benchmark/data" / d) for d in ("carla_f", "b2d_c", "b2d_adv", "bench2drive")])
    ap.add_argument("--exclude-m", type=float, default=60.0)
    ap.add_argument("--approach-m", type=float, default=60.0)
    ap.add_argument("--min-approach-m", type=float, default=35.0)
    ap.add_argument("--exit-m", type=float, default=30.0)
    ap.add_argument("--max-connections", type=int, default=16)
    ap.add_argument("--lanes-per-junction", type=int, default=2, help="at most this many incoming lanes per junction")
    ap.add_argument("--max-lanes-per-town", type=int, default=0, help="0 = no cap (sampled, multi-lane first)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import carla
    import carla_f_metrics as M
    maps = M.CarlaMaps()
    rng = random.Random(a.seed)
    t0 = time.time()
    excl = bench_points(a.exclude_dirs)
    print(f"[pair] benchmark points per town: { {k: len(v) for k, v in sorted(excl.items())} }", flush=True)
    out = Path(a.out); (out / "routes").mkdir(parents=True, exist_ok=True)

    def same_dir_lanes(wp):
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

    def walk_back(wp, metres):
        """Same-lane waypoints going backwards from wp (excluded), stopping at the previous junction."""
        pts, w, d = [], wp, 0.0
        while d < metres:
            pr = [p for p in w.previous(2.0) if not p.is_junction]
            if not pr: break
            same = [p for p in pr if (p.road_id, p.lane_id) == (w.road_id, w.lane_id)] or pr
            w = same[0]; d += 2.0; pts.append(w)
        return pts, d

    def walk_fwd(wp, metres):
        pts, w, d = [], wp, 0.0
        while d < metres:
            nx = [n for n in w.next(2.0) if not n.is_junction]
            if not nx: break
            w = nx[0]; d += 2.0; pts.append(w)
        return pts

    def pos(t): return dict(x=f"{t.location.x:.1f}", y=f"{t.location.y:.1f}", z=f"{t.location.z + 0.5:.1f}")

    manifest, skip, n_route = [], collections.Counter(), 0
    for town in a.towns:
        try:
            cmap = maps.get(town)
        except FileNotFoundError:
            print(f"[pair] no OpenDRIVE for {town}, skipped", flush=True); continue
        ex = excl.get(town, []) + (excl.get(town.replace("HD", ""), []) if town.endswith("HD") else []) + excl.get(town + "HD", [])
        ex = np.asarray(ex, dtype=np.float64).reshape(-1, 2)
        junctions = {}
        for s, e in cmap.get_topology():
            for w in (s, e):
                if w.is_junction:
                    j = w.get_junction(); junctions[j.id] = j
        print(f"[pair] {town}: {len(junctions)} junctions ({time.time() - t0:.0f} s)", flush=True)
        town_lanes = []
        for jid, j in junctions.items():
            c = j.bounding_box.location
            if len(ex) and float(np.min(np.hypot(ex[:, 0] - c.x, ex[:, 1] - c.y))) < a.exclude_m:
                skip["near a benchmark route"] += 1; continue
            conns = j.get_waypoints(carla.LaneType.Driving)
            if len(conns) > a.max_connections:
                skip["too many connections"] += 1; continue
            by_in = collections.defaultdict(list)
            for s, e in conns:
                pr = s.previous(1.0)
                if not pr: continue
                inc = pr[0]
                if inc.is_junction: continue
                m = manoeuvre(wrap(e.transform.rotation.yaw - s.transform.rotation.yaw))
                if m is None: continue
                by_in[(inc.road_id, inc.lane_id)].append((m, s, e, inc))
            cand = []
            for key, lst in by_in.items():
                mans = {m for m, *_ in lst}
                if "S" not in mans or not (mans & {"L", "R"}):
                    continue
                inc = lst[0][3]
                idx, n = same_dir_lanes(inc)
                p = "single" if n == 1 else "leftmost" if idx == 0 else "rightmost" if idx == n - 1 else "middle"
                cand.append((key, lst, inc, p, n, "".join(sorted(mans))))
            rng.shuffle(cand)
            cand.sort(key=lambda c: c[4] == 1)   # multi-lane approaches first
            for key, lst, inc, p, n, reach in cand[: a.lanes_per_junction]:
                back, dist = walk_back(inc, a.approach_m)
                if dist < a.min_approach_m:
                    skip["approach too short"] += 1; continue
                town_lanes.append((jid, key, lst, inc, p, n, reach, back, dist))
        if a.max_lanes_per_town and len(town_lanes) > a.max_lanes_per_town:
            rng.shuffle(town_lanes); town_lanes.sort(key=lambda t: t[5] == 1); town_lanes = town_lanes[: a.max_lanes_per_town]
        (out / "routes" / town).mkdir(parents=True, exist_ok=True)
        for jid, key, lst, inc, p, n, reach, back, dist in town_lanes:
            weather = rng.choice(WEATHERS)
            approach = list(reversed(back))
            done = set()
            for m, s, e, _ in lst:
                if m in done: continue          # one connection per manoeuvre
                done.add(m)
                mid = s.next(max(1.0, s.transform.location.distance(e.transform.location) / 2))
                exit_pts = walk_fwd(e, a.exit_m)
                if not exit_pts:
                    skip["no exit road"] += 1; continue
                wps = [approach[0]] + approach[10::10] + [inc] + ([mid[0]] if mid else []) + [e, exit_pts[-1]]
                rid = n_route; n_route += 1
                root = ET.Element("routes"); r = ET.SubElement(root, "route", id=str(rid), town=town)
                ws = ET.SubElement(r, "weathers")
                for pct in ("0.0", "100.0"):
                    c, pr_, pd, wt, wi, fd, sa, sz = weather
                    ET.SubElement(ws, "weather", cloudiness=str(c), fog_density=str(fd), precipitation=str(pr_),
                                  precipitation_deposits=str(pd), route_percentage=pct, sun_altitude_angle=str(sa),
                                  sun_azimuth_angle=str(sz), wetness=str(wt), wind_intensity=str(wi))
                wp_el = ET.SubElement(r, "waypoints")
                for w in wps:
                    ET.SubElement(wp_el, "position", **pos(w.transform))
                ET.SubElement(r, "scenarios")
                name = f"{town}_j{jid}_r{key[0]}_l{key[1]}_{m}"
                ET.ElementTree(root).write(out / "routes" / town / f"{name}.xml")
                manifest.append(dict(route=name, id=rid, town=town, junction=jid, road=key[0], lane=key[1], pos=p,
                                     n_lanes=n, reach=reach, man=m, approach_m=round(dist, 1),
                                     dyaw=round(wrap(e.transform.rotation.yaw - s.transform.rotation.yaw), 1),
                                     weather=list(weather)))
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    lines = [f"{len(manifest)} routes, {len({(r['town'], r['junction'], r['road'], r['lane']) for r in manifest})} approaches "
             f"({time.time() - t0:.0f} s); skipped junctions / lanes: {dict(skip)}", ""]
    c = collections.Counter((r["town"], r["pos"], r["reach"], r["man"]) for r in manifest)
    for k, v in sorted(c.items()):
        lines.append(f"  {k[0]:9s} {k[1]:9s} reach {k[2]:3s} {k[3]}  {v}")
    text = "\n".join(lines); print(text); (out / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
