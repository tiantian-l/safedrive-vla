"""Why a lane-follow / lane-change instruction failed although the trajectory is on the GT.

Re-computes carla_f_metrics' lane check for chosen routes and prints, for the GT tail and
the ego tail, every sample's (road_id, lane_id, is_junction) plus the two majorities.
Also shows what the check gives if the ego tail were taken a few metres further along
(the GT tail is cut at the ego's driven distance). Offline: needs only the ``carla`` package
and the OpenDRIVE files (CARLA_ROOT), no simulator.

    CARLA_ROOT=$P/carla0915 $PY scripts/diag/lane_check.py work_dirs/track3/carla_f \
        --routes-dir ../track3/routes/carla_f --routes 212_language,41_language
"""
import argparse, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark" / "metrics"))
import carla  # noqa: E402
import carla_f_metrics as M  # noqa: E402


def lanes(cmap, pts):
    out = []
    for x, y, z in pts:
        wp = cmap.get_waypoint(carla.Location(x=x, y=y, z=z), project_to_road=True, lane_type=carla.LaneType.Driving)
        out.append(None if wp is None else (wp.road_id, wp.lane_id, wp.is_junction))
    return out


def show(tag, pts, ls):
    print(f"  {tag}: {len(pts)} samples")
    step = max(1, len(pts) // 12)
    for p, l in list(zip(pts, ls))[::step]:
        print(f"     ({p[0]:9.2f},{p[1]:9.2f})  road {l[0] if l else '-':>5} lane {l[1] if l else '-':>3} {'JUNCTION' if l and l[2] else ''}")
    c = Counter((l[0], l[1]) for l in ls if l and not l[2])
    print(f"     majority (non-junction): {c.most_common(3)}")


ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--routes-dir", required=True); ap.add_argument("--routes", required=True)
a = ap.parse_args()
maps = M.CarlaMaps()
for xml in sorted(Path(a.routes_dir).glob("*.xml")):
    if not any(r in xml.stem for r in a.routes.split(",")):
        continue
    route, gt, ins = M.parse_route(xml)
    traj = M.load_trajectory(Path(a.run) / "viz" / xml.stem)
    cmap = maps.get(route.attrib["town"])
    tc, gc = M.cumulative_distances(traj), M.cumulative_distances(gt)
    print(f"\n=== {xml.stem}  town {route.attrib['town']}  driven {tc[-1]:.1f} m  GT {gc[-1]:.1f} m  GT waypoints {len(gt)}")
    big = [(round(gc[i], 1), round(gc[i + 1], 1)) for i in range(len(gc) - 1) if gc[i + 1] - gc[i] > 6]
    if big: print(f"  GT waypoint gaps > 6 m (from, to): {big}")
    for i, x in enumerate(ins):
        kind = x["expected_behavior"].get("type")
        if kind not in ("lane_follow", "lane_change"):
            continue
        s = M.trigger_distance(x) or 0.0
        nxt = M.trigger_distance(ins[i + 1]) if i + 1 < len(ins) else None
        end = nxt if nxt is not None else tc[-1]
        te, ge = min(end, tc[-1]), min(end, gc[-1])
        def tail(cum, e): return max(s, e - M.LANE_TAIL_FRACTION * max(e - s, 0.0)), e
        gs, ge2 = tail(gc, max(s, ge)); ts, te2 = tail(tc, max(s, te))
        print(f" #{i} {kind} {x['expected_behavior'].get('direction','')}  window {s:.1f}-{end:.1f} m;  GT tail [{gs:.1f},{ge2:.1f}]  ego tail [{ts:.1f},{te2:.1f}]")
        gp = M.segment(gt, gc, gs, ge2); tp = M.segment(traj, tc, ts, te2)
        show("GT tail", gp, lanes(cmap, gp)); show("ego tail", tp, lanes(cmap, tp))
        e = M.majority_lane(cmap, gp); g = M.majority_lane(cmap, tp)
        print(f"  -> expected {e}  actual {g}  passed {e is not None and e == g}")
        for extra in (5, 10, 15):  # GT tail if the ego had driven further before the route ended
            gpx = M.segment(gt, gc, *tail(gc, min(ge + extra, gc[-1])))
            print(f"     GT tail if ego drove +{extra} m: {M.majority_lane(cmap, gpx)}")
