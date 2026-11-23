"""[26] / E17 check of the paired junction routes (11/23). Offline, CPU; carla package + OpenDRIVE ($CARLA_ROOT),
no simulator.

Every route XML of pair_junctions.py is densified exactly as the leaderboard does (GlobalRoutePlanner.trace_route
between consecutive XML positions, 1 m hops). Per route: the manoeuvre of the traced path (heading change from the
start lane to the end, |d| <= 30 deg S, < -30 L, > 30 R), whether it matches the intended one, the traced length vs
the straight-line sum (a detour means the planner took another way), the lane (road, lane) of the first trace point,
and the junction ids the trace passes. Per group (same start lane): do all routes start on the same lane and pass
the same first junction?
Writes check/clean_groups.json (groups to collect). Also draws --n-plots groups (stratified by lane position / reach) on top of the lane centre lines around the
junction: work_dirs/pair_junctions/check/<group>.png (grey = driving lanes, colours = traced routes, dot = start).

    CARLA_ROOT=$P/carla0915 $PY scripts/data/check_pair_routes.py --pairs work_dirs/pair_junctions
"""
import argparse, collections, json, math, os, random, sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark" / "metrics"))
sys.path.insert(0, os.path.join(os.environ.get("CARLA_ROOT", ""), "PythonAPI", "carla"))


def wrap(a): return (a + 180.0) % 360.0 - 180.0
def man(d): return None if abs(d) > 150 else "L" if d < -30 else "R" if d > 30 else "S"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True); ap.add_argument("--n-plots", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import carla
    import carla_f_metrics as M
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    root = Path(a.pairs)
    manifest = json.load(open(root / "manifest.json"))
    maps, planners = M.CarlaMaps(), {}
    res = []
    for r in manifest:
        town = r["town"]
        if town not in planners:
            planners[town] = GlobalRoutePlanner(maps.get(town), 1.0)
            print(f"[check] planner {town}", flush=True)
        grp = planners[town]
        xml = root / "routes" / town / f"{r['route']}.xml"
        pos = [carla.Location(float(p.attrib["x"]), float(p.attrib["y"]), float(p.attrib["z"]))
               for p in ET.parse(xml).getroot().iter("position")]
        trace = []
        for s, e in zip(pos, pos[1:]):
            seg = grp.trace_route(s, e)
            trace += seg if not trace else seg[1:]
        wps = [w for w, _ in trace]
        straight = sum(s.distance(e) for s, e in zip(pos, pos[1:]))
        length = sum(w1.transform.location.distance(w2.transform.location) for w1, w2 in zip(wps, wps[1:]))
        juncs = []
        for w in wps:
            if w.is_junction and (not juncs or juncs[-1] != w.get_junction().id):
                juncs.append(w.get_junction().id)
        d = wrap(wps[-1].transform.rotation.yaw - wps[0].transform.rotation.yaw) if wps else float("nan")
        res.append(dict(route=r["route"], group=f"{town}_j{r['junction']}_r{r['road']}_l{r['lane']}", man=r["man"],
                        traced=man(d), dyaw=round(d, 1), ok=man(d) == r["man"], length=round(length, 1),
                        straight=round(straight, 1), detour=length > 1.3 * straight + 5,
                        start_lane=[wps[0].road_id, wps[0].lane_id] if wps else None, junctions=juncs, pos=r["pos"],
                        reach=r["reach"], xy=[[round(w.transform.location.x, 1), round(w.transform.location.y, 1)] for w in wps[::2]]))
    groups = collections.defaultdict(list)
    for x in res: groups[x["group"]].append(x)
    bad_group = [g for g, xs in groups.items() if len({tuple(x["start_lane"]) for x in xs}) > 1
                 or len({x["junctions"][0] if x["junctions"] else None for x in xs}) > 1]
    lines = [f"{len(res)} routes in {len(groups)} groups",
             f"traced manoeuvre == intended: {sum(x['ok'] for x in res)} / {len(res)}",
             f"detours (traced > 1.3 x straight + 5 m): {sum(x['detour'] for x in res)}",
             f"groups whose routes start on different lanes or pass a different first junction: {len(bad_group)}",
             f"groups fully clean (all routes ok, no detour, same start lane & junction): "
             f"{sum(1 for g, xs in groups.items() if g not in bad_group and all(x['ok'] and not x['detour'] for x in xs))}",
             "", "mismatches (first 30):"]
    for x in [x for x in res if not x["ok"] or x["detour"]][:30]:
        lines.append(f"  {x['route']:40s} intended {x['man']} traced {x['traced']} dyaw {x['dyaw']:+.0f} len {x['length']} / {x['straight']}")
    text = "\n".join(lines); print(text)
    clean = sorted(g for g, xs in groups.items() if g not in bad_group and all(x["ok"] and not x["detour"] for x in xs))
    (root / "check").mkdir(exist_ok=True)
    json.dump(clean, open(root / "check" / "clean_groups.json", "w"))
    (root / "check" / "summary.txt").write_text(text + "\n")
    json.dump(res, open(root / "check" / "check.json", "w"))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = random.Random(a.seed)
    strata = collections.defaultdict(list)
    for g, xs in groups.items(): strata[(xs[0]["pos"], xs[0]["reach"])].append(g)
    pick = []
    while len(pick) < a.n_plots and any(strata.values()):
        for k in sorted(strata):
            if strata[k] and len(pick) < a.n_plots:
                pick.append(strata[k].pop(rng.randrange(len(strata[k]))))
    lane_cache = {}
    col = {"L": "tab:blue", "S": "tab:green", "R": "tab:red"}
    for g in pick:
        xs = groups[g]; town = g.split("_j")[0]
        if town not in lane_cache:
            lane_cache[town] = [(w.transform.location.x, w.transform.location.y) for w in maps.get(town).generate_waypoints(2.0)]
        allxy = [p for x in xs for p in x["xy"]]
        x0, x1 = min(p[0] for p in allxy) - 20, max(p[0] for p in allxy) + 20
        y0, y1 = min(p[1] for p in allxy) - 20, max(p[1] for p in allxy) + 20
        fig, ax = plt.subplots(figsize=(6, 6))
        ln = [p for p in lane_cache[town] if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
        if ln: ax.scatter([p[0] for p in ln], [p[1] for p in ln], s=1, c="0.8")
        for x in xs:
            ax.plot([p[0] for p in x["xy"]], [p[1] for p in x["xy"]], c=col.get(x["man"], "k"), lw=2,
                    label=f"{x['man']} (traced {x['traced']}{', detour' if x['detour'] else ''})")
            ax.scatter([x["xy"][0][0]], [x["xy"][0][1]], c="k", s=30, zorder=5)
        ax.set_aspect("equal"); ax.invert_yaxis()   # CARLA y points down on a map image
        ax.set_title(f"{g}\n{xs[0]['pos']} reach {xs[0]['reach']}", fontsize=9); ax.legend(fontsize=8)
        fig.savefig(root / "check" / f"{g}.png", dpi=90, bbox_inches="tight"); plt.close(fig)
    print(f"{len(pick)} plots -> {root / 'check'}")


if __name__ == "__main__":
    main()
