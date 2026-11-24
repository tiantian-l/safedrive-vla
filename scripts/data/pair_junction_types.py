"""E17 ([26]): what kind of junctions are the paired routes at? (11/24) CPU, OpenDRIVE only (no carla, no simulator).

The trial group (Town12_j19970) was a rural STOP junction without traffic; the CARLA-F straight failures (17, 55,
218, 25, 56) are city junctions with traffic lights and several lanes. Per approach (incoming road / lane into a
junction), from the .xodr of the town:
  control   light = the junction has a <controller> or a dynamic signal (traffic light, type 1000001) stands on the
            incoming road or on a connecting road of the junction; else stop (206) / yield (205) on the incoming
            road; else none
  arms      number of distinct roads entering the junction
  lanes     driving lanes on the incoming road at the junction end: same direction / both directions
Junction of an approach = the junction its incoming road links to on the side the lane drives to (lane < 0:
successor, lane > 0: predecessor); for the paired routes it is checked against the junction id in the manifest.

Inputs: work_dirs/pair_junctions/{manifest.json, check/collect.json}; for comparison (statistics only, never for
training) the CARLA-F approach lanes of work_dirs/diag/lane_test.json (lane_position.py) if present.
Output: <pairs>/junction_types.json (per group), junction_types.txt (tables), trial2_groups.txt (one paired
group per town / control / lanes cell, traffic-light junctions first; for collect_pairs.py --groups @file),
redlight_groups.txt (--red traffic-light groups for the red-light data, collected with --no-all-green).

    CARLA_ROOT=$P/carla0915 python scripts/data/pair_junction_types.py --pairs work_dirs/pair_junctions
"""
import argparse, collections, json, sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark" / "generation"))
from opendrive import _default_xodr_search_roots, _resolve_xodr_path  # noqa: E402

LIGHT, STOP, YIELD = {"1000001"}, {"206"}, {"205"}


class Town:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.roads, self.junctions, self.ctrl_signals = {}, {}, {}
        for c in root.findall("controller"):
            self.ctrl_signals[c.get("id")] = [x.get("signalId") for x in c.findall("control")]
        for j in root.findall("junction"):
            self.junctions[j.get("id")] = dict(
                controllers=[c.get("id") for c in j.findall("controller")],
                incoming={c.get("incomingRoad") for c in j.findall("connection")},
                connecting={c.get("connectingRoad") for c in j.findall("connection")})
        for r in root.findall("road"):
            link = r.find("link")
            ends = {}
            if link is not None:
                for side in ("predecessor", "successor"):
                    e = link.find(side)
                    if e is not None:
                        ends[side] = (e.get("elementType"), e.get("elementId"))
            sigs = [(s.get("type"), s.get("dynamic")) for s in r.iter("signal")]
            secs = r.find("lanes").findall("laneSection")
            def driving(sec, side):
                el = sec.find(side)
                return 0 if el is None else sum(l.get("type") == "driving" for l in el.findall("lane"))
            lanes = {k: (driving(sec, "left"), driving(sec, "right")) for k, sec in (("first", secs[0]), ("last", secs[-1]))}
            self.roads[r.get("id")] = dict(junction=r.get("junction"), ends=ends, signals=sigs, lanes=lanes)

    def approach(self, road, lane):
        r = self.roads.get(str(road))
        if r is None:
            return None
        side, sec = ("successor", "last") if int(lane) < 0 else ("predecessor", "first")
        end = r["ends"].get(side)
        if not end or end[0] != "junction":
            return dict(junction=None)
        jid = end[1]; j = self.junctions.get(jid, {})
        sig = list(r["signals"])
        for cr in j.get("connecting", ()):
            sig += self.roads.get(cr, {}).get("signals", [])
        types = {t for t, _ in sig}
        light = bool(j.get("controllers")) or bool(types & LIGHT) or any(d == "yes" for _, d in sig)
        own = {t for t, _ in r["signals"]}
        control = "light" if light else "stop" if own & STOP else "yield" if own & YIELD else "none"
        left, right = r["lanes"][sec]
        same = right if int(lane) < 0 else left
        return dict(junction=int(jid), control=control, arms=len(j.get("incoming", ())), lanes_same=same,
                    lanes_total=left + right)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="work_dirs/pair_junctions")
    ap.add_argument("--test", default="work_dirs/diag/lane_test.json")
    ap.add_argument("--trial", type=int, default=8, help="groups for the second trial (trial2_groups.txt)")
    ap.add_argument("--red", type=int, default=120, help="traffic-light groups for the red-light data (redlight_groups.txt)")
    a = ap.parse_args()
    pairs = REPO / a.pairs
    man = {r["route"]: r for r in json.load(open(pairs / "manifest.json"))}
    collect = json.load(open(pairs / "check" / "collect.json"))
    roots, towns = _default_xodr_search_roots(), {}

    def town(t):
        if t not in towns:
            p = _resolve_xodr_path(t, roots)
            if p is None:
                sys.exit(f"no .xodr for {t} under {roots} (set CARLA_ROOT)")
            towns[t] = Town(p)
        return towns[t]

    groups, bad_j = {}, 0
    for kind in ("paired", "unpaired"):
        for g, names in collect[kind].items():
            r = man[names[0]]
            f = town(r["town"]).approach(r["road"], r["lane"]) or {}
            if f.get("junction") != r["junction"]:
                bad_j += 1
            groups[g] = dict(kind=kind, town=r["town"], pos=r["pos"], reach=r["reach"],
                             routes=[n.rsplit("_", 1)[1] for n in names], **f)
    lines = [f"{len(groups)} groups; junction id from xodr != manifest: {bad_j}", ""]

    def table(title, items):
        lines.append(title)
        c = collections.Counter((x["town"] if x["town"] == "Town12" else "old towns", x.get("control"),
                                 "1 lane" if x.get("lanes_same", 0) <= 1 else "2+ lanes") for x in items)
        n = len(items) or 1
        for k, v in sorted(c.items(), key=lambda kv: (kv[0][0], -kv[1])):
            lines.append(f"  {k[0]:9s} {str(k[1]):6s} {k[2]:8s} {v:5d}  {100 * v / n:5.1f}%")
        arms = collections.Counter(x.get("arms") for x in items)
        lines.append("  arms: " + ", ".join(f"{k}: {v}" for k, v in sorted(arms.items(), key=lambda kv: str(kv[0]))))
        lines.append("")

    table("E17 paired groups (town / control / same-direction lanes):", [x for x in groups.values() if x["kind"] == "paired"])
    test = REPO / a.test
    tests = []
    if test.exists():
        for t in json.load(open(test)):
            if not t.get("junction"):
                continue
            f = town(t["town"]).approach(t["road"], t["lane"]) or {}
            tests.append(dict(town=t["town"], route=t["route"], idx=t["idx"], type=t["type"], dir=t.get("dir"),
                              gt=t.get("gt_man"), **f))
        table("CARLA-F junction instructions (comparison only):", tests)
        lines.append("CARLA-F straight / lane instructions at junctions:")
        for x in tests:
            if x.get("gt") == "S":
                lines.append(f"  {x['route'].replace('bench2drive_', '').replace('_language_rebuilt', ''):28s} {x['town']:8s} "
                             f"control {x.get('control')}, arms {x.get('arms')}, lanes {x.get('lanes_same')}/{x.get('lanes_total')}")
    else:
        lines.append(f"({a.test} not found: no CARLA-F comparison)")
    # second trial: one paired group per (town class, control, lanes) cell, light junctions first, max --trial groups
    import random
    rng = random.Random(0)
    cells = collections.defaultdict(list)
    for g, x in groups.items():
        if x["kind"] == "paired":
            cells[(x["town"] == "Town12", x.get("control"), x.get("lanes_same", 0) > 1)].append(g)
    order = sorted(cells, key=lambda k: (k[1] != "light", not k[0], not k[2]))
    pick = [rng.choice(sorted(cells[k])) for k in order][: a.trial]
    (pairs / "trial2_groups.txt").write_text(",".join(pick) + "\n")
    # red-light data ([C], 11/24): traffic-light groups collected with normal light cycles (collect_pairs --no-all-green);
    # the expert stops at red on part of them. Town12 and old towns, paired and unpaired, random --red groups.
    light = sorted(g for g, x in groups.items() if x.get("control") == "light")
    rng2 = random.Random(1)
    red = sorted(rng2.sample(light, min(a.red, len(light))))
    (pairs / "redlight_groups.txt").write_text(",".join(red) + "\n")
    lines += ["", f"red-light groups: {len(red)} of {len(light)} traffic-light groups, "
              f"{sum(len(groups[g]['routes']) for g in red)} routes (redlight_groups.txt)"]
    lines += ["", f"second trial groups ({len(pick)}, in trial2_groups.txt):"]
    lines += [f"  {g}: {groups[g]['town']} {groups[g].get('control')} arms {groups[g].get('arms')} lanes "
              f"{groups[g].get('lanes_same')}/{groups[g].get('lanes_total')} {groups[g]['pos']} routes {groups[g]['routes']}"
              for g in pick]
    text = "\n".join(lines)
    print(text)
    (pairs / "junction_types.txt").write_text(text + "\n")
    json.dump(dict(groups=groups, carla_f=tests), open(pairs / "junction_types.json", "w"), indent=1)


if __name__ == "__main__":
    main()
