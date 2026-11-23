"""E17 ([26]): rewrite the weather of the paired junction routes with SimLingo's random weather (11/23).

SimLingo's training routes (RenzKa/simlingo dataset_generation/split_route_files.py, non-easy lists) draw every
weather parameter from a list; their released route files (data/simlingo.zip, 23078 routes) have a different
begin / end weather per route and ~30 % sun altitudes below 0 (night). Here: the same lists, one draw per
incoming-lane group (all routes of a group, both route_percentage 0 and 100, get the same weather) so that the
paired recordings look alike before the junction (routes of a group differ in length, a begin -> end blend would
give different weather at the same place). The draw is seeded by the group name -> reproducible, and only the
<weathers> element is touched (routes / waypoints and check/collect.json stay valid; no need to re-run check).
Pure python, no carla.

    python scripts/data/pair_weather.py --pairs work_dirs/pair_junctions
"""
import argparse, collections, json, random, zlib
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PARAMS = ["cloudiness", "precipitation", "precipitation_deposits", "wetness", "wind_intensity",
          "sun_azimuth_angle", "sun_altitude_angle", "fog_density"]
RANGES = {  # split_route_files.py, easy_weather = False
    "cloudiness": [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 40.0, 50.0, 60.0, 80.0, 100.0],
    "precipitation": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
    "precipitation_deposits": [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
    "wetness": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0],
    "wind_intensity": [5.0, 10.0, 25.0, 30.0, 50.0, 60.0, 80.0, 100.0],
    "sun_azimuth_angle": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0],
    "sun_altitude_angle": [-90.0, -45.0, -30.0, -10, -15.0, 5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0,
                           70.0, 80.0, 90.0],
    "fog_density": [0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0, 70.0, 100.0],
}


def group_of(route):
    return route.rsplit("_", 1)[0]          # Town12_j19970_r1007_l-1_S -> Town12_j19970_r1007_l-1


def simlingo_weather(group, seed=0):
    rng = random.Random(zlib.crc32(f"{group}|{seed}".encode()))
    return {p: float(rng.choice(RANGES[p])) for p in PARAMS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="work_dirs/pair_junctions")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pairs = REPO / a.pairs
    xmls = sorted((pairs / "routes").glob("*/*.xml"))
    stats = collections.Counter()
    weathers = {}
    for f in xmls:
        g = group_of(f.stem)
        w = weathers.setdefault(g, simlingo_weather(g, a.seed))
        tree = ET.parse(f)
        r = tree.getroot().find("route")
        old = r.find("weathers")
        idx = list(r).index(old)
        r.remove(old)
        ws = ET.Element("weathers")
        for pct in ("0.0", "100.0"):
            ET.SubElement(ws, "weather", route_percentage=pct, **{k: str(v) for k, v in w.items()})
        r.insert(idx, ws)
        tree.write(f)
        stats["routes"] += 1
    stats["groups"] = len(weathers)
    alt = [w["sun_altitude_angle"] for w in weathers.values()]
    stats["groups sun_altitude < 0 (night)"] = sum(x < 0 for x in alt)
    stats["groups sun_altitude <= 5"] = sum(x <= 5 for x in alt)
    man = pairs / "manifest.json"
    if man.exists():
        m = json.load(open(man))
        for rec in m:
            rec["weather"] = weathers.get(group_of(rec["route"]), rec.get("weather"))
        json.dump(m, open(man, "w"), indent=1)
    json.dump(weathers, open(pairs / "weather.json", "w"), indent=1)
    print(dict(stats))


if __name__ == "__main__":
    main()
