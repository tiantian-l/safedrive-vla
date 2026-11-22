"""Pick training-data junction approaches on lanes that can go straight AND turn, and copy their front-camera
frames for a visual check (11/22, after E16). DIAGNOSTIC, training data only, CPU, no carla needed.

Input: work_dirs/hard_straight/approaches.json (hard_straight.py scan: expert manoeuvre, lane reach L/S/R from
the OpenDRIVE map ~12 m before entry, lane position). Cells = (lane position, reach, expert manoeuvre) among
lanes whose reach has S and a turn; per cell --per-cell approaches (seeded random). For each approach the frames
closest to --at metres before the junction entry are copied (downscaled), plus a JSON index with the
measurement facts (speed, planner command, distance, expert heading change from the reference frame to
2 frames after the junction exit).

    $PY scripts/diag/sample_junction_frames.py --approaches work_dirs/hard_straight/approaches.json \
        --out work_dirs/hard_straight/samples
"""
import argparse, collections, glob, gzip, json, math, os, random
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
CMD = {1: "L", 2: "R", 3: "S", 4: "follow", 5: "lcL", 6: "lcR"}
CELLS = [("leftmost", "LS", "L"), ("leftmost", "LS", "S"),
         ("rightmost", "RS", "R"), ("rightmost", "RS", "S"),
         ("single", "LS", "L"), ("single", "LS", "S"),
         ("single", "RS", "R"), ("single", "RS", "S"),
         ("single", "LRS", "L"), ("single", "LRS", "S"), ("single", "LRS", "R")]


def meas(route_dir, frame):
    with gzip.open(f"{route_dir}/measurements/{frame:04d}.json.gz", "rt") as fh:
        return json.load(fh)


def yaw(me):
    em = me["ego_matrix"]
    return math.degrees(math.atan2(em[1][0], em[0][0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approaches", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="database/simlingo")
    ap.add_argument("--per-cell", type=int, default=6)
    ap.add_argument("--at", type=float, nargs="+", default=[20.0, 10.0, 3.0])
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = os.path.abspath(REPO / a.root)
    recs = json.load(open(a.approaches))
    out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    index = []
    for pos, reach, man in CELLS:
        cell = [r for r in recs if r["pos"] == pos and r["reach"] == reach and r["man"] == man]
        rng.shuffle(cell)
        got = 0
        for r in cell:
            if got >= a.per_cell:
                break
            rd = f"{root}/{r['route']}"
            fr = {int(k): v for k, v in r["frames"].items()}
            try:
                ref = meas(rd, r["ref_frame"])
                last_in = max(k for k, d in fr.items() if d == 0.0) if any(d == 0.0 for d in fr.values()) else max(fr)
                after = meas(rd, last_in + 3)
                dyaw = (yaw(after) - yaw(ref) + 180) % 360 - 180
            except Exception as e:  # noqa: BLE001
                print(f"skip {r['route']}: {e}"); continue
            shots = []
            for at in a.at:
                pre = [(k, d) for k, d in fr.items() if d > 0.0]
                if not pre:
                    continue
                k, d = min(pre, key=lambda kd: abs(kd[1] - at))
                imgs = glob.glob(f"{rd}/rgb/{k:04d}.*")
                if not imgs:
                    continue
                im = cv2.imread(imgs[0])
                h, w = im.shape[:2]
                im = im[: int(h - h * 4.8 / 16)]                       # bonnet crop as in training
                im = cv2.resize(im, (a.width, int(im.shape[0] * a.width / w)))
                name = f"{pos}_{reach}_{man}_{got}_{int(round(at))}m.jpg"
                cv2.imwrite(str(out / "img" / name), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
                me = meas(rd, k)
                shots.append(dict(img=name, frame=k, dist=d, speed=round(float(me.get("speed", 0)), 1),
                                  command=CMD.get(int(me.get("command", 0)), "?")))
            if not shots:
                continue
            index.append(dict(cell=f"{pos}/{reach}/{man}", k=got, town=r["town"], route=r["route"], n_lanes=r["n_lanes"],
                              reach=reach, pos=pos, man=man, expert_dyaw=round(dyaw, 1), shots=shots))
            got += 1
        print(f"{pos:9s} {reach:3s} expert {man}: {len(cell):5d} approaches, sampled {got}")
    json.dump(index, open(out / "index.json", "w"), indent=1)
    # the full table, for the record
    c = collections.Counter((r["pos"], r["reach"], r["man"]) for r in recs)
    lines = ["lane position / reach / expert manoeuvre (all approaches of the scan)"]
    for (p, rc, m), v in sorted(c.items()):
        lines.append(f"  {p:9s} {rc:4s} {m}  {v}")
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(index)} approaches -> {out}")


if __name__ == "__main__":
    main()
