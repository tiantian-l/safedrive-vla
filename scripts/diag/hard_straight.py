"""Hard straight examples (11/22): does the released model already get "looks like a turn, goes straight" right
on the training data? DIAGNOSTIC, training data only (SimLingo), not part of the agent.

Question: the model turns at some competition junctions where it should go straight (17, 55, 56, 151, 218 ...),
seemingly from the image (leftmost lane / turn arrow / open corner). SimLingo has such junctions too: the
ego lane can turn but the expert goes straight. If the model already predicts straight on those *training*
frames, retraining on the same data is unlikely to help (the failure is a generalisation gap); if it turns
there too, upsampling them with a consistent anchor / heavier path loss is worth a fine-tune.

Part "scan" (CPU; needs the carla package + OpenDRIVE files under $CARLA_ROOT, no simulator):
  every junction approach of the relabel index (scripts/data/build_relabel_index.py: the expert's manoeuvre
  L / S / R and metres to the junction entry) gets the lane features of the ego lane ~12 m before entry
  (same functions as lane_position.py): manoeuvres reachable from that lane (L/S/R), lane position.
  Groups:  hardS = expert straight, lane can also turn (reachable has L or R)
           easyS = expert straight, lane can only go straight
           L / R = expert turns (controls)
Part "eval" (GPU): released model on frames of each group (relabel buckets, train and val splits), with the
  competition-style sentence of the expert's own manoeuvre and the world-model anchor either
  "test" = the manoeuvre's anchor (what the agent does in the competition) or
  "orig" = the planner command of the frame (what training saw; "follow the road" 10-30 m before entry).
  Per frame: path-head heading 15 m ahead vs the expert's, path loss, action loss.

    CARLA_ROOT=$P/carla0915 $PY scripts/diag/hard_straight.py scan --out work_dirs/hard_straight/approaches.json
    $PY scripts/diag/hard_straight.py eval --approaches work_dirs/hard_straight/approaches.json \
        --ckpt ckpts/safedrive-vla/safedrive_vla/model.ckpt --out work_dirs/hard_straight/eval.json
"""
import argparse, collections, gzip, json, math, os, random, re, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "diag"))

BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
MAN_COMMAND = {"L": 1, "R": 2, "S": 3}
MAN_SENTENCE = {"L": "left", "S": "straight", "R": "right"}
GROUPS = ("hardS", "easyS", "L", "R")


def wrap(a): return (a + 180.0) % 360.0 - 180.0


def bin_of(dist):
    for k, (lo, hi) in enumerate(BINS):
        if lo <= dist < hi or (hi == 30 and dist == 30):
            return k
    return None


def approaches_of(frames):
    """{frame(str): [man, dist]} of one route -> list of approaches, each {frame(int): dist}, in frame order.
    Within an approach the distance to the entry falls (0 inside the junction); a jump up or a frame gap
    starts the next one."""
    out, cur, prev_f, prev_d, prev_m = [], None, None, None, None
    for f in sorted(frames, key=int):
        man, dist = frames[f][0], float(frames[f][1])
        fi = int(f)
        new = cur is None or fi != prev_f + 1 or man != prev_m or (dist > prev_d and not (prev_d == 0.0 and dist == 0.0))
        if new:
            cur = dict(man=man, frames={}); out.append(cur)
        cur["frames"][fi] = dist
        prev_f, prev_d, prev_m = fi, dist, man
    return out


def group_of(man, reach):
    if man in ("L", "R"):
        return man
    if man == "S" and reach:
        return "hardS" if ("L" in reach or "R" in reach) else "easyS"
    return None


# ---------------------------------------------------------------- scan (lane features; from lane_position.py)
def scan(a):
    sys.path.insert(0, str(REPO / "benchmark" / "metrics"))
    import carla  # noqa: F401
    import carla_f_metrics as M

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

    def manoeuvres(wp):
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
            for _ in range(60):
                if not e.is_junction: break
                n2 = e.next(2.0)
                if not n2: break
                e = n2[0]
            d = wrap(e.transform.rotation.yaw - y0)
            out.add("L" if d < -30 else "R" if d > 30 else "S")
        return "".join(sorted(out))

    import carla
    maps = M.CarlaMaps()
    root = Path(os.path.abspath(REPO / a.root))
    index = json.load(open(root / "relabel_index.json"))["routes"]
    keys = sorted(index)
    if a.routes:
        random.Random(0).shuffle(keys); keys = keys[: a.routes]
    by_town = collections.defaultdict(list)
    for k in keys:
        m = re.search(r"Town\d+", Path(k).name)
        if m: by_town[m.group(0)].append(k)
    recs, t0, cnt, skip = [], time.time(), 0, collections.Counter()
    for town, ks in sorted(by_town.items()):
        try:
            cmap = maps.get(town)
        except FileNotFoundError:
            skip[f"no map {town}"] += len(ks); continue
        print(f"[scan] {town}: {len(ks)} routes ({time.time() - t0:.0f} s)", flush=True)
        for k in ks:
            for ap in approaches_of(index[k]):
                fr = {f: d for f, d in ap["frames"].items() if d <= 30.0}
                pre = [(f, d) for f, d in fr.items() if d >= 3.0]
                if not pre:
                    skip["no frame >= 3 m before entry"] += 1; continue
                ref, ref_d = min(pre, key=lambda fd: abs(fd[1] - a.ref_m))
                mp = root / k / "measurements" / f"{ref:04d}.json.gz"
                try:
                    with gzip.open(mp, "rt") as fh: me = json.load(fh)
                except Exception:
                    skip["measurement unreadable"] += 1; continue
                em = me["ego_matrix"]
                wp = cmap.get_waypoint(carla.Location(x=em[0][3], y=em[1][3], z=em[2][3]), project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
                if wp is None or wp.is_junction:
                    skip["ref frame off road / in junction"] += 1; continue
                idx, n = same_dir_lanes(wp)
                reach = manoeuvres(wp)
                pos = "single" if n == 1 else "leftmost" if idx == 0 else "rightmost" if idx == n - 1 else "middle"
                recs.append(dict(route=k, town=town, man=ap["man"], reach=reach, pos=pos, n_lanes=n, ref_frame=ref,
                                 ref_m=round(ref_d, 1), group=group_of(ap["man"], reach),
                                 frames={str(f): d for f, d in sorted(fr.items())}))
            cnt += 1
            if cnt % 500 == 0:
                print(f"[scan] {cnt} routes, {len(recs)} approaches ({time.time() - t0:.0f} s)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(a.out, "w"))
    lines = [f"{len(recs)} approaches from {cnt} routes ({time.time() - t0:.0f} s); skipped {dict(skip)}", ""]
    lines.append("expert manoeuvre x lane reach (approaches)")
    c = collections.Counter((r["man"], r["reach"]) for r in recs)
    for (m, rch), v in sorted(c.items()):
        lines.append(f"  expert {m}  reach {rch or '?':4s} {v:6d}")
    lines.append("\ngroups (approaches; frames <= 30 m)")
    for g in GROUPS:
        rs = [r for r in recs if r["group"] == g]
        lines.append(f"  {g:6s} {len(rs):6d} approaches, {sum(len(r['frames']) for r in rs):7d} frames")
    lines.append("\nhardS by lane position / reach / town")
    rs = [r for r in recs if r["group"] == "hardS"]
    for key in ("pos", "reach", "town"):
        lines.append(f"  {key}: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(r[key] for r in rs).most_common()))
    lines.append("\nstraight share among approaches whose lane can turn, by lane position")
    for p in ("leftmost", "middle", "rightmost", "single"):
        rr = [r for r in recs if r["pos"] == p and ("L" in r["reach"] or "R" in r["reach"]) and "S" in r["reach"]]
        if rr:
            cc = collections.Counter(r["man"] for r in rr)
            lines.append(f"  {p:9s} n={len(rr):5d}  " + "  ".join(f"{m} {100 * cc[m] / len(rr):5.1f}%" for m in "LSR"))
    text = "\n".join(lines)
    print(text, flush=True)
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


# ---------------------------------------------------------------- eval (released model on the groups)
def heading15(path):
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or len(p) == 0:
        return float("nan")
    x, y = p[min(15, len(p)) - 1, :2]
    return math.degrees(math.atan2(y, x))


def evaluate_main(a):
    import torch
    from omegaconf import OmegaConf
    from probe import load
    from safedrive_vla.constants import COMMAND_TO_META_COMMAND, META_COMMAND_TO_INDEX, TASK_PROMPT
    from safedrive_vla.data.datamodule import LMDRIVE_TEMPLATES, SafeDriveDataModule
    from safedrive_vla.data.dataset import DrivingDataset
    from safedrive_vla.data.driving_mode import classify_driving_mode

    t0 = time.time()
    recs = json.load(open(a.approaches))
    want = {}                                    # (route, frame) -> (group, dist, approach record)
    for r in recs:
        if r["group"] in GROUPS:
            for f, d in r["frames"].items():
                want[(r["route"], int(f))] = (r["group"], float(d), r)
    device = torch.device("cuda")
    model, processor, cfg = load(a.ckpt, device)
    root = Path(os.path.abspath(REPO / a.root))
    wm = OmegaConf.to_container(cfg.model.world_model, resolve=True) if cfg.model.get("world_model") is not None else None
    kw = dict(data_root=str(root), data_scale=1.0, lmdrive_templates=LMDRIVE_TEMPLATES,
              codebook_path=cfg.model.action_token.codebook_path, use_mode_token=cfg.model.use_mode_token, world_model=wm)
    sentences = json.load(open(REPO / "safedrive_vla" / "assets" / "relabel_sentences.json"))
    dm = SafeDriveDataModule.__new__(SafeDriveDataModule)
    dm.processor, dm.tokenizer = processor, processor.tokenizer
    dm.action_ids = (int(model.action_start_id), int(model.action_end_id))
    dm.wm_enabled, dm.world_model = wm is not None, dict(wm or {})

    rng = random.Random(a.seed)
    cand = collections.defaultdict(list)         # (split, group, bin) -> [(ds, i, dist, rec)]
    for split in ("train", "val"):
        for bucket in ("relabel_straight", "relabel_turn"):
            ds = DrivingDataset(split=split, bucket_name=bucket, **kw)
            ds.tfs = lambda image: image
            for i in range(len(ds)):
                mpath = str(ds.measurements[i], encoding="utf-8")
                key = os.path.relpath(os.path.dirname(mpath), str(root))   # measurements dir -> route dir (Relabeler.route_key)
                hit = want.get((key, int(ds.sample_start[i])))
                if hit is None:
                    continue
                g, dist, rec = hit
                b = bin_of(dist)
                if b is not None:
                    cand[(split, g, b)].append((ds, i, dist, rec))
    print(f"[eval] candidates: { {f'{s}/{g}/{BINS[b]}': len(v) for (s, g, b), v in sorted(cand.items())} } "
          f"({time.time() - t0:.0f} s)", flush=True)

    def build(ds, i, sentence, anchor_cmd):
        measurements, current, mpath = ds.load_measurements(i)
        traj = ds.load_trajectory(measurements, current, 0.0, 0.0)
        tp = ds.augment_target_point(np.array(current["target_point"]), 0.0, 0.0)
        ntp = ds.augment_target_point(np.array(current["target_point_next"]), 0.0, 0.0)
        speed = current["speed"]
        prompt = f"Current speed: {round(speed, 1)} m/s. Command: {sentence}. {TASK_PROMPT}".replace("..", ".")
        meta_name = COMMAND_TO_META_COMMAND.get(anchor_cmd)
        meta, mode = -1, None
        if meta_name is not None:
            meta = META_COMMAND_TO_INDEX[meta_name]
            mode = classify_driving_mode(traj["waypoints"], meta_name, float(speed), float(current.get("speed_limit", 0.0)))
        s = ds._build_sample(i, prompt=prompt, image=ds.load_image(i, shifted_camera=False), waypoints=traj["waypoints"],
                             path=traj["path"], target_points=np.array([tp, ntp]), speed=speed, measurement_path=mpath,
                             meta_command=meta, mode=mode)
        return s, int(current.get("command", 0)), traj["path"]

    @torch.no_grad()
    def run(samples):
        batch = dm.collate(samples)
        batch = type(batch)(*[x.to(device) if torch.is_tensor(x) else x for x in batch])
        model.set_conditioning(batch.target_points, batch.wm_tubelet, batch.wm_actions, batch.wm_state)
        try:
            logits, hidden = model.forward_vlm(batch.input_ids, batch.attention_mask, batch.pixel_values)
        finally:
            model.clear_conditioning()
        lab = batch.labels[:, 1:]
        lg = logits[:, :-1, :].float()
        ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), lab.reshape(-1).clamp(min=0), reduction="none").view(lab.shape)
        is_act = (lab >= model.action_start_id) & (lab < model.action_end_id)
        act = ((ce * is_act).sum(1) / is_act.sum(1).clamp(min=1)).cpu().numpy()
        pred, rows = model.predict_path(batch.input_ids, hidden.to(model.path_head.mlp[0].weight.dtype))
        pred = pred.float()
        gt = batch.path.to(pred)[rows]
        pl = ((pred - gt) ** 2).mean(dim=(1, 2)).cpu().numpy()
        out = [dict(action_loss=float("nan"), path_loss=float("nan"), heading=float("nan")) for _ in samples]
        for j, r in enumerate(rows.tolist()):
            out[r] = dict(action_loss=float(act[r]), path_loss=float(pl[j]), heading=heading15(pred[j].cpu().numpy()))
        return out

    records, n_done = [], 0
    for (split, g, b), lst in sorted(cand.items(), key=lambda kv: (kv[0][0], GROUPS.index(kv[0][1]), kv[0][2])):
        rng.shuffle(lst)
        per = a.per_bin if g in ("hardS", "easyS") else a.per_bin_turn
        seen_ap = collections.Counter()
        for ds, i, dist, rec in lst:
            if sum(seen_ap.values()) >= per:
                break
            apk = (rec["route"], rec["ref_frame"])
            if seen_ap[apk] >= 1:                 # one frame per approach and bin
                continue
            man = rec["man"]
            try:
                _, current, _ = ds.load_measurements(i)
                planner = int(current.get("command", 0))
                sentence = sentences[MAN_SENTENCE[man]][0]
                samples, conds = [], []
                for cond, anchor in (("test", MAN_COMMAND[man]), ("orig", planner)):
                    s, _, gt_path = build(ds, i, sentence, anchor)
                    samples.append(s); conds.append(cond)
                res = run(samples)
            except Exception as e:                 # noqa: BLE001
                print(f"[eval] skip {ds.measurements[i]} {ds.sample_start[i]}: {e}", flush=True)
                continue
            seen_ap[apk] += 1
            r = dict(split=split, group=g, bin=list(BINS[b]), dist=round(dist, 1), man=man, reach=rec["reach"],
                     pos=rec["pos"], town=rec["town"], route=rec["route"], frame=int(ds.sample_start[i]), planner=planner,
                     gt_heading=round(heading15(gt_path), 1))
            for c, x in zip(conds, res):
                r[c] = {k: round(v, 4) for k, v in x.items()}
            records.append(r)
            n_done += 1
            if n_done % 50 == 0:
                print(f"[eval] {n_done} frames ({time.time() - t0:.0f} s)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(a.out, "w"))
    text = summary(records)
    print(text, flush=True)
    Path(a.out).with_suffix(".txt").write_text(text + "\n")
    print(f"[eval] done: {len(records)} frames in {time.time() - t0:.0f} s -> {a.out}", flush=True)


def summary(records, dev=15.0):
    """Per split / group / bin: expert vs predicted heading; share of frames whose predicted heading is
    more than `dev` deg off the expert's ("wrong"); for hardS also the share that turns toward a
    reachable turn side."""
    L = []
    for cond in ("test", "orig"):
        L.append(f"\n=== anchor = {cond} ({'the manoeuvre anchor, as in the competition' if cond == 'test' else 'planner command of the frame, as in training'})")
        L.append("split group  bin      n  gt_head  pred_head  |err|  wrong(>%d)  turned  path_loss  act_loss" % dev)
        for split in ("train", "val"):
            for g in GROUPS:
                for lo, hi in BINS + [(0, 30)]:
                    rs = [r for r in records if r["split"] == split and r["group"] == g and
                          (r["bin"] == [lo, hi] if (lo, hi) != (0, 30) else True) and cond in r]
                    if not rs:
                        continue
                    gt = np.array([r["gt_heading"] for r in rs]); pr = np.array([r[cond]["heading"] for r in rs])
                    err = np.abs(pr - gt)
                    turned = np.mean([(("L" in r["reach"] and r[cond]["heading"] < -dev) or ("R" in r["reach"] and r[cond]["heading"] > dev))
                                      for r in rs]) if g == "hardS" else float("nan")
                    L.append(f"{split:5s} {g:6s} {lo:2d}-{hi:<2d} {len(rs):5d}  {np.nanmean(gt):+6.1f}   {np.nanmean(pr):+6.1f}   "
                             f"{np.nanmean(err):5.1f}   {np.nanmean(err > dev):6.2f}    {turned:5.2f}   "
                             f"{np.nanmean([r[cond]['path_loss'] for r in rs]):7.4f}  {np.nanmean([r[cond]['action_loss'] for r in rs]):6.3f}")
                L.append("")
        hs = [r for r in records if r["group"] == "hardS" and cond in r]
        if hs:
            L.append("hardS by lane position (all bins, both splits): position  n  wrong  turned")
            for p in ("leftmost", "middle", "rightmost", "single"):
                rs = [r for r in hs if r["pos"] == p]
                if rs:
                    err = np.array([abs(r[cond]["heading"] - r["gt_heading"]) for r in rs])
                    tr = np.mean([("L" in r["reach"] and r[cond]["heading"] < -dev) or ("R" in r["reach"] and r[cond]["heading"] > dev) for r in rs])
                    L.append(f"  {p:9s} {len(rs):4d}  {np.mean(err > dev):.2f}  {tr:.2f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="part", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--root", default="database/simlingo"); s.add_argument("--out", required=True)
    s.add_argument("--routes", type=int, default=0, help="0 = all routes of the index")
    s.add_argument("--ref-m", type=float, default=12.0, help="lane features at the frame closest to this distance before entry")
    e = sub.add_parser("eval")
    e.add_argument("--approaches", required=True); e.add_argument("--ckpt", required=True); e.add_argument("--out", required=True)
    e.add_argument("--root", default="database/simlingo")
    e.add_argument("--per-bin", type=int, default=40, help="frames per (split, straight group, bin)")
    e.add_argument("--per-bin-turn", type=int, default=15, help="frames per (split, turn group, bin)")
    e.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    scan(a) if a.part == "scan" else evaluate_main(a)


if __name__ == "__main__":
    main()
