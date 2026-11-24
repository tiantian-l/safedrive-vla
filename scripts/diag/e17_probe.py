"""E17 ([26]) data check before training (11/24). DIAGNOSTIC; runs on the collected E17 pairs (training towns only).

Part "vla" (GPU): the released model on the collected frames 0-30 m before the junction entry. Within a paired
group the frames before the junction are the same image (trial 1: mean pixel difference 0.4-0.7 / 255 up to
~2 m before the entry), so every frame is run with each manoeuvre m of its group:
  m      sentence of m (relabel_sentences.json, first one) + world-model anchor of m   (what E17 trains)
  txt_m  sentence of m + anchor of the frame's planner command                       (text only)
and with its own manoeuvre and the planner-command anchor ("orig", what SimLingo training saw).
The expert target for m at that frame = the path of the group's route m at the same frame index (the routes
of a group drive identically until they split). Per bin: expert vs predicted path heading 15 m ahead,
share > 15 deg off ("wrong"). Reading: if the model already follows m everywhere (like E16 on SimLingo),
E17 has little to teach; if it goes the image's way regardless of m (as on the CARLA-F failures), it has.

Part "wm" (GPU): the world model's own validation loss (world_model/train.py: teacher-forced + autoregressive
L1 on normalized V-JEPA 2 latents, expert actions) on the E17 routes vs a sample of SimLingo validation routes.
Similar loss = the new frames are in the world model's distribution (no retraining needed).

    $PY scripts/diag/e17_probe.py vla --root database/e17_try2 --ckpt ckpts/safedrive-vla/safedrive_vla/model.ckpt \
        --out work_dirs/e17_probe/try2_vla.json
    $PY scripts/diag/e17_probe.py wm --root database/e17_try2 --out work_dirs/e17_probe/try2_wm.json
"""
import argparse, collections, glob, gzip, json, math, os, random, re, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "diag"))

BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
MAN_COMMAND = {"L": 1, "R": 2, "S": 3}
MAN_SENTENCE = {"L": "left", "S": "straight", "R": "right"}
ROUTE_RE = re.compile(r"_Rep\d+_(?P<route>.+)_route\d+_[\d_]+$")


def heading15(path):
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or len(p) == 0:
        return float("nan")
    x, y = p[min(15, len(p)) - 1, :2]
    return math.degrees(math.atan2(y, x))


def route_info(route_dir):
    """route name, group, manoeuvre, kind; per frame driven metres to the first junction frame (negative after)."""
    m = ROUTE_RE.search(os.path.basename(route_dir))
    name = m.group("route")
    ms = []
    for f in sorted(glob.glob(f"{route_dir}/measurements/*.json.gz")):
        with gzip.open(f, "rt") as fh:
            ms.append(json.load(fh))
    cum = [0.0]
    for a, b in zip(ms, ms[1:]):
        cum.append(cum[-1] + math.dist(a["pos_global"][:2], b["pos_global"][:2]))
    j = next((i for i, x in enumerate(ms) if x["junction"]), None)
    dist = [None if j is None else cum[j] - c for c in cum]
    return dict(route=name, group=name.rsplit("_", 1)[0], man=name[-1], kind=Path(route_dir).parent.name,
                meas=ms, dist=dist, entry=j)


def bin_of(d):
    if d is None:
        return None
    for k, (lo, hi) in enumerate(BINS):
        if lo <= d < hi:
            return k
    return None


def vla(a):
    import torch
    from omegaconf import OmegaConf
    from probe import load
    from safedrive_vla.constants import COMMAND_TO_META_COMMAND, META_COMMAND_TO_INDEX, TASK_PROMPT
    from safedrive_vla.data.datamodule import LMDRIVE_TEMPLATES, SafeDriveDataModule
    from safedrive_vla.data.dataset import DrivingDataset
    from safedrive_vla.data.driving_mode import classify_driving_mode

    t0 = time.time()
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

    infos, groups = {}, collections.defaultdict(dict)   # route dir -> info; group -> {man: info}
    items = []                                           # (ds, i, info, frame)
    for split in ("train", "val"):                       # the hash split of BaseDataset; together = every clean route
        ds = DrivingDataset(split=split, bucket_name="all", **kw)
        ds.tfs = lambda image: image
        for i in range(len(ds)):
            rd = os.path.dirname(str(ds.measurements[i], encoding="utf-8"))
            if rd not in infos:
                infos[rd] = route_info(rd)
                groups[infos[rd]["group"]][infos[rd]["man"]] = infos[rd]
            items.append((ds, i, infos[rd], int(ds.sample_start[i])))
    print(f"[vla] {len(infos)} routes, {len(groups)} groups, {len(items)} frames ({time.time() - t0:.0f} s)", flush=True)

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
        return ds._build_sample(i, prompt=prompt, image=ds.load_image(i, shifted_camera=False), waypoints=traj["waypoints"],
                                path=traj["path"], target_points=np.array([tp, ntp]), speed=speed, measurement_path=mpath,
                                meta_command=meta, mode=mode)

    @torch.no_grad()
    def run(samples):
        batch = dm.collate(samples)
        batch = type(batch)(*[x.to(device) if torch.is_tensor(x) else x for x in batch])
        model.set_conditioning(batch.target_points, batch.wm_tubelet, batch.wm_actions, batch.wm_state)
        try:
            _, hidden = model.forward_vlm(batch.input_ids, batch.attention_mask, batch.pixel_values)
        finally:
            model.clear_conditioning()
        pred, rows = model.predict_path(batch.input_ids, hidden.to(model.path_head.mlp[0].weight.dtype))
        out = [float("nan")] * len(samples)
        for j, r in enumerate(rows.tolist()):
            out[r] = heading15(pred[j].float().cpu().numpy())
        return out

    from safedrive_vla.data.simlingo_base import BaseDataset
    records = []
    for ds, i, info, fr in items:
        d = info["dist"][fr] if fr < len(info["dist"]) else None
        b = bin_of(d)
        if b is None:
            continue
        planner = int(info["meas"][fr]["command"])
        mans = sorted(groups[info["group"]])
        conds = []
        for m in mans:
            conds.append((m, sentences[MAN_SENTENCE[m]][0], MAN_COMMAND[m]))
            conds.append(("txt_" + m, sentences[MAN_SENTENCE[m]][0], planner))
        conds.append(("orig", sentences[MAN_SENTENCE[info["man"]]][0], planner))
        try:
            res = run([build(ds, i, s, anc) for _, s, anc in conds])
        except Exception as e:                       # noqa: BLE001
            print(f"[vla] skip {info['route']} {fr}: {e}", flush=True)
            continue
        gt = {}
        for m in mans:
            other = groups[info["group"]][m]
            if fr < len(other["meas"]):
                gt[m] = heading15(BaseDataset.equal_spacing_route(np.array(other["meas"][fr]["route"])))
        r = dict(route=info["route"], group=info["group"], kind=info["kind"], man=info["man"], frame=fr, dist=round(d, 1),
                 bin=list(BINS[b]), planner=planner, gt={k: round(v, 1) for k, v in gt.items()},
                 pred={c: round(v, 1) for (c, _, _), v in zip(conds, res)})
        records.append(r)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(out, "w"))
    text = vla_summary(records)
    print(text, flush=True)
    out.with_suffix(".txt").write_text(text + "\n")
    print(f"[vla] done: {len(records)} frames in {time.time() - t0:.0f} s -> {out}", flush=True)


def vla_summary(records, dev=15.0):
    L = ["Per manoeuvre m and bin (metres before the junction entry): expert heading of route m, predicted heading,",
         f"share > {dev:.0f} deg off. Frames of every route of a group are run with every manoeuvre of the group;",
         "S rows are the question (does the model go straight when told, on a lane that can also turn).", ""]
    for cond, title in (("", "sentence + anchor of m (E17 training input)"), ("txt_", "sentence of m, planner-command anchor (text only)")):
        L.append(f"=== {title}")
        L.append("  m  bin      n   expert   pred   |err|  wrong")
        for m in "SLR":
            for lo, hi in BINS + [(0, 30)]:
                rs = [r for r in records if m in r["gt"] and cond + m in r["pred"] and ((lo, hi) == (0, 30) or r["bin"] == [lo, hi])]
                if not rs:
                    continue
                g = np.array([r["gt"][m] for r in rs]); p = np.array([r["pred"][cond + m] for r in rs])
                e = np.abs(p - g)
                L.append(f"  {m}  {lo:2d}-{hi:<2d} {len(rs):5d}  {np.nanmean(g):+6.1f}  {np.nanmean(p):+6.1f}  {np.nanmean(e):5.1f}  {np.nanmean(e > dev):5.2f}")
            L.append("")
    rs = [r for r in records if "orig" in r["pred"] and r["man"] in r["gt"]]
    if rs:
        e = np.array([abs(r["pred"]["orig"] - r["gt"][r["man"]]) for r in rs])
        L.append(f"=== own sentence, planner-command anchor (SimLingo training input): n {len(rs)}, |err| {np.nanmean(e):.1f}, wrong {np.nanmean(e > dev):.2f}")
    sp = [r for r in records if {"L", "S"} <= set(r["pred"]) or {"R", "S"} <= set(r["pred"])]
    if sp:
        L.append("\n=== separation within a frame (pred heading turn - straight), by bin: mean (expert)")
        for lo, hi in BINS:
            for t in "LR":
                xs = [(r["pred"][t] - r["pred"]["S"], r["gt"][t] - r["gt"]["S"]) for r in sp
                      if r["bin"] == [lo, hi] and t in r["pred"] and "S" in r["pred"] and t in r["gt"] and "S" in r["gt"]]
                if xs:
                    x = np.array(xs)
                    L.append(f"  {lo:2d}-{hi:<2d} {t}-S  n {len(x):4d}  pred {np.nanmean(x[:, 0]):+6.1f}  expert {np.nanmean(x[:, 1]):+6.1f}")
    return "\n".join(L)


def wm(a):
    import torch, yaml
    from torch.utils.data import DataLoader
    from world_model.data.carla_clips import CarlaClipDataset, collate, find_routes
    from world_model.models.world_model import WorldModel, WorldModelConfig

    t0 = time.time()
    cfg = yaml.safe_load(open(REPO / a.config))
    cfg["model"]["encoder_ckpt"] = str(REPO / cfg["model"]["encoder_ckpt"])
    device = torch.device("cuda")
    model = WorldModel(WorldModelConfig(**cfg["model"])).to(device)
    state = torch.load(REPO / a.predictor, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])
    model.eval()
    d = cfg["data"]

    def make(routes):
        return CarlaClipDataset(routes, num_frames=d["num_frames"], tubelet_size=d["tubelet_size"],
                                img_hw=(d["img_h"], d["img_w"]), stride=d["stride"])

    sets = {}
    e17 = find_routes(str(REPO / a.root / "data"))
    for kind in ("paired", "unpaired"):
        rs = [r for r in e17 if f"/{kind}/" in r]
        if rs:
            sets[f"e17_{kind}"] = rs
    val = []
    for split in d["val_splits"]:
        val += find_routes(os.path.join(str(REPO / d["root"]), split))
    rng = random.Random(0); rng.shuffle(val)
    sets["simlingo_val"] = sorted(val[: a.val_routes])
    tr = []
    for split in d["splits"][:1]:
        tr += find_routes(os.path.join(str(REPO / d["root"]), split))
    tr = [r for r in tr if "Town12" in os.path.basename(r)]
    rng.shuffle(tr)
    sets["simlingo_train_town12"] = sorted(tr[: a.val_routes])
    res = {}
    for name, routes in sets.items():
        loader = DataLoader(make(routes), batch_size=8, shuffle=False, num_workers=8, collate_fn=collate)
        sums, n = collections.defaultdict(float), 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(batch["video"], batch["actions"], batch["states"], auto_steps=5)
                for k in ("loss", "loss_tf", "loss_ar", "std_ratio_tf", "std_ratio_ar"):
                    sums[k] += float(out[k].item())
                n += 1
        res[name] = dict(routes=len(routes), batches=n, **{k: round(v / max(1, n), 5) for k, v in sums.items()})
        print(f"[wm] {name}: {res[name]} ({time.time() - t0:.0f} s)", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="part", required=True)
    v = sub.add_parser("vla")
    v.add_argument("--root", required=True, help="collect_pairs --out (has data/simlingo/e17/...)")
    v.add_argument("--ckpt", required=True); v.add_argument("--out", required=True)
    w = sub.add_parser("wm")
    w.add_argument("--root", required=True); w.add_argument("--out", required=True)
    w.add_argument("--config", default="world_model/configs/world_model.yaml")
    w.add_argument("--predictor", default="ckpts/safedrive-vla/world_model/latest.pt")
    w.add_argument("--val-routes", type=int, default=40)
    a = ap.parse_args()
    a.root = Path(a.root)
    vla(a) if a.part == "vla" else wm(a)


if __name__ == "__main__":
    main()
