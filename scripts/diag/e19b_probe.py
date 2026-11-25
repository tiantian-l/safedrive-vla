"""E19b probe (11/25): does the sentence decide the manoeuvre on held-out E17 junctions? DIAGNOSTIC (training towns only).

Frames: E17 routes of the held-out splits of scripts/data/build_e17_index.py (Town12 junctions never trained on +
every old-town route), 0-15 m before the junction entry (bins 0-5 / 5-10 / 10-15). The routes of a group share
the picture before the junction, so each frame is run with every manoeuvre m of its group under five inputs:
  m       sentence of m + world-model anchor of m                          (E17 training input; must stay right)
  txt_m   sentence of m + anchor of the frame's planner command            (T4: text only, dream present)
  mask_m  sentence of m + world block masked out (E19b 'mask')            (T3 for mask)
  fol_m   sentence of m + follow-the-road anchor (E19b 'follow')           (T3 for follow; = the CARLA-F straight case)
  anc_m   neutral sentence "follow the road" + anchor of m                 (trust in the dream; must not collapse)
Path heading 15 m ahead vs the expert of m (the group's route m at the same frame index). Summary per input:
separation turn - straight (pred vs expert) per bin, and share > 15 deg off the expert. The mode token is teacher-
forced with the expert's (as in training), so the path head sees the same prefix as in e17_probe.py.

Gates (claude/ideas_2026-11-25_additions.md): released model only-text separation 0-5 m ~0.2 deg; target > 20 deg
for mask_ (mask pilot) / fol_ (follow pilot) and larger txt_; anc_ separation must not collapse (released ~38 deg at
0-5 m); m correctness must stay.

    $PY scripts/diag/e19b_probe.py --root database/e17_pairs --ckpt ckpts/safedrive-vla/safedrive_vla/model.ckpt \
        --out work_dirs/e19b_probe/released.json
    $PY scripts/diag/e19b_probe.py --root database/e17_pairs --ckpt ckpts/ft/E19b_mask_s300/model.ckpt \
        --out work_dirs/e19b_probe/mask_s300.json
"""
import argparse, collections, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "diag"))
from e17_probe import MAN_COMMAND, MAN_SENTENCE, heading15, route_info   # noqa: E402

BINS = [(0, 5), (5, 10), (10, 15)]
PREFIXES = (("", "sentence + anchor of m (training input)"), ("txt_", "T4: sentence of m, planner anchor"),
            ("mask_", "T3 mask: sentence of m, world block masked"), ("fol_", "T3 follow: sentence of m, follow-the-road anchor"),
            ("anc_", "trust: 'follow the road', anchor of m"))
FOLLOW_CMD, NEUTRAL = 4, "follow the road"


def bin_of(d):
    if d is None:
        return None
    for k, (lo, hi) in enumerate(BINS):
        if lo <= d < hi:
            return k
    return None


def run_probe(a):
    import torch
    from omegaconf import OmegaConf
    from probe import load
    from safedrive_vla.constants import COMMAND_TO_META_COMMAND, META_COMMAND_TO_INDEX, TASK_PROMPT
    from safedrive_vla.data.datamodule import LMDRIVE_TEMPLATES, SafeDriveDataModule
    from safedrive_vla.data.dataset import DrivingDataset
    from safedrive_vla.data.driving_mode import classify_driving_mode
    from safedrive_vla.data.simlingo_base import BaseDataset

    t0 = time.time()
    device = torch.device("cuda")
    model, processor, cfg = load(a.ckpt, device)
    root = os.path.abspath(REPO / a.root)
    idx = json.load(open(a.index or os.path.join(root, "e17_index.json")))
    keep = {os.path.join(root, rel) for rel, sp in idx["split"].items() if sp in a.splits}
    wm = OmegaConf.to_container(cfg.model.world_model, resolve=True) if cfg.model.get("world_model") is not None else None
    kw = dict(data_root=root, data_scale=1.0, lmdrive_templates=LMDRIVE_TEMPLATES,
              codebook_path=cfg.model.action_token.codebook_path, use_mode_token=cfg.model.use_mode_token, world_model=wm)
    sentences = json.load(open(REPO / "safedrive_vla" / "assets" / "relabel_sentences.json"))
    dm = SafeDriveDataModule.__new__(SafeDriveDataModule)
    dm.processor, dm.tokenizer = processor, processor.tokenizer
    dm.action_ids = (int(model.action_start_id), int(model.action_end_id))
    dm.wm_enabled, dm.world_model = wm is not None, dict(wm or {})

    infos, groups, items = {}, collections.defaultdict(dict), []
    for split in ("train", "val"):                       # BaseDataset hash split; together = every clean route
        ds = DrivingDataset(split=split, bucket_name="all", **kw)
        ds.tfs = lambda image: image
        for i in range(len(ds)):
            rd = os.path.dirname(str(ds.measurements[i], encoding="utf-8"))
            if rd not in keep:
                continue
            if rd not in infos:
                infos[rd] = route_info(rd)
                groups[infos[rd]["group"]][infos[rd]["man"]] = infos[rd]
            items.append((ds, i, infos[rd], int(ds.sample_start[i])))
    print(f"[e19b] {len(infos)} held-out routes ({','.join(a.splits)}), {len(groups)} groups, {len(items)} frames "
          f"({time.time() - t0:.0f} s)", flush=True)

    def build(ds, i, sentence, anchor_cmd, dropped=False):
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
        return s._replace(wm_dropped=True) if dropped else s

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
            s = sentences[MAN_SENTENCE[m]][0]
            conds += [(m, s, MAN_COMMAND[m], False), ("txt_" + m, s, planner, False), ("mask_" + m, s, MAN_COMMAND[m], True),
                      ("fol_" + m, s, FOLLOW_CMD, False), ("anc_" + m, NEUTRAL, MAN_COMMAND[m], False)]
        try:
            res = run([build(ds, i, s, anc, drop) for _, s, anc, drop in conds])
        except Exception as e:                       # noqa: BLE001
            print(f"[e19b] skip {info['route']} {fr}: {e}", flush=True)
            continue
        gt = {}
        for m in mans:
            other = groups[info["group"]][m]
            if fr < len(other["meas"]):
                gt[m] = heading15(BaseDataset.equal_spacing_route(np.array(other["meas"][fr]["route"])))
        records.append(dict(route=info["route"], group=info["group"], split=idx["split"].get(os.path.relpath(
            os.path.dirname(str(ds.measurements[i], encoding="utf-8")), root)), man=info["man"], frame=fr, dist=round(d, 1),
            bin=list(BINS[b]), planner=planner, gt={k: round(v, 1) for k, v in gt.items()},
            pred={c: round(v, 1) for (c, _, _, _), v in zip(conds, res)}))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(out, "w"))
    text = summary(records, a.ckpt)
    print(text, flush=True)
    out.with_suffix(".txt").write_text(text + "\n")
    print(f"[e19b] done: {len(records)} frames in {time.time() - t0:.0f} s -> {out}", flush=True)


def summary(records, ckpt="", dev=15.0):
    L = [f"E19b probe {ckpt}: {len(records)} held-out frames, path heading 15 m ahead (deg, + = right).", ""]
    for split in (None, "holdout", "holdout_town"):
        rs_all = [r for r in records if split is None or r.get("split") == split]
        if not rs_all:
            continue
        L.append(f"##### {'all held-out' if split is None else split}: {len(rs_all)} frames")
        L.append("separation turn - straight within a frame, mean (median |.|); expert in brackets")
        L.append(f"  {'input':48s} " + "  ".join(f"{lo:2d}-{hi:<2d} L-S          R-S         " for lo, hi in BINS))
        for pre, title in PREFIXES:
            cells = []
            for lo, hi in BINS:
                for t in "LR":
                    xs = [(r["pred"][pre + t] - r["pred"][pre + "S"], r["gt"][t] - r["gt"]["S"]) for r in rs_all
                          if r["bin"] == [lo, hi] and pre + t in r["pred"] and pre + "S" in r["pred"] and t in r["gt"] and "S" in r["gt"]]
                    if xs:
                        x = np.array(xs, dtype=float)
                        cells.append(f"{np.nanmean(x[:, 0]):+5.1f}({np.nanmedian(np.abs(x[:, 0])):4.1f})[{np.nanmean(x[:, 1]):+5.1f}]")
                    else:
                        cells.append(f"{'-':20s}")
            L.append(f"  {title:48s} " + " ".join(f"{c:20s}" for c in cells))
        L.append(f"share > {dev:.0f} deg off the expert of m (n), 0-15 m")
        for pre, title in PREFIXES[:4]:
            row = []
            for m in "SLR":
                e = [abs(r["pred"][pre + m] - r["gt"][m]) for r in rs_all if pre + m in r["pred"] and m in r["gt"]]
                row.append(f"{m} {np.nanmean(np.array(e) > dev):4.2f} ({len(e)})" if e else f"{m}   - ")
            L.append(f"  {title:48s} " + "   ".join(row))
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="database/e17_pairs"); ap.add_argument("--index")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="holdout,holdout_town")
    a = ap.parse_args()
    a.splits = tuple(a.splits.split(","))
    run_probe(a)


if __name__ == "__main__":
    main()
