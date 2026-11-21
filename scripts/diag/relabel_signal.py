"""[20] diagnosis (11/21): is there a text signal in the junction frames? DIAGNOSTIC, not part of the agent.

For training-data frames approaching / inside a junction (relabel index, <= 30 m before entry), feed the
released model the same frame with the navigation sentence of each manoeuvre (left / straight / right,
competition-style wording from assets/relabel_sentences.json) and compare, against the expert's labels:
  - action_loss (CE of the expert's action tokens) and path_loss (MSE of the expert's path),
  - the path head's heading 15 m ahead,
with the world-model anchor either kept at the planner command of the frame ("orig", what training saw)
or switched to the sentence's manoeuvre ("match", what the agent does at test time).

If the loss with the expert's own manoeuvre is clearly lower than with the other manoeuvres, the labels
carry a text signal and a stronger fine-tune could teach the model to use it; if not, the image already
predicts the labels and relabelling the prompt cannot force the model to read the instruction.

    OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /path/to/python scripts/diag/relabel_signal.py --ckpt ckpts/safedrive-vla/safedrive_vla/model.ckpt \
        --out work_dirs/E14_diag/relabel_signal.json [--split val] [--per-bin 25]
"""
import argparse, json, math, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "diag"))

BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
MANS = ("L", "S", "R")
MAN_COMMAND = {"L": 1, "R": 2, "S": 3}          # route-planner command ids (relabel.MANOEUVRE_TO_COMMAND)
MAN_SENTENCE = {"L": "left", "S": "straight", "R": "right"}


def heading15(path):
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or len(p) == 0:
        return float("nan")
    x, y = p[min(15, len(p)) - 1, :2]
    return math.degrees(math.atan2(y, x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val", choices=("val", "train"))
    ap.add_argument("--per-bin", type=int, default=25, help="frames per (manoeuvre, distance bin)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from omegaconf import OmegaConf
    from probe import load                                   # scripts/diag/probe.py
    from safedrive_vla.constants import COMMAND_TO_META_COMMAND, META_COMMAND_TO_INDEX, TASK_PROMPT
    from safedrive_vla.data.datamodule import LMDRIVE_TEMPLATES, SafeDriveDataModule
    from safedrive_vla.data.dataset import DrivingDataset
    from safedrive_vla.data.driving_mode import classify_driving_mode
    from safedrive_vla.data.relabel import Relabeler

    t0 = time.time()
    device = torch.device("cuda")
    model, processor, cfg = load(a.ckpt, device)
    data_root = str(REPO / "database" / "simlingo")
    wm = OmegaConf.to_container(cfg.model.world_model, resolve=True) if cfg.model.get("world_model") is not None else None
    kw = dict(data_root=data_root, data_scale=1.0, lmdrive_templates=LMDRIVE_TEMPLATES,
              codebook_path=cfg.model.action_token.codebook_path, use_mode_token=cfg.model.use_mode_token, world_model=wm)
    relab = Relabeler(index=f"{data_root}/relabel_index.json", sentences=str(REPO / "safedrive_vla/assets/relabel_sentences.json"),
                      data_root=data_root)
    sentences = relab.lib

    # collate as in training (only the attributes SafeDriveDataModule.collate uses)
    dm = SafeDriveDataModule.__new__(SafeDriveDataModule)
    dm.processor, dm.tokenizer = processor, processor.tokenizer
    dm.action_ids = (int(model.action_start_id), int(model.action_end_id))
    dm.wm_enabled, dm.world_model = wm is not None, dict(wm or {})

    # candidate frames from the relabel buckets of the chosen split
    rng = random.Random(a.seed)
    cand = defaultdict(list)    # (man, bin) -> [(dataset, index, dist)]
    datasets = []
    for bucket in ("relabel_straight", "relabel_turn"):
        ds = DrivingDataset(split=a.split, bucket_name=bucket, **kw)
        ds.tfs = lambda image: image                         # no image augmentation
        datasets.append(ds)
        for i in range(len(ds)):
            mpath = str(ds.measurements[i], encoding="utf-8")
            entry = relab.routes.get(relab.route_key(mpath + "/x"), {}).get(str(int(ds.sample_start[i])))
            if entry is None:
                continue
            man, dist = entry[0], float(entry[1])
            if man not in MANS or dist > 30:
                continue
            b = next((k for k, (lo, hi) in enumerate(BINS) if lo <= dist < hi or (hi == 30 and dist == 30)), None)
            if b is not None:
                cand[(man, b)].append((ds, i, dist))
    print(f"[diag] candidates per (manoeuvre, bin): { {f'{m}{BINS[b]}': len(v) for (m, b), v in sorted(cand.items())} }"
          f" ({time.time() - t0:.0f} s)", flush=True)

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
    def evaluate(samples):
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

    records = []
    n_done = 0
    for (man, b), lst in sorted(cand.items()):
        rng.shuffle(lst)
        for ds, i, dist in lst[: a.per_bin]:
            try:
                samples, meta = [], []
                _, current, _ = ds.load_measurements(i)
                planner = int(current.get("command", 0))
                for anchor_mode in ("orig", "match"):
                    for m in MANS:
                        sentence = sentences[MAN_SENTENCE[m]][0]
                        anchor = planner if anchor_mode == "orig" else MAN_COMMAND[m]
                        s, _, gt_path = build(ds, i, sentence, anchor)
                        samples.append(s); meta.append((anchor_mode, m))
                res = evaluate(samples)
            except Exception as e:            # noqa: BLE001 - skip a broken frame, keep the run going
                print(f"[diag] skip {ds.measurements[i]} {ds.sample_start[i]}: {e}", flush=True)
                continue
            rec = dict(man=man, bin=list(BINS[b]), dist=round(dist, 1), planner=planner,
                       gt_heading=round(heading15(gt_path), 1), frame=int(ds.sample_start[i]),
                       route=relab.route_key(str(ds.measurements[i], encoding="utf-8") + "/x"))
            for (am, m), r in zip(meta, res):
                rec[f"{am}_{m}"] = {k: round(v, 4) for k, v in r.items()}
            records.append(rec)
            n_done += 1
            if n_done % 25 == 0:
                print(f"[diag] {n_done} frames ({time.time() - t0:.0f} s)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(a.out, "w"))

    # summary: per (manoeuvre, bin), loss with the expert's manoeuvre vs the other two
    lines = []
    for am in ("orig", "match"):
        lines.append(f"\n=== anchor = {am} ({'planner command of the frame' if am == 'orig' else 'the sentence manoeuvre'})")
        lines.append("man  bin       n  gt_head  action: own  other  frac(own<other) | path: own  other  frac | pred heading L / S / R")
        for man in MANS:
            for b, (lo, hi) in enumerate(BINS):
                rs = [r for r in records if r["man"] == man and r["bin"] == [lo, hi]]
                if not rs:
                    continue
                own_a = np.array([r[f"{am}_{man}"]["action_loss"] for r in rs])
                oth_a = np.array([np.mean([r[f"{am}_{m}"]["action_loss"] for m in MANS if m != man]) for r in rs])
                own_p = np.array([r[f"{am}_{man}"]["path_loss"] for r in rs])
                oth_p = np.array([np.mean([r[f"{am}_{m}"]["path_loss"] for m in MANS if m != man]) for r in rs])
                hd = [np.nanmean([r[f"{am}_{m}"]["heading"] for r in rs]) for m in MANS]
                lines.append(f"{man}  {lo:2d}-{hi:<2d}  {len(rs):4d}  {np.mean([r['gt_heading'] for r in rs]):+6.1f}  "
                             f"{np.nanmean(own_a):6.3f} {np.nanmean(oth_a):6.3f}  {np.nanmean(own_a < oth_a):5.2f}      | "
                             f"{np.nanmean(own_p):6.3f} {np.nanmean(oth_p):6.3f} {np.nanmean(own_p < oth_p):5.2f} | "
                             f"{hd[0]:+6.1f} {hd[1]:+6.1f} {hd[2]:+6.1f}")
    text = "\n".join(lines)
    print(text, flush=True)
    Path(a.out).with_suffix(".txt").write_text(text + "\n")
    print(f"[diag] done: {len(records)} frames in {time.time() - t0:.0f} s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
