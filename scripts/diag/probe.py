"""E10 offline probe: on frames dumped by an eval run with --dump (E9), change ONE input at a time
and record where the model's path head points. DIAGNOSTIC, not part of the agent.

Frames: per route, straight / lane-follow instruction window (from the route XML). If the path
heading (15 m, ego frame) leaves +-30 deg there, the three dumps before / at that first wrong tick;
otherwise three dumps 2.0 / 1.5 / 1.0 s before the instruction ends (approaching the junction).
Variants (each changes one thing relative to the dumped inputs):
  base; dir_left / dir_straight / dir_right (training-format command, no distance);
  wording (training format with distance, follow the road, paraphrases, LMDrive style);
  speed in the prompt 3 / 6 / 10 m/s; world-model anchor = each of the 6 meta-commands;
  VLA image shifted 40 px left / right; road markings erased in the ego-lane trapezoid / whole lower image.

    $PY scripts/diag/probe.py work_dirs/E9_dump/carla_f --routes-dir benchmark/dev_routes/probe11/carla_f \
        --ckpt ckpts/safedrive-vla/safedrive_vla/model.ckpt --out work_dirs/E9_dump/probe
"""
import argparse, glob, json, math, re, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def path_heading(path, dist=15.0):
    pts = [(0.0, 0.0)] + [tuple(p[:2]) for p in path]; acc = 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        s = math.hypot(bx - ax, by - ay)
        if acc + s >= dist: return math.degrees(math.atan2(by - ay, bx - ax))
        acc += s
    return math.degrees(math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0]))


def norm(t): return (t or "").strip().rstrip(". ").strip().lower()


# ---------------------------------------------------------------- variants
CMD_RE = re.compile(r"Command: .*?\. Predict the actions\.")
SPD_RE = re.compile(r"Current speed: -?[\d.]+ m/s\.")


def with_command(prompt, cmd):
    new, n = CMD_RE.subn(f"Command: {cmd}. Predict the actions.", prompt)
    if n != 1: raise ValueError(f"no command in prompt: {prompt[:120]}")
    return new


def shift(img, dx):
    m = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def erase_marks(img, lane_only):
    """Paint bright road markings (white / yellow paint) with the local road colour, below the horizon."""
    out = img.copy(); h, w = img.shape[:2]
    y0 = int(h * 0.45)
    mask = np.zeros((h, w), np.uint8)
    if lane_only:  # trapezoid of the ego lane ahead
        poly = np.array([[int(w * 0.30), h - 1], [int(w * 0.70), h - 1], [int(w * 0.56), y0], [int(w * 0.44), y0]])
        cv2.fillPoly(mask, [poly], 1)
    else:
        mask[y0:] = 1
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.int16)
    for ys in range(y0, h, 12):  # row bands: local median road colour
        band = slice(ys, min(h, ys + 12)); m = mask[band] > 0
        if not m.any(): continue
        g = gray[band][m]; med = np.median(g)
        bright = (gray[band] > med + 35) & m
        col = np.median(img[band][m & (np.abs(gray[band] - med) < 15)], axis=0) if (m & (np.abs(gray[band] - med) < 15)).any() else np.array([med] * 3)
        sub = out[band]; sub[bright] = col.astype(np.uint8); out[band] = sub
    return out


def variants(prompt, img, actions, anchors, meta_names):
    V = [("base", prompt, img, actions)]
    for d, c in (("dir_left", "turn left at the next intersection"), ("dir_straight", "go straight at the next intersection"),
                 ("dir_right", "turn right at the next intersection")):
        V.append((d, with_command(prompt, c), img, actions))
    for d, c in (("w_dist20", "go straight at the next intersection in 20 meter"), ("w_follow", "follow the road"),
                 ("w_para_ahead", "keep driving straight ahead"), ("w_para_noturn", "do not turn, continue straight"),
                 ("w_lmdrive", "Next intersection, go straight")):
        V.append((d, with_command(prompt, c), img, actions))
    for v in (3.0, 6.0, 10.0):
        V.append((f"spd_{int(v)}", SPD_RE.sub(f"Current speed: {v} m/s.", prompt, count=1), img, actions))
    if anchors is not None and actions.size:
        for i, name in enumerate(meta_names):
            V.append((f"anc_{name}", prompt, img, anchors[i].copy()))
    V.append(("shift_l40", prompt, shift(img, -40), actions)); V.append(("shift_r40", prompt, shift(img, 40), actions))
    V.append(("erase_lane", prompt, erase_marks(img, True), actions)); V.append(("erase_all", prompt, erase_marks(img, False), actions))
    return V


# ---------------------------------------------------------------- model
def load(ckpt, device):
    import hydra, torch
    from omegaconf import OmegaConf
    from safedrive_vla.models.backbone import load_processor
    ckpt = Path(ckpt); cfg_path = ckpt.parent / "config.yaml"
    if not cfg_path.exists(): cfg_path = ckpt.parent.parent / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    rp = lambda p: p if p is None or Path(p).is_absolute() else str(REPO / p)
    if cfg.model.get("world_model") is not None:
        for k in ("encoder_ckpt", "predictor_ckpt", "action_anchor_path"): cfg.model.world_model[k] = rp(cfg.model.world_model[k])
    cfg.model.action_token.codebook_path = rp(cfg.model.action_token.codebook_path)
    processor, _ = load_processor(cfg.model.variant, int(cfg.model.action_token.num_tokens))
    dd = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    model = hydra.utils.instantiate(cfg.model, processor=processor, gradient_checkpointing=False, _recursive_=False)
    torch.set_default_dtype(dd)
    if ckpt.is_dir():
        tag = (ckpt / "latest").read_text().strip() if (ckpt / "latest").exists() else "checkpoint"
        state = torch.load(ckpt / tag / "mp_rank_00_model_states.pt", map_location="cpu", weights_only=False)["module"]
    else:
        state = torch.load(ckpt, map_location="cpu", weights_only=False); state = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected: raise RuntimeError(f"unexpected keys {unexpected[:5]}")
    return model.to(device).eval(), processor, cfg


def infer(model, processor, device, prompt, img, tubelet, state, actions, target_points, tokenizer, action_tokenizer, horizon):
    import torch
    from PIL import Image
    conv = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
    text = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=[Image.fromarray(img)], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    tp = torch.from_numpy(target_points).to(device).unsqueeze(0)
    wm = [torch.from_numpy(x).to(device).unsqueeze(0) for x in (tubelet, state, actions)] if tubelet.size else [None] * 3
    model.set_conditioning(target_points=tp, wm_tubelet=wm[0], wm_state=wm[1], wm_actions=wm[2])
    plen = inputs["input_ids"].shape[1]
    try:
        gen = model.vlm.generate(**inputs, do_sample=False, max_new_tokens=128,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        path = None
        if bool((gen == model.actions_token_id).any()):
            _, hidden = model.forward_vlm(gen, torch.ones_like(gen), inputs["pixel_values"])
            p, _ = model.predict_path(gen, hidden.to(model.path_head.mlp[0].weight.dtype))
            path = p[0].float().cpu().numpy()
    finally:
        model.clear_conditioning()
    answer = tokenizer.decode(gen[0, plen:], skip_special_tokens=False)
    mode = next((m for m in ("STRICT", "CAUTIOUS", "FALLBACK") if f"<{m}>" in answer), None)
    idx = [t - model.action_start_id for t in gen[0, plen:].tolist() if model.action_start_id <= t < model.action_end_id][:horizon]
    idx += [0] * (horizon - len(idx))
    traj = action_tokenizer.rollout_xy(idx)
    speed = float(np.linalg.norm(traj[1] - traj[3]) * 2.0)
    return (None if path is None else round(path_heading(path), 1)), mode, round(speed, 2)


# ---------------------------------------------------------------- frames
def pick_frames(run_route_dir, xml):
    route = ET.parse(xml).getroot().find(".//route")
    texts = {norm(e.findtext("text")) for e in route.find("instructions")
             if e.find("expected_behavior").get("type") == "lane_follow" or e.find("expected_behavior").get("direction") == "straight"}
    dbg = {int(k): v for k, v in json.load(open(run_route_dir / "agent_debug.json")).items()}
    dumps = sorted(int(p.stem) for p in (run_route_dir / "dump").glob("*.npz"))
    win = [k for k in sorted(dbg) if norm(dbg[k].get("instruction")) in texts]
    if not win or not dumps: return [], None
    fw = next((k for k in win if dbg[k].get("speed", 0) > 1 and dbg[k].get("path") and abs(path_heading(dbg[k]["path"])) > 30), None)
    if fw is not None: targets = [(fw - 10, "-0.5s"), (fw - 5, "-0.25s"), (fw, "first_wrong")]
    else:
        end = win[-1]; targets = [(end - 40, "-2.0s_end"), (end - 30, "-1.5s_end"), (end - 20, "-1.0s_end")]
    out = []
    for t, lab in targets:
        cand = [d for d in dumps if d <= t] or dumps
        d = max(cand)
        if d not in [o[0] for o in out]: out.append((d, lab))
    return out, fw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--routes-dir", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--only", default="")
    a = ap.parse_args()
    import torch
    from safedrive_vla.action_tokenizer import ActionTokenizer
    from safedrive_vla.constants import META_COMMANDS
    from safedrive_vla.data.dataset import load_action_anchors
    out = Path(a.out); (out / "examples").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model, processor, cfg = load(a.ckpt, device)
    tok = processor.tokenizer; at = ActionTokenizer.from_file(cfg.model.action_token.codebook_path); hz = int(cfg.model.action_token.horizon)
    wmc = cfg.model.get("world_model")
    anchors = load_action_anchors(wmc.action_anchor_path, int(wmc.rollout_steps)) if wmc is not None else None
    recs = []; t0 = time.time()
    for xml in sorted(Path(a.routes_dir).glob("*.xml")):
        if a.only and not any(o in xml.stem for o in a.only.split(",")): continue
        att = sorted(glob.glob(str(Path(a.run) / "viz" / xml.stem / "*/")))
        if not att: print(f"skip {xml.stem}: no run"); continue
        rdir = Path(att[-1]); frames, fw = pick_frames(rdir, xml)
        print(f"== {xml.stem}: first wrong tick {fw}, frames {frames}", flush=True)
        for tick, lab in frames:
            z = np.load(rdir / "dump" / f"{tick:05d}.npz")
            img = cv2.imdecode(z["vla_png"], cv2.IMREAD_COLOR)[:, :, ::-1].copy()
            prompt = str(z["prompt"]); acts = z["wm_actions"]
            for name, p, im, ac in variants(prompt, img, acts, anchors, META_COMMANDS):
                h, mode, spd = infer(model, processor, device, p, im, z["wm_tubelet"], z["wm_state"], ac, z["target_points"], tok, at, hz)
                recs.append(dict(route=xml.stem, tick=tick, frame=lab, first_wrong=fw, variant=name, heading=h, mode=mode, model_speed=spd,
                                 instruction=str(z["instruction"]), anchor=int(z["wm_meta_command"])))
                if lab == frames[-1][1] and name in ("base", "shift_l40", "erase_lane", "erase_all"):
                    cv2.imwrite(str(out / "examples" / f"{xml.stem[12:]}_{tick}_{name}.jpg"), im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            json.dump(recs, open(out / "probe.json", "w"))
            print(f"   tick {tick} ({lab}) done, {len(recs)} records, {time.time() - t0:.0f}s", flush=True)
    # summary: heading per variant (rows) x frame (columns)
    lines = []
    for route in sorted({r["route"] for r in recs}):
        rr = [r for r in recs if r["route"] == route]; ticks = sorted({r["tick"] for r in rr})
        lines.append(f"\n=== {route[12:]}  instruction: {rr[0]['instruction']}  (first wrong tick {rr[0]['first_wrong']})")
        lines.append(f"  {'variant':16s} " + " ".join(f"{t:>14d}" for t in ticks))
        for v in dict.fromkeys(r["variant"] for r in rr):
            cells = []
            for t in ticks:
                x = next((r for r in rr if r["tick"] == t and r["variant"] == v), None)
                h = x["heading"] if x else None
                tag = "" if h is None else ("S" if abs(h) <= 30 else ("L" if h < 0 else "R"))
                cells.append(f"{('-' if h is None else f'{h:+.0f}'):>6} {tag:1s} {(x['mode'] or '?')[:4] if x else '':4s}")
            lines.append(f"  {v:16s} " + " ".join(f"{c:>14s}" for c in cells))
    txt = "\n".join(lines); print(txt); (out / "summary.txt").write_text(txt)


if __name__ == "__main__":
    main()
