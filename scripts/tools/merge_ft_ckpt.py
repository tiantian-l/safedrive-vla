"""Turn a fine-tuning DeepSpeed checkpoint into a model.ckpt the agent loads (E14, 11/20).

DeepSpeed (exclude_frozen_parameters) stores only the trainable tensors, so they are written over the released
model.ckpt (the fine-tune's init_ckpt): frozen parts (e.g. the vision encoder with freeze_vision, the world model)
come from it unchanged. The released config.yaml is copied next to the output (same architecture; the agent reads it).

    $PY scripts/tools/merge_ft_ckpt.py --base ckpts/safedrive-vla/safedrive_vla/model.ckpt \
        --ds work_dirs/safedrive_vla/E14_ft_Bp/<run>/checkpoints/step=001560.ckpt --out ckpts/ft/E14_Bp_s1560/model.ckpt
"""
import argparse, shutil, sys
from pathlib import Path

# Lightning .ckpt files pickle objects from safedrive_vla (hyper-parameters): make the repo importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--base", required=True); ap.add_argument("--ds", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()

base = torch.load(a.base, map_location="cpu", weights_only=False)
sd = base["state_dict"] if "state_dict" in base else base
ds_dir = Path(a.ds)
if ds_dir.is_dir():   # DeepSpeed: trainable tensors only
    tag = (ds_dir / "latest").read_text().strip() if (ds_dir / "latest").exists() else "checkpoint"
    trained = torch.load(ds_dir / tag / "mp_rank_00_model_states.pt", map_location="cpu", weights_only=False)["module"]
    full = False
else:                 # Lightning DDP .ckpt (save_weights_only): every tensor of the module
    tag = ""
    trained = torch.load(ds_dir, map_location="cpu", weights_only=False)["state_dict"]
    full = True

unknown, changed, groups = [], 0, {}
for k, v in trained.items():
    kk = k
    for pref in ("_forward_module.", "module."):
        if kk.startswith(pref): kk = kk[len(pref):]
    if kk not in sd:
        unknown.append(kk); continue   # full checkpoints: frozen weights loaded from HF / world-model files
    if tuple(sd[kk].shape) != tuple(v.shape):
        raise SystemExit(f"shape mismatch {kk}: {tuple(sd[kk].shape)} vs {tuple(v.shape)}")
    diff = float((sd[kk].float() - v.float()).abs().max())
    sd[kk] = v.to(sd[kk].dtype)
    changed += 1
    g = ".".join(kk.split(".")[:2]); groups.setdefault(g, [0, 0.0]); groups[g][0] += 1; groups[g][1] = max(groups[g][1], diff)
if unknown and full:
    print(f"{len(unknown)} tensors not in the base checkpoint skipped (frozen, loaded from HF / world-model files at init)")
elif unknown:
    raise SystemExit(f"{len(unknown)} trained tensors not in the base checkpoint, e.g. {unknown[:5]}")
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
torch.save(base, out)
shutil.copy(Path(a.base).parent / "config.yaml", out.parent / "config.yaml")
print(f"{changed} / {len(sd)} tensors replaced from {ds_dir}/{tag} -> {out}")
for g, (n, d) in sorted(groups.items()):
    print(f"  {g:40s} {n:5d} tensors, max |change| {d:.2e}")
