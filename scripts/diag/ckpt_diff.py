"""Which parts of a merged fine-tune model.ckpt differ from the released one (E14 check, 11/21).

    OMP_NUM_THREADS=1 $PY scripts/diag/ckpt_diff.py ckpts/safedrive-vla/safedrive_vla/model.ckpt ckpts/ft/E14_Bp_s1560/model.ckpt

Groups tensors by component (vision encoder, LoRA, language-model base weights, embeddings / lm_head,
projector, heads) and prints, per group, how many tensors changed and the largest / typical change.
A frozen part (vision encoder with freeze_vision) must show 0 changed.
"""
import re, sys
from collections import defaultdict

import torch


def group(k: str) -> str:
    if "vision_model" in k: return "vision encoder"
    if "lora_" in k: return "LoRA (" + ("language model" if "language_model" in k else "other") + ")"
    if "embed_tokens" in k: return "embed_tokens"
    if "lm_head" in k: return "lm_head"
    if "multi_modal_projector" in k or ".mlp1." in k: return "vision->LM projector"
    if "language_model" in k: return "language model (non-LoRA)"
    return ".".join(k.split(".")[:2])


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return ck["state_dict"] if "state_dict" in ck else ck


a, b = load(sys.argv[1]), load(sys.argv[2])
stats = defaultdict(lambda: [0, 0, 0.0, []])   # n, changed, max, rel changes
for k, va in a.items():
    vb = b[k]
    d = (va.float() - vb.float()).abs()
    s = stats[group(k)]
    s[0] += 1
    m = float(d.max()) if d.numel() else 0.0
    if m > 0:
        s[1] += 1
        s[3].append(float(d.norm() / (va.float().norm() + 1e-12)))
    s[2] = max(s[2], m)
print(f"{'group':32s} {'tensors':>7s} {'changed':>7s} {'max|d|':>9s} {'median rel ||d||/||w||':>22s}")
for g, (n, c, m, rel) in sorted(stats.items()):
    med = sorted(rel)[len(rel) // 2] if rel else 0.0
    print(f"{g:32s} {n:7d} {c:7d} {m:9.2e} {med:22.2e}")
missing = set(b) - set(a)
if missing:
    print(f"{len(missing)} tensors only in {sys.argv[2]}, e.g. {sorted(missing)[:3]}")
# rows of the new <action_i> / mode tokens in the embedding (if present): changed rows
for k in a:
    if k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight") or "embed_tokens" in k and k.endswith("weight"):
        d = (a[k].float() - b[k].float()).abs().amax(dim=1)
        idx = torch.nonzero(d).flatten()
        print(f"{k}: {a[k].shape[0]} rows, {len(idx)} changed" + (f" (rows {int(idx.min())}..{int(idx.max())})" if len(idx) else ""))
