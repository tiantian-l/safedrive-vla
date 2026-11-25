"""Data module: bucketed sampling of driving + action-dreaming samples and
chat-template collation for the InternVL3 processor."""

from __future__ import annotations

import itertools
import os
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from safedrive_vla.constants import WORLD_BEGIN_TOKEN, WORLD_END_TOKEN, world_block_text
from safedrive_vla.data.dataset import ActionDreamingDataset, DrivingDataset, DrivingSample

IGNORE_INDEX = -100
MAX_LENGTH = 2048
LMDRIVE_TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "lmdrive_commands.json")


class TrainingBatch(NamedTuple):
    input_ids: torch.Tensor          # [B, L] prompt + target, left padded
    attention_mask: torch.Tensor     # [B, L]
    labels: torch.Tensor             # [B, L], IGNORE_INDEX on prompt and padding
    pixel_values: torch.Tensor       # processor image tiles
    target_points: torch.Tensor      # [B, 2, 2]
    path: torch.Tensor               # [B, 20, 2] path-head target
    wm_tubelet: Optional[torch.Tensor] = None   # [B, 2, 3, H, W] uint8
    wm_state: Optional[torch.Tensor] = None     # [B, 4]
    wm_actions: Optional[torch.Tensor] = None   # [B, K, 3]


def mask_world_blocks(input_ids: torch.Tensor, attention_mask: torch.Tensor, dropped: List[bool],
                      begin_id: int, end_id: int) -> torch.Tensor:
    """E19b anchor dropout 'mask': attention mask with the world block <world_begin> ... <world_end> of the dropped
    rows set to 0, so nothing after it can attend to the dream ("no dream this time"). Position ids are unaffected:
    the training forward passes none, and Qwen2 then uses arange (checked: transformers 5.5.3 modeling_qwen2)."""
    out = attention_mask.clone()
    for b, drop in enumerate(dropped):
        if not drop:
            continue
        i0 = (input_ids[b] == begin_id).nonzero(as_tuple=False).flatten()
        i1 = (input_ids[b] == end_id).nonzero(as_tuple=False).flatten()
        if i0.numel() != 1 or i1.numel() != 1 or int(i1[0]) <= int(i0[0]):
            raise RuntimeError("wm_dropped sample without exactly one world block")
        out[b, int(i0[0]): int(i1[0]) + 1] = 0
    return out


class SafeDriveDataModule(LightningDataModule):
    """Mixes driving and action-dreaming samples 50/50. Within each, samples
    are drawn from SimLingo's scenario buckets with the given ``train_partitions``
    weights (``all`` = every frame)."""

    def __init__(
        self,
        processor,
        data_root: str,
        data_scale: float,
        codebook_path: str,
        action_start_id: int,
        num_action_tokens: int,
        batch_size: int,
        num_workers: int,
        train_partitions: Dict[str, float],
        train_partitions_action_dreaming: Dict[str, float],
        use_mode_token: bool = True,
        world_model: Optional[Dict[str, Any]] = None,
        dreaming_fraction: float = 0.5,
        relabel: Optional[Dict[str, Any]] = None,
        extra_dreaming: Optional[List[Dict[str, Any]]] = None,
        speed_sentences: Optional[Dict[str, Any]] = None,
        e17: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # E19b: E17 paired junction frames with anchor dropout (safedrive_vla/data/e17.py): {root, index, weight,
        # sentences, p_comp, anchor_m, dropout: {mode: none | mask | follow, p}}. Its weight is an absolute share of
        # all samples; every other bucket (driving and dreaming) is scaled by (1 - weight). Off by default.
        self.e17 = dict(e17) if e17 else None
        # E19a: extra action-dreaming roots, each its own bucket with an absolute share of all samples
        # ([28] red-light dreams: {name: red_light, root: database/e17_red, weight: 0.03}); the SimLingo
        # buckets keep their relative weights and share the rest. Off by default.
        self.extra_dreaming = [dict(x) for x in (extra_dreaming or [])]
        # E19a [29]: rephrase SimLingo speed dreams ({path, p}); off by default.
        self.dreaming_kwargs = dict(speed_sentences=dict(speed_sentences)) if speed_sentences else {}
        self.dreaming_fraction = float(dreaming_fraction)
        self.driving_kwargs = dict(relabel=dict(relabel)) if relabel else {}
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.action_ids = (int(action_start_id), int(action_start_id) + int(num_action_tokens))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.partitions = [dict(train_partitions), dict(train_partitions_action_dreaming)]
        self.world_model = dict(world_model or {})
        self.wm_enabled = bool(self.world_model)
        self.dataset_kwargs = dict(
            data_root=data_root,
            data_scale=data_scale,
            lmdrive_templates=LMDRIVE_TEMPLATES,
            codebook_path=codebook_path,
            use_mode_token=use_mode_token,
            world_model=self.world_model,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        bucket_names: List[str] = []
        bucket_weights: List[float] = []
        datasets = {}
        for dataset_cls, partitions in zip((DrivingDataset, ActionDreamingDataset), self.partitions):
            total = sum(partitions.values())
            dreaming = dataset_cls is ActionDreamingDataset
            suffix = "_action_dreaming" if dreaming else ""
            extra = self.dreaming_kwargs if dreaming else self.driving_kwargs
            share = self.dreaming_fraction if dreaming else 1.0 - self.dreaming_fraction
            for bucket, weight in partitions.items():
                ds = dataset_cls(split="train", bucket_name=bucket, **self.dataset_kwargs, **extra)
                if len(ds) == 0:
                    continue
                datasets[bucket + suffix] = ds
                bucket_names.append(bucket + suffix)
                bucket_weights.append(float(weight) / total * share)

        # The extra dreams take their share from the SimLingo dreaming buckets only, so every driving bucket
        # (e.g. leading_object_traffic.traffic_light) keeps exactly its share of all samples.
        extra_share = sum(float(x["weight"]) for x in self.extra_dreaming)
        if not 0.0 <= extra_share < self.dreaming_fraction:
            raise ValueError(f"extra_dreaming weights must sum to [0, dreaming_fraction), got {extra_share}")
        if extra_share > 0.0:
            scale = (self.dreaming_fraction - extra_share) / self.dreaming_fraction
            bucket_weights = [w * scale if b.endswith("_action_dreaming") else w for b, w in zip(bucket_names, bucket_weights)]
        for x in self.extra_dreaming:
            kwargs = dict(self.dataset_kwargs, data_root=os.path.abspath(x["root"]))
            ds = ActionDreamingDataset(split="train", bucket_name="all", **kwargs)
            if len(ds) == 0:
                raise ValueError(f"extra_dreaming {x['name']}: no dream frames under {x['root']}")
            name = f"{x['name']}_action_dreaming"
            datasets[name] = ds
            bucket_names.append(name)
            bucket_weights.append(float(x["weight"]))

        if self.e17 and float(self.e17.get("weight", 0.0)) > 0.0:
            from safedrive_vla.data.e17 import E17JunctionDataset
            w = float(self.e17["weight"])
            if not 0.0 < w < 1.0:
                raise ValueError(f"e17 weight must be in (0, 1), got {w}")
            kw = {k: v for k, v in self.e17.items() if k != "weight"}
            ds = E17JunctionDataset(split="train", codebook_path=self.dataset_kwargs["codebook_path"],
                                    lmdrive_templates=LMDRIVE_TEMPLATES, use_mode_token=self.dataset_kwargs["use_mode_token"],
                                    world_model=self.world_model, **kw)
            if len(ds) == 0:
                raise ValueError(f"e17: no window frames under {self.e17.get('root')}")
            bucket_weights = [x * (1.0 - w) for x in bucket_weights]
            datasets["e17"] = ds
            bucket_names.append("e17")
            bucket_weights.append(w)

        self.train_dataset = ConcatDataset([datasets[b] for b in bucket_names])
        per_sample = itertools.chain.from_iterable([w] * len(datasets[b]) for b, w in zip(bucket_names, bucket_weights))
        num_samples = int(min(len(datasets[b]) / max(w, 1e-9) for b, w in zip(bucket_names, bucket_weights)))
        self.train_sampler = WeightedRandomSampler(list(per_sample), num_samples=num_samples, replacement=True)
        print(f"[data] train buckets {dict(zip(bucket_names, bucket_weights))}, {num_samples} samples per epoch")

        self.val_dataset = ConcatDataset([
            DrivingDataset(split="val", bucket_name="all", **self.dataset_kwargs, **self.driving_kwargs),
            ActionDreamingDataset(split="val", bucket_name="all", **self.dataset_kwargs),
        ])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler,
                          num_workers=self.num_workers, drop_last=True, pin_memory=True, collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=True, pin_memory=True, collate_fn=self.collate)

    # ------------------------------------------------------------------
    def collate(self, samples: List[DrivingSample]) -> TrainingBatch:
        images = [Image.fromarray(np.transpose(s.image[0], (1, 2, 0))) for s in samples]
        conversations = []
        for s in samples:
            conv = [dict(turn, content=[dict(part) for part in turn["content"]]) for turn in s.conversation]
            if self.wm_enabled:  # world tokens go after the user prompt
                conv[0]["content"][0]["text"] += world_block_text(int(self.world_model.get("rollout_steps", 5)))
            conversations.append(conv)
        prompts = [[turn for turn in conv if turn["role"] == "user"] for conv in conversations]

        def encode(convs, add_generation_prompt: bool):
            text = self.processor.apply_chat_template(convs, add_generation_prompt=add_generation_prompt, tokenize=False)
            return self.processor(text=text, images=images, padding=True, return_tensors="pt",
                                  truncation=True, max_length=MAX_LENGTH)

        full = encode(conversations, add_generation_prompt=False)
        prompt_only = encode(prompts, add_generation_prompt=True)

        # Supervise only the assistant tokens.
        input_ids = full["input_ids"]
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        for b in range(len(samples)):
            non_pad = (input_ids[b] != pad_id).nonzero(as_tuple=False).flatten()
            if non_pad.numel() == 0:
                continue
            prompt_len = int(prompt_only["attention_mask"][b].sum().item())
            labels[b, : int(non_pad[0].item()) + prompt_len] = IGNORE_INDEX
            labels[b, input_ids[b] == pad_id] = IGNORE_INDEX
            if not bool(((labels[b] >= self.action_ids[0]) & (labels[b] < self.action_ids[1])).any()):
                raise RuntimeError("Sample truncated before its action tokens; increase MAX_LENGTH.")

        attention_mask = full["attention_mask"]
        dropped = [bool(getattr(s, "wm_dropped", False)) for s in samples]
        if any(dropped):
            attention_mask = mask_world_blocks(input_ids, attention_mask, dropped,
                                               self.tokenizer.convert_tokens_to_ids(WORLD_BEGIN_TOKEN),
                                               self.tokenizer.convert_tokens_to_ids(WORLD_END_TOKEN))

        wm = {}
        if self.wm_enabled:
            wm = dict(
                wm_tubelet=torch.from_numpy(np.stack([s.wm_tubelet for s in samples])),
                wm_state=torch.from_numpy(np.stack([s.wm_state for s in samples]).astype(np.float32)),
                wm_actions=torch.from_numpy(np.stack([s.wm_actions for s in samples]).astype(np.float32)),
            )
        return TrainingBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=full["pixel_values"],
            target_points=torch.tensor(np.stack([s.target_points for s in samples]), dtype=torch.float32),
            path=torch.tensor(np.stack([s.path for s in samples]), dtype=torch.float32),
            **wm,
        )
