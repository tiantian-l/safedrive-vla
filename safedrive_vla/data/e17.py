"""E19b (11/25): E17 paired junction frames with world-model anchor dropout ([27] option B).

E17 (database/e17_pairs, scripts/data/collect_pairs.py) has, per junction approach lane, one PDM-Lite route per
manoeuvre from the same start: the frames before the junction are the same picture, only the manoeuvre differs.
With the anchor (world-model dream) removed, the sentence is the only input that tells the manoeuvres apart.

Frames: the window of scripts/data/build_e17_index.py (<= 10 m before the entry + inside the junction until the
heading changed 30 deg; straight routes until the exit), routes of the index split ``split`` (train / holdout /
holdout_town). Prompt: the relabelling of safedrive_vla/data/relabel.py (competition-style sentence of the
manoeuvre with probability ``p_comp``, else the training format with the distance); world-model anchor = the
manoeuvre's anchor (``anchor_m``). Labels (trajectory, path, mode) are the expert's, never changed.

Dropout (``dropout: {mode, p}``), applied with probability p per sample:
  mask    the world block <world_begin> ... <world_end> is removed from attention (collate sets its attention_mask
          to 0): "no dream this time". Positions are unchanged (training forward uses arange position ids).
  follow  the world model dreams under the follow-the-road anchor instead of the manoeuvre's.
  none    no dropout (default).
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import ujson

from safedrive_vla.constants import META_COMMAND_TO_INDEX
from safedrive_vla.data.dataset import DrivingDataset, DrivingSample
from safedrive_vla.data.relabel import Relabeler
from safedrive_vla.data.simlingo_base import IMG_AUGMENTATION_PROB, BaseDataset, image_augmenter

DROPOUT_MODES = ("none", "mask", "follow")


class E17JunctionDataset(DrivingDataset):
    def __init__(
        self,
        root: str,
        index: str,
        sentences: str,
        codebook_path: str,
        lmdrive_templates: str,
        use_mode_token: bool = True,
        world_model: Optional[Dict[str, Any]] = None,
        split: str = "train",
        p_comp: float = 0.8,
        anchor_m: float = 15.0,
        dropout: Optional[Dict[str, Any]] = None,
        **_unused,
    ):
        # The SimLingo route scan / hash split of BaseDataset is replaced by the E17 index (window frames, own split).
        self.data_root = os.path.abspath(root)
        self.split = split
        self.bucket_name = "e17"
        self.data_scale = 1.0
        self.tfs = image_augmenter(prob=IMG_AUGMENTATION_PROB)
        with open(lmdrive_templates, "r") as f:
            self.command_templates = ujson.load(f)
        with open(index) as f:
            idx = json.load(f)
        splits = {split} if split != "holdout_all" else {"holdout", "holdout_town"}
        images, measurements, starts = [], [], []
        n_routes = n_dirty = 0
        for rel in sorted(idx["routes"]):
            if idx["split"].get(rel) not in splits:
                continue
            route_dir = os.path.join(self.data_root, rel)
            if not BaseDataset._route_is_clean(route_dir):
                n_dirty += 1
                continue
            n_routes += 1
            for fr in sorted(int(k) for k in idx["routes"][rel]):
                images.append(f"{route_dir}/rgb/{fr:04}.jpg")
                measurements.append(f"{route_dir}/measurements")
                starts.append(fr)
        self.images = np.array(images).astype(np.bytes_)
        self.measurements = np.array(measurements).astype(np.bytes_)
        self.sample_start = np.array(starts)
        self.dreams = None
        self._init_safedrive(codebook_path, use_mode_token, world_model)
        # Every indexed frame is in the window, so text_m covers all of them; no lane-keeping sentences elsewhere.
        self.relabeler = Relabeler(index=index, sentences=sentences, text_m=1e9, anchor_m=anchor_m,
                                   p_comp=p_comp, p_nonjunction=0.0, data_root=self.data_root)
        d = dict(dropout or {})
        self.dropout_mode = str(d.get("mode", "none"))
        self.dropout_p = float(d.get("p", 0.0))
        if self.dropout_mode not in DROPOUT_MODES:
            raise ValueError(f"e17 dropout mode must be one of {DROPOUT_MODES}, got {self.dropout_mode}")
        if self.dropout_mode == "follow" and self.action_anchors is None:
            raise ValueError("e17 dropout 'follow' needs the world model with action anchors")
        self.follow_index = META_COMMAND_TO_INDEX["follow_road"]
        print(f"[e17/{split}] {len(self.images)} window frames ({n_routes} routes, {n_dirty} skipped as imperfect); "
              f"dropout {self.dropout_mode} p {self.dropout_p}")

    def drop(self, sample: DrivingSample) -> DrivingSample:
        """The sample with its dream removed ('mask') or replaced by the follow-the-road dream ('follow')."""
        if self.dropout_mode == "mask":
            return sample._replace(wm_dropped=True)
        if self.dropout_mode == "follow" and sample.wm_actions is not None:
            return sample._replace(wm_actions=self.action_anchors[self.follow_index].copy())
        return sample

    def __getitem__(self, index: int) -> DrivingSample:
        sample = DrivingDataset.__getitem__(self, index)
        if self.dropout_mode != "none" and random.random() < self.dropout_p:
            sample = self.drop(sample)
        return sample
