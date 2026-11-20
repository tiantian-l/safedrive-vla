"""Junction relabelling of driving samples ([17], 11/20). Pure Python (no torch), so it can be tested offline.

Why: the route-planner ``command`` of a SimLingo frame switches to turn / go-straight only ~7-9 m before the
junction, while the expert's 2.5 s future already turns 10-30 m before it. Near junctions the prompt
(and the world-model anchor) therefore said "follow the road" while the label turned or went straight.
The index built by scripts/data/build_relabel_index.py stores, for frames approaching a junction, the
manoeuvre the expert actually makes there (the planner's own junction command, checked against the
expert's heading) and the distance to the junction entry. Labels (trajectory, path) are never changed.

Config (``data.relabel``):
  index:          relabel_index.json from build_relabel_index.py
  sentences:      JSON with competition-style paraphrases {left, right, straight, lane}
  text_m:         relabel the prompt within this distance before the junction entry (and inside it)
  anchor_m:       relabel the world-model anchor (and hence the mode rule) within this distance; < 0 = never
  p_comp:         probability of a competition-style sentence (no distance) for a relabelled frame;
                  otherwise the training format "Command: <text> in N meter." with N = distance to the junction
  p_nonjunction:  probability that a follow-the-road frame outside the window gets a lane-keeping sentence
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Optional, Tuple

COMMAND_TEXT = {1: "go left at the next intersection", 2: "go right at the next intersection",
                3: "go straight at the next intersection", 4: "follow the road"}
MANOEUVRE_TO_COMMAND = {"L": 1, "R": 2, "S": 3}


def normalise(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


_SHARED = {}


class Relabeler:
    @classmethod
    def shared(cls, **kwargs) -> "Relabeler":
        """One instance per configuration and process (every bucket dataset would otherwise load the index)."""
        key = tuple(sorted(kwargs.items()))
        if key not in _SHARED:
            _SHARED[key] = cls(**kwargs)
        return _SHARED[key]

    def __init__(self, index: str, sentences: str, text_m: float = 30.0, anchor_m: float = 15.0,
                 p_comp: float = 0.8, p_nonjunction: float = 0.2, data_root: str = ""):
        with open(index) as f:
            raw = json.load(f)
        self.routes = raw["routes"]
        with open(sentences) as f:
            self.lib = json.load(f)
        for k in ("left", "right", "straight", "lane"):
            if not self.lib.get(k):
                raise ValueError(f"sentence library lacks '{k}'")
        self.text_m, self.anchor_m = float(text_m), float(anchor_m)
        self.p_comp, self.p_nonjunction = float(p_comp), float(p_nonjunction)
        self.data_root = os.path.abspath(data_root) if data_root else ""
        n = sum(len(v) for v in self.routes.values())
        print(f"[relabel] {len(self.routes)} routes, {n} frames; text {self.text_m} m, anchor {self.anchor_m} m, "
              f"p_comp {self.p_comp}, p_nonjunction {self.p_nonjunction}")

    def route_key(self, measurement_path: str) -> str:
        route_dir = os.path.dirname(os.path.dirname(measurement_path))
        return os.path.relpath(route_dir, self.data_root) if self.data_root else route_dir

    def apply(self, measurement_path: str, frame: int, command: int,
              rng: random.Random = random) -> Optional[Tuple[str, int]]:
        """(navigation sentence, anchor command) for a relabelled frame, or None to keep the original."""
        if command in (5, 6):                       # lane changes are never touched
            return None
        entry = self.routes.get(self.route_key(measurement_path), {}).get(str(int(frame)))
        if entry is not None and entry[1] <= self.text_m:
            man, dist = entry[0], float(entry[1])
            new = MANOEUVRE_TO_COMMAND[man]
            if rng.random() < self.p_comp:
                kind = {"L": "left", "R": "right"}.get(man) or ("straight" if rng.random() < 0.5 else "lane")
                nav = f"Command: {rng.choice(self.lib[kind])}."
            elif dist >= 1.0:
                nav = f"Command: {COMMAND_TEXT[new]} in {int(round(dist))} meter."
            else:
                nav = f"Command: {COMMAND_TEXT[new]}."
            anchor = new if (self.anchor_m >= 0 and dist <= self.anchor_m) else command
            return nav, anchor
        if command == 4 and rng.random() < self.p_nonjunction:
            return f"Command: {rng.choice(self.lib['lane'])}.", command
        return None
