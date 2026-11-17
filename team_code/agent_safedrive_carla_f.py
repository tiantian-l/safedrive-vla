"""SafeDriveVLA agent for language-instruction benchmarks (CARLA-F).

The navigation signal is the active natural-language instruction of the route
XML (``ROUTES``), shown as ``Command: <instruction>.``; the route-planner
command and target waypoints are hidden from the model. Instructions are
triggered by the distance travelled (or by a scenario becoming active). Once
the ego completes a turn or lane-change instruction, the prompt falls back to
``follow the road``.

The action anchor of world-model dreaming is indexed by the meta-command
parsed from the instruction text (``instruction_meta_commands.json``, the
paraphrase pools of the benchmark). Target-speed instructions name no maneuver
and use ``follow_road``, the anchor their frames get in training.
"""

from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import carla
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from safedrive_vla.constants import META_COMMAND_TO_INDEX, TASK_PROMPT
from team_code.agent_safedrive import FORCE_MODE, SafeDriveAgent

INSTRUCTION_META_COMMANDS = Path(__file__).resolve().parent / "instruction_meta_commands.json"
NAVIGATION_COMMANDS = {1, 2, 3, 5, 6}   # route-planner commands of maneuver instructions
LANE_FOLLOW_COMMAND = 4
COMPLETION_MIN_PROGRESS_M = 3.0
SCENARIO_ACTIVE_DISTANCE_M = 30.0
# --force-mode oracle (DIAGNOSTIC ONLY, reads the simulator state): an actor is a
# hazard if it is, now or along a constant-velocity prediction, within this
# distance of the ego's last path-head path.
ORACLE_HORIZON_S = (0.0, 0.5, 1.0, 1.5, 2.0)
ORACLE_RADIUS_M = {"vehicle": 2.0, "walker": 1.5}
ORACLE_RANGE_M = 40.0
_SPEED_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*m\s*/\s*s", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
# Rewrite the active instruction into a phrasing seen in training (see
# SafeDriveLanguageAgent.normalized_navigation). Off by default.
INSTR_NORM = os.environ.get("SAFEDRIVE_INSTR_NORM", "0") == "1"
# Meta-command -> route-planner command text of the training prompts
# (``Command: <text>.``, SimLingo's COMMAND_TEXT without the distance).
META_COMMAND_TEXT = {
    "follow_road": "follow the road",
    "go_straight_intersection": "go straight at the next intersection",
    "turn_left": "go left at the next intersection",
    "turn_right": "go right at the next intersection",
    "lane_change_left": "do a lane change to the left",
    "lane_change_right": "do a lane change to the right",
}


def get_entry_point():
    return "SafeDriveLanguageAgent"


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


@dataclass
class Instruction:
    id: int
    text: str
    command_id: int
    trigger_type: str            # start | distance_traveled | scenario_active
    trigger_value: float = 0.0
    scenario_name: str = ""
    duration_meters: float = -1.0
    # runtime state
    is_active: bool = False
    start_distance: float = 0.0
    planner_command_seen: bool = False


class SafeDriveLanguageAgent(SafeDriveAgent):
    COMPLETION_MONITOR = True
    # instruction text -> meta-command; text outside the table: no anchor
    META_COMMAND_TABLE = INSTRUCTION_META_COMMANDS
    UNKNOWN_META_COMMAND: Optional[str] = None
    # Instruction normalization (SAFEDRIVE_INSTR_NORM=1) only for CARLA-F; the
    # conflict agent keeps the raw text, whose unsafe requests the model must refuse.
    NORMALIZE_INSTRUCTION = True

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self.instructions: List[Instruction] = []
        self.current_instruction: Optional[Instruction] = None
        self.completed_ids = set()
        self.distance_traveled = 0.0
        self.last_location: Optional[carla.Location] = None
        self.scenario_variables: Dict[str, str] = {}
        self.scenario_trigger_points: Dict[str, carla.Location] = {}
        self._hero = None
        self._parse_route_xml(os.environ.get("ROUTES", ""))
        with open(self.META_COMMAND_TABLE) as f:
            self.instruction_meta_commands = {_normalize(k): v for k, v in json.load(f).items()}
        self.instruction = "follow the road"

    def _parse_route_xml(self, path: str) -> None:
        route = ET.parse(path).getroot().find(".//route")
        for elem in route.find("instructions").findall("instruction"):
            trigger = elem.find("trigger")
            self.instructions.append(Instruction(
                id=int(elem.attrib.get("id", 0)),
                text=elem.findtext("text") or "follow the road",
                command_id=int(elem.findtext("command_id") or LANE_FOLLOW_COMMAND),
                trigger_type=trigger.attrib.get("type", "start") if trigger is not None else "start",
                trigger_value=float(trigger.attrib.get("value", 0.0)) if trigger is not None else 0.0,
                scenario_name=trigger.attrib.get("scenario_name", "") if trigger is not None else "",
                duration_meters=float(elem.findtext("duration_meters") or -1.0),
            ))
        self.instructions.sort(key=lambda x: x.id)
        scenarios = route.find("scenarios")
        for i, elem in enumerate(scenarios.findall("scenario") if scenarios is not None else []):
            name = elem.attrib.get("name", f"scenario_{i}")
            self.scenario_variables[name] = f"ScenarioRouteNumber{i}"
            point = elem.find("trigger_point")
            if point is not None:
                self.scenario_trigger_points[name] = carla.Location(
                    x=float(point.attrib.get("x", 0)), y=float(point.attrib.get("y", 0)), z=float(point.attrib.get("z", 0)))

    # ------------------------------------------------------------------
    # instruction selection
    # ------------------------------------------------------------------
    def _ego_location(self) -> Optional[carla.Location]:
        if self._hero is None:
            self._hero = CarlaDataProvider.get_hero_actor()
        if self._hero is None:
            return None
        location = CarlaDataProvider.get_location(self._hero)
        if location is None:
            self._hero = CarlaDataProvider.get_hero_actor()
            location = CarlaDataProvider.get_location(self._hero) if self._hero is not None else None
        return location

    def _scenario_active(self, instr: Instruction, ego_location: carla.Location) -> bool:
        if not instr.scenario_name:
            return False
        variable = self.scenario_variables.get(instr.scenario_name)
        try:
            if variable and py_trees.blackboard.Blackboard().get(variable):
                return True
        except Exception:
            pass
        # Fall back to the proximity of the scenario's actors.
        world = CarlaDataProvider.get_world()
        max_distance = instr.trigger_value if instr.trigger_value > 0 else SCENARIO_ACTIVE_DISTANCE_M
        trigger_location = self.scenario_trigger_points.get(instr.scenario_name)
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name", "") != "scenario":
                continue
            location = actor.get_location()
            if location is None or location.z < -10.0:
                continue
            if trigger_location is not None and location.distance(trigger_location) <= max_distance:
                return True
            if location.distance(ego_location) <= max_distance:
                return True
        return False

    def _active_instruction(self, ego_location: carla.Location) -> Optional[Instruction]:
        """The last triggered instruction that is neither completed nor expired."""
        active = None
        for instr in self.instructions:
            if instr.id in self.completed_ids:
                continue
            if instr.trigger_type == "start":
                triggered = True
            elif instr.trigger_type == "distance_traveled":
                triggered = self.distance_traveled >= instr.trigger_value
            elif instr.trigger_type == "scenario_active":
                triggered = self._scenario_active(instr, ego_location)
            else:
                raise ValueError(f"Unsupported instruction trigger: {instr.trigger_type}")
            if triggered:
                if not instr.is_active:
                    instr.is_active = True
                    instr.start_distance = self.distance_traveled
                if instr.duration_meters > 0 and self.distance_traveled > instr.start_distance + instr.duration_meters:
                    triggered = False
            if triggered:
                active = instr
        return active

    def on_route_command(self, current_command: int) -> None:
        """A maneuver instruction is completed once the route planner has gone
        through its command and is back to lane following."""
        instr = self.current_instruction
        if not self.COMPLETION_MONITOR or instr is None or instr.command_id not in NAVIGATION_COMMANDS:
            return
        if current_command == instr.command_id:
            instr.planner_command_seen = True
        progress = self.distance_traveled - instr.start_distance
        if instr.planner_command_seen and current_command == LANE_FOLLOW_COMMAND and progress >= COMPLETION_MIN_PROGRESS_M:
            self.completed_ids.add(instr.id)
            instr.is_active = False
            self.current_instruction = None
            self.instruction = "follow the road"

    def tick(self, input_data):
        location = self._ego_location()
        if location is not None:
            if self.last_location is not None:
                self.distance_traveled += location.distance(self.last_location)
            self.last_location = location
            self.current_instruction = self._active_instruction(location)
        instr = self.current_instruction
        self.instruction = instr.text.rstrip(". ") if instr is not None else "follow the road"
        self.navigation_override = self.normalized_navigation() if INSTR_NORM and self.NORMALIZE_INSTRUCTION else None
        return super().tick(input_data)

    def meta_command_name(self) -> Optional[str]:
        """Meta-command parsed from the active instruction's text (never from the route)."""
        text = (self.current_instruction.text if self.current_instruction is not None else self.instruction).strip()
        meta = self.instruction_meta_commands.get(_normalize(text))
        if meta is None and _SPEED_RE.search(text):
            meta = "follow_road"
        return meta if meta is not None else self.UNKNOWN_META_COMMAND

    def normalized_navigation(self) -> Optional[str]:
        """The active instruction in a training-time phrasing (prompt after
        "Current speed: v m/s."), built only from the instruction text:

        * a target-speed instruction -> ``Command: follow the road. <Text>.``, the
          action-dreaming format (navigation sentence + instruction, no task prompt);
        * any other instruction -> ``Command: <planner text>. Predict the actions.``,
          the driving format, e.g. "go left at the next intersection" (no distance).

        ``None`` (keep the raw instruction) when the text maps to no meta-command."""
        meta = self.meta_command_name()
        if meta is None:
            return None
        command = f"Command: {META_COMMAND_TEXT[meta]}."
        if self.instruction_named_speed() is not None and meta == "follow_road":
            text = self.instruction[:1].upper() + self.instruction[1:]
            return f"{command} {text}."
        return f"{command} {TASK_PROMPT}"

    def instruction_named_speed(self) -> Optional[float]:
        """Speed named by the active instruction, if it is a target-speed
        instruction (one with a speed and a duration)."""
        instr = self.current_instruction
        if instr is None or instr.duration_meters <= 0:
            return None
        match = _SPEED_RE.search(instr.text)
        return float(match.group(1)) if match else None

    # ------------------------------------------------------------------
    # --force-mode (diagnostic)
    # ------------------------------------------------------------------
    def forced_mode_token(self) -> Optional[str]:
        """strict_nav: <STRICT> while a straight / lane-follow instruction (no
        target speed) is active; built from the instruction text only.
        oracle: <FALLBACK> for a red light or a crossing / oncoming actor on the
        ego's path, <CAUTIOUS> for a lead vehicle or a standing obstacle on it,
        otherwise the model's own mode; oracle_full: <STRICT> when nothing is
        in the way. The oracles read privileged simulator state: diagnostics
        only, never for a submission."""
        self.oracle_hazard = None
        if FORCE_MODE == "strict_nav":
            instr = self.current_instruction
            if instr is None or _SPEED_RE.search(instr.text):
                return None
            return "<STRICT>" if self.meta_command_name() in ("follow_road", "go_straight_intersection") else None
        if FORCE_MODE in ("oracle", "oracle_full"):
            self.oracle_hazard = self._oracle_hazard()
            if self.oracle_hazard in ("red_light", "crossing"):
                return "<FALLBACK>"
            if self.oracle_hazard in ("lead", "static"):
                return "<CAUTIOUS>"
            return "<STRICT>" if FORCE_MODE == "oracle_full" else None
        if FORCE_MODE:
            raise ValueError(f"Unknown SAFEDRIVE_FORCE_MODE: {FORCE_MODE}")
        return None

    def _oracle_hazard(self) -> Optional[str]:
        """Privileged hazard on the ego's path (diagnostic only): red_light |
        crossing (a moving actor not driving ahead of us in our direction) |
        lead (a vehicle ahead moving our way) | static (a standing actor) | None."""
        if self._hero is None or self._ego_location() is None:
            return None
        hero = self._hero
        try:
            if hero.is_at_traffic_light() and hero.get_traffic_light_state() == carla.TrafficLightState.Red:
                return "red_light"
        except RuntimeError:
            pass
        path = getattr(self, "last_path", None)
        path = [(float(p[0]), float(p[1])) for p in path] if path is not None else [(float(x), 0.0) for x in range(1, 21)]
        path = [(0.0, 0.0)] + path
        tf = hero.get_transform()
        ego, fwd, right = tf.location, tf.get_forward_vector(), tf.get_right_vector()

        def to_ego(v):  # world vector -> (forward, right)
            return v.x * fwd.x + v.y * fwd.y, v.x * right.x + v.y * right.y

        def dist_to_path(px, py):
            best = float("inf")
            for (ax, ay), (bx, by) in zip(path, path[1:]):
                dx, dy = bx - ax, by - ay
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-9)))
                best = min(best, math.hypot(px - ax - t * dx, py - ay - t * dy))
            return best

        found = set()
        world = CarlaDataProvider.get_world()
        for actor in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.*")):
            if actor.id == hero.id:
                continue
            loc = actor.get_location()
            if loc is None or loc.z < -10.0 or loc.distance(ego) > ORACLE_RANGE_M:
                continue
            x, y = to_ego(loc - ego)
            if x < -1.0 and actor.type_id.startswith("vehicle"):
                continue  # behind the ego: not ours to yield to
            vx, vy = to_ego(actor.get_velocity())
            radius = ORACLE_RADIUS_M["walker" if actor.type_id.startswith("walker") else "vehicle"]
            if not any(dist_to_path(x + vx * t, y + vy * t) < radius for t in ORACLE_HORIZON_S):
                continue
            speed = math.hypot(vx, vy)
            if speed < 0.5:
                found.add("static")
            elif x > 0 and vx > 2.0 and abs(vy) < 1.0 and actor.type_id.startswith("vehicle"):
                found.add("lead")
            else:
                found.add("crossing")
        for kind in ("crossing", "lead", "static"):
            if kind in found:
                return kind
        return None

    def anchor_meta_command(self) -> int:
        """Meta-command parsed from the instruction text."""
        meta = self.meta_command_name()
        return META_COMMAND_TO_INDEX[meta] if meta is not None else -1
