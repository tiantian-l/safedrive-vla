# e7sub: subset for E6 + E7 (--instr-norm-speed + --curve-speed), 11/21

Copied from benchmark/data/carla_f (identical to roboworld2026/track3 routes/carla_f for these
routes). All 10 are scored routes.

- M2 "direction right, too fast / turn ended early": 103_turn_right (fails 2/2 with the speed
  rewrite, 1/4 without), 10_turn_right (fails in every run).
- 111_turn_right: wrong lane after the right turn in E3 / E13.
- Turns that pass and must keep passing: 09, 142 (turns at ~10 m/s), 17_turn_right.
- SE guard (turn right after the speed window / >= 22 m/s): 209_turn_left, 209_turn_right,
  172_turn_right (stalled in E6), 39_lane_follow (27 m/s).

Compare with E6 (work_dirs/E6_speed/carla_f, same routes) and E1 / norm20 runs.
