# norm20: A/B subset for --instr-norm (CARLA-F, 20 of the 40 public routes)

Copied from roboworld2026/track3 routes/carla_f. All 20 are scored routes.

- 10 routes the baseline (submission 1044953) failed at a junction or lane-follow
  instruction: 10_turn_right, 111_turn_right, 151, 159, 17_turn_straight, 194, 25,
  55, 56, 90_lane_follow.
- 10 routes the baseline passed completely, to catch regressions (turns, straight,
  lane changes, lane follow, speed segments): 09, 103, 142, 143, 137, 200,
  181_turn_straight, 31, 44, 218. Long routes (175, 181_turn_left) are left out.

Run off and on with --debug-log in the same session; compare NCR / SE with
benchmark/metrics carla_f_metrics and the prompts in agent_debug.json.
