# fail_cf: re-run E1 CARLA-F failures that have no front-camera images yet

Copied from roboworld2026/track3 routes/carla_f. All 6 are scored routes the baseline
(submission 1044953, E1) failed and that norm20 runs never recorded failing with images:

- 39_lane_follow, 65_lane_change_left (lane-level: high-speed merge, lane-change timing)
- 41_lane_change_right, 212_lane_follow (trajectory on the GT, still judged failed)
- 194_turn_straight (left turn at the end of the 24 m/s segment)
- 111_turn_right (turned left at the first junction, judged a right-turn pass)

Run with --save-viz --debug-log, default agent (no --instr-norm / --force-mode).
Closed-loop runs flip often: a route may pass this time.
