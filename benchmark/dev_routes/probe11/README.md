# probe11: junction frames for the offline probe (E9 dump -> E10 probe)

Copied from roboworld2026/track3 routes/carla_f (via norm20). Straight / lane-follow junctions:
- 8 that fail: 17, 25, 151, 159 (go straight), 55, 56, 90, 218 (lane follow)
- 3 controls that pass: 181 (same instruction text as 17, lane can also turn left),
  137 (same text as 17, straight-only lane), 200 (straight-only lane)

Run with --dump --debug-log --save-viz (default agent), then scripts/diag/probe.py on the run.
