# se13: CARLA-F subset for the speed-tracking A/B test

13 of the 35 scored CARLA-F track routes (copied from roboworld2026/track3,
routes/carla_f), chosen to span the target speeds and instruction types:

| route | town | target m/s | after the speed instruction |
|---|---|---|---|
| 39_lane_follow | Town12 | 27 | lane follow |
| 194_turn_straight | Town04 | 24 | straight |
| 209_turn_left | Town04 | 24 | left turn |
| 172_turn_right | Town15 | 23 | right, left |
| 143_turn_straight | Town13 | 18 | straight, lane change |
| 99_lane_follow | Town13 | 18 | lane follow |
| 181_turn_straight | Town07 | 16 | straight, left, straight |
| 65_lane_change_left | Town13 | 16 | three lane changes |
| 212_lane_follow | Town10HD | 15 | lane follow |
| 31_lane_change_left | Town12 | 13 | three lane changes |
| 111_turn_right | Town12 | 12 | right, follow, two lane changes |
| 09_turn_right | Town12 | 9 | right, two lane changes |
| 17_turn_right | Town12 | 6 | right turn |

All 5 scored routes above the 20.5 m/s codebook cap are in the 4 fastest rows
(the 5th, 209_turn_right, is the twin of 209_turn_left). The "off" arm is the
full baseline run on the same routes (work_dirs/track3/carla_f); only the "on"
arm (--speed-track) needs to be run. Watch both SE and NCR: speed tracking must
not break the turn and lane-change instructions that follow.
