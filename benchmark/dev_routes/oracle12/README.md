# oracle12: B2D-C subset for the --force-mode oracle diagnostic (11/17)

Question: if the model put out the right driving mode at the right time, would the
B2D-C driving improve? (= the ceiling of better mode labels, without training.)
Three failing routes per group of the 11/9 baseline (DS in brackets):

| group | routes |
|---|---|
| junction with conflicting traffic | 104 (60), 216 (42), 198 (37) |
| red light | 59 (70), 97 (70), 98 (70) |
| accelerating into an obstacle | 147 (50, pedestrian), 64 (50, pedestrian), 192 (60, hard-braking lead) |
| blocked lane | 65 (36), 161 (60), 169 (60) |

Baseline (model's own mode) = the 11/9 full run, work_dirs/track3/b2d_c.
--force-mode oracle / oracle_full read privileged simulator state: DIAGNOSTIC ONLY,
never part of a submission.
