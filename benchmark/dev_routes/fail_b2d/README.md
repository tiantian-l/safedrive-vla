# fail_b2d: re-run the 25 E1 B2D-C routes with DS < 100, with images

Copied from roboworld2026/track3 routes/b2d_c: every B2D-C route the baseline
(submission 1044953, E1) did not finish with DS 100 (failure browser, 11/18).
Run with --save-viz --debug-log, default agent (forced creep on, as in E1).
4 routes are long (172, 18, 208, 47: 150-290 game s): use --route-walltime 21600.
