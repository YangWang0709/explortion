Implementation Plan
Updated: 2026-05-30

Completed Stage 2A:

1. Preserve existing official repo changes before editing.
2. Inspect SSCNet model, test.py checkpoint loading, dataloader output, metrics class convention, and config.
3. Implement standalone best-checkpoint `.npz` inference:
   `offline_infer_npz.py`
4. Implement read-only prediction access layer:
   `prediction_layer.py`
5. Implement smoke test:
   `test_offline_prediction_layer.py`
6. Verify single sample and first 5 test samples in `env_isaaclab`.
7. Update project notes and context.

Completed Stage 2B:

1. Implement strict paper-faithful expert scorer:
   `sc_explorer_paper_expert.py`
2. Build measured set S only from sensor-derived NYU fields:
   `tsdf_lr`, `position`, or their union.
3. Define P as predicted-by-SC and not measured:
   `PredictionLayer.confidence >= tau and not S`.
4. Implement paper gains:
   `I_exp`, `I_sc`, `I_hybrid`, `I_occ`, and `I_conf`.
5. Implement non-blocking ray casting by default, plus optional `sc_blocking` ablation.
6. Implement approximate position/yaw time cost and per-candidate gain/cost utilities.
7. Implement CLI:
   `run_paper_expert_offline.py`
8. Implement smoke test:
   `test_paper_expert.py`
9. Disable earlier target-label mock observed-map prototype entry points.
10. Verify single sample and first 5 existing prediction outputs only.
11. Update notes and context.

Completed Stage 2C:

1. Implement reusable paper expert dataset generation:
   `generate_paper_expert_dataset.py`
2. Build output directory structure:
   `samples/`, `predictions/`, and `logs/`.
3. Save one expert `.npz` sample per scene with candidate features,
   feature names, top candidate ids, expert action, scores, paths, modes,
   sample id, strict no-target note, and tree limitation note.
4. Save `manifest.jsonl` with ok/failed rows. Failed samples are recorded,
   not silently skipped.
5. Save `metadata.json` with checkpoint, code version, scorer/tool paths,
   feature names, scorer parameters, strict no-target flag, and Eq. 12
   limitation flag.
6. Reuse existing Stage 2A batch5 prediction outputs for smoke generation.
7. Implement format/leakage validator:
   `test_paper_expert_dataset.py`
8. Verify Stage 2C smoke dataset:
   5 ok, 0 failed, 5 sample `.npz` files, manifest, metadata, combined smoke npz.
9. Verify forbidden target fields are absent from expert samples.
10. Update notes and context.

Completed Stage 3A:

1. Create IL module directory:
   `ssc_network/il/`
2. Implement paper expert dataset loading:
   `il/paper_expert_dataset.py`
3. Implement `PaperExpertDataset`, `collate_paper_expert_batch`,
   `compute_feature_stats`, and `save_feature_stats`.
4. Ensure Dataset reads only expert sample `.npz` files from manifest ok rows,
   not raw `sample_npz` or `prediction_npz`.
5. Validate expert action range, valid-mask action, finite feature/scores, and
   absence of `target_lr`, `target_hr`, `gt`, and `ground_truth` fields.
6. Implement variable-candidate padding in collate.
7. Implement `CandidateMLPPolicy` skeleton:
   shared candidate MLP, logits `[B, N]`, invalid candidates masked to `-1e9`.
8. Implement `train_bc.py` skeleton:
   training disabled unless `--dry_run`; dry-run performs one forward and CE
   loss only, with no optimizer step and no model save.
9. Implement dataset smoke test:
   `il/test_dataset.py`
10. Run Stage 3A smoke tests on the Stage 2C smoke dataset:
    dataset size 5, item shape `(16, 15)`, batch shape `(2, 16, 15)`,
    stats shape `(15,)`, logits `(2, 16)`, CE loss finite.
11. Save feature stats:
    `/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/feature_stats.npz`
12. Update notes and context.

Completed Stage 4A-1:

1. Re-read local project context and `ssc_network_training_notes.md`.
2. Confirmed current next step is simulator continuous exploration sensing,
   not NYU static rollout, not RL/BC training, and not SSCNet/expert
   integration.
3. Checked `env_isaaclab` Python, pip, torch, Isaac Lab, Isaac Sim, omni, pxr,
   and GPU status.
4. Found Isaac Lab repo:
   `/home/ubuntu22/IsaacLab`
5. Found Isaac Sim pip install:
   `/home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim`
6. Identified official examples:
   `scripts/tutorials/00_sim/create_empty.py` and
   `scripts/tutorials/04_sensors/run_usd_camera.py`.
7. Ran official headless empty scene smoke and observed `Setup complete`.
8. Ran official USD camera/depth smoke with headless rendering environment
   variables and observed `distance_to_image_plane` depth shape
   `torch.Size([2, 480, 640, 1])`.
9. Implemented external simulator smoke directory:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/`
10. Implemented minimal indoor-like Isaac depth scene:
    `/home/ubuntu22/sc_explorer_ws/sim_explorer/minimal_depth_scene.py`
11. Captured 3 fixed camera poses and saved depth, pose, camera, and scene
    metadata under:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke`
12. Implemented measured-only depth-to-voxel ray marching:
    `/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py`
13. Implemented pure Python smoke tests:
    `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_depth_to_voxel.py`
14. Verified observed voxel map shape `(80, 80, 30)` with
    `unknown_count=143335`, `free_count=44435`, `occupied_count=4230`.
15. Updated simulator notes, training notes, and project context.

Completed Stage 4A-2:

1. Re-read simulator notes, SSCNet notes, and project context.
2. Confirmed Stage 4A-1 produced measured-only Isaac depth observations and
   `observed_state_step2.npy`.
3. Confirmed Stage 4A-2 must not run RL, PPO, behavior cloning, imitation
   learning training, SSCNet retraining, Isaac-depth-to-SSCNet inference, NYU
   target label use, ground-truth use, or prediction writes into observed_map.
4. Inspected Stage 4A-1 metadata:
   `observed_summary.json`, `camera_info.json`, `pose_002.json`, and
   `scene_metadata.json`.
5. Implemented simulator paper expert core:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
6. Implemented `EmptyPredictionLayer`, world/grid conversion, frontier
   detection, candidate sampling, observed-map raycasting, paper gains, utility
   scoring, and top-N expert selection.
7. Implemented CLI:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
8. Implemented visualization:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py`
9. Implemented smoke test:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_paper_expert.py`
10. Ran py_compile checks for the new simulator expert files.
11. Ran Stage 4A-2 expert step:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a2_sim_expert_step.log`
12. Generated output directory:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke`
13. Generated:
    `expert_step_decision.npz`, `expert_step_decision.json`,
    `expert_step_candidates.jsonl`, `expert_topdown.png`, and
    `expert_score_bar.png`.
14. Verified one-step stats:
    `frontier_count=5929`, `frontier_adjacent_free_count=5876`,
    `candidates=64`, `top_n=16`, best candidate id `63`, score
    `88.83270299135849`.
15. Ran Stage 4A-2 smoke test:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a2_sim_expert_test.log`
16. Smoke test verified output fields/files, EmptyPredictionLayer gains
    (`gain_sc=0`, `gain_occ=0`, `gain_conf=0`, `gain_hybrid=gain_exp`),
    finite scores, valid expert_action, visualization PNGs, no observed_state
    modification, and no RL/optimizer/policy training.
17. Updated simulator notes, SSCNet notes, and project context.

Completed Stage 4A-3:

1. Re-read simulator notes, SSCNet notes, and project context.
2. Confirmed Stage 4A-2 is complete and Stage 4A-3 is a deterministic
   empty-prediction simulator rollout, not RL, PPO, BC, IL training, SSCNet
   training, or SSCNet inference on Isaac depth.
3. Added a rollout-facing measured-only depth update wrapper:
   `update_observed_state_from_depth(...)` in
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py`.
4. Implemented rollout utility/data helpers:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
5. Implemented Isaac headless rollout runner:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py`
6. Implemented rollout visualization:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
7. Implemented smoke test and real-output validator:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_rollout.py`
8. The rollout loop initializes `observed_state` all UNKNOWN, captures depth at
   each current camera pose, updates `observed_state` only from Isaac depth,
   runs `select_sim_expert_action(...)` with `EmptyPredictionLayer`, converts
   the best candidate to a planar teleport next pose, and repeats.
9. The default motion mode is planar:
   candidate x/y, fixed camera height 1.2m, candidate yaw, roll/pitch level.
10. The optional `voxel3d` mode is available, but was not the default smoke.
11. Saved per-step transition `.npz` files with candidate features, feature
    names, positions, yaws, valid mask, expert action, scores, selected next
    pose, before/after measured coverage counts, frontier/candidate counts,
    modes, done flags, and leakage checks.
12. Saved episode `transitions.jsonl`, `observed_state_final.npy`,
    `episode_summary.json`, and global `manifest.jsonl`.
13. Generated visualizations:
    `rollout_topdown_path.png`, `observed_ratio_curve.png`,
    `frontier_count_curve.png`, `step_topdown_000.png` through
    `step_topdown_009.png`, and `rollout_index.html`.
14. Ran py_compile checks for new/updated Stage 4A-3 files.
15. Ran synthetic rollout smoke validation without launching Isaac.
16. Ran the real Isaac headless rollout:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a3_rollout_empty_pred.log`
17. Real rollout output:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000`
18. Verified rollout result:
    `steps_completed=10`, `done_reason=max_steps`,
    observed_ratio `0.0 -> 0.21754166666666666`,
    final unknown/free/occupied `150232 / 35873 / 5895`,
    average frontier_count `4525.6`, average candidates `64.0`,
    gain_sc min/mean/max `0.0 / 0.0 / 0.0`.
19. Ran Stage 4A-3 smoke test:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a3_rollout_test.log`
20. Smoke test verified synthetic transition serialization, real episode files,
    at least two steps, observed_ratio non-decreasing, valid expert_action,
    `gain_sc=0` for EmptyPredictionLayer, no forbidden target/ground-truth
    fields, prediction did not write observed_map, and no RL/optimizer/BC/IL
    training.
21. Updated simulator notes, SSCNet notes, project context, and decisions.

Completed Stage 4A-3.2:

1. Re-read simulator notes, SSCNet notes, and project context.
2. Confirmed Stage 4A-3 is complete and Stage 4A-3.2 should only build a
   more complex scripted Isaac scene plus depth/observed-map smoke tests.
3. Confirmed no RL, PPO, behavior cloning training, imitation-learning
   training, SSCNet training, SSCNet inference on Isaac depth, PredictionLayer
   connection, prediction writes into observed_map, target labels, or ground
   truth should be used.
4. Implemented scene metadata/spawn factory:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py`
5. Implemented medium-complexity fixed-pose RGB/depth capture:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/medium_complex_depth_scene.py`
6. Implemented medium scene render and observed-map visualization:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/render_medium_complex_scene_views.py`
7. Implemented pure Python metadata/schema test:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_medium_complex_scene_metadata.py`
8. Updated `depth_to_voxel.py` with explicit map bounds CLI arguments and
   `observed_state_final.npy` output.
9. Built a deterministic `three_rooms` scene with bounds `x/y=[-6,6]`,
   `z=[0,3]`, 12m x 12m floor, 2.2m walls, 3 rooms, 1 corridor, 3 openings,
   13 wall segments, 13 cuboid obstacles, and 5 fixed camera poses.
10. Ran py_compile checks for new/updated Stage 4A-3.2 files.

Completed Stage 4A-6.3:

1. Re-read simulator notes, SSCNet training notes, current project context,
   TODO, implementation plan, decisions, Codex log, and ChatGPT summary.
2. Confirmed Stage 4A-6.2 was complete and the current task was alignment
   convention reconciliation, not rollout tuning, RL, PPO, BC, IL, optimizer
   work, SSCNet training, prediction fusion, or prediction writeback.
3. Audited SSCNet code:
   `utils/projection_layer.py`, `dataloaders/dataloader.py`,
   `models/SSCNet.py`, `offline_infer_npz.py`, `src/ssc_network_node.py`,
   `utils/utils.py`, and `voxel_utils/voxel_util.cpp`.
4. Implemented:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/document_sscnet_axis_convention.py`
5. The audit documented two flatten paths: raw Python
   `np.ravel_multi_index((x,y,z),(240,144,240))` and C++/ROS
   `z*(240*144)+y*240+x`. It also documented that `Project2Dto3D` does
   `view(W,H,D)` then `permute(D,H,W)`, and that `target_lr.T` reverses stored
   target axes.
6. Added named alignment conventions and helper functions in
   `isaac_sscnet_preprocess.py`:
   `current_default_v0`, `xz_swap_diagnostic`, and `code_consistent_v1`,
   plus local-index to camera coordinates, camera coordinates to world, world
   to global grid, and convention-specific high-res position flattening.
7. Updated `run_isaac_map_predict_single.py` to accept
   `--alignment_convention`, to save convention metadata, and to align via the
   shared convention helpers.
8. Updated `isaac_map_predictor.py` and `run_sim_expert_rollout_sc_pred.py`
   so dynamic rollout map_predict can use `--alignment_convention
   code_consistent_v1` without reloading the model per step.
9. Updated `visualize_isaac_prediction_alignment.py` to record/display the
   selected convention.
10. Implemented:
    `/home/ubuntu22/sc_explorer_ws/sim_explorer/fix_prediction_alignment_convention.py`
11. Ran convention evaluation on existing local predictions using future
    observed maps only as post-hoc validation. It compared
    `current_default_v0`, `xz_swap_diagnostic`, and `code_consistent_v1`;
    `code_consistent_v1` reproduced the x/z diagnostic improvement with code
    justification.
12. Added deterministic synthetic blob projection output:
    `synthetic_blob_projection.csv`, `synthetic_blob_projection.png`, and
    `synthetic_alignment_test.md`.
13. Reran fixed Stage 4A-5 single-frame map_predict smoke with
    `code_consistent_v1`.
14. Reran fixed Stage 4A-5.1 one-step SC-aware expert smoke using the fixed
    global prediction layer.
15. Reran fixed Stage 4A-6 5-step dynamic SC-aware rollout with
    `code_consistent_v1`.
16. Compared fixed rollout to the empty baseline and to the original SC
    rollout. Fixed alignment improved post-hoc prediction diagnostics but did
    not change the 5-step action sequence or observed_ratio versus original SC.
17. Implemented:
    `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_alignment_convention_fix.py`
18. Ran py_compile and `test_alignment_convention_fix.py`.
19. Verified prediction remained read-only and information-gain-only,
    observed_state hashes were unchanged, future observations were
    evaluation-only, the checkpoint was not modified, and no
    RL/PPO/optimizer/BC/IL/SSCNet training ran.
20. Updated simulator notes, SSCNet notes, TODO, current state, decisions,
    Codex log, ChatGPT summary, and implementation plan.
11. Ran metadata test:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_metadata_test.log`
12. Ran real Isaac headless medium scene capture:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_depth.log`
13. Generated smoke output:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_smoke`
14. Ran measured-only depth_to_voxel with bounds `x/y=[-6,6]`, `z=[0,3]`:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_depth_to_voxel.log`
15. Verified observed map shape `(120, 120, 30)` with unknown/free/occupied
    `339813 / 86064 / 6123` and observed_ratio `0.21339583333333334`.
16. Ran real Isaac headless visualization:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_viz.log`
17. Generated visualization output:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_viz`
18. Verified required images/files exist and are nonblank.
19. Ran optional one-step expert smoke with EmptyPredictionLayer:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_expert_step.log`
20. Optional expert result:
    `frontier_count=20919`, `candidates=64`, best score
    `53.62160777611031`, best grid `[64, 91, 13]`, best world
    `[0.45, 3.15, 1.35]`, `gain_sc=0.0`.
21. Updated simulator notes, SSCNet notes, project context, and decisions.

Completed Stage 4A-3.5:

1. Re-read simulator notes, SSCNet notes, and project context.
2. Confirmed Stage 4A-3.2 is complete and Stage 4A-3.5 should add A*
   observed-free path-cost scoring only, not RL, PPO, BC, IL training,
   SSCNet inference on Isaac depth, PredictionLayer connection, prediction
   writeback, target labels, or ground truth.
3. Implemented observed-free A* planner:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/astar_planner.py`
4. Implemented A* planner tests:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_astar_planner.py`
5. Updated simulator expert:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
   with `path_cost_mode=euclidean|astar`, candidate reachability flags,
   A* path length/expanded-cell features, and invalidation of unreachable
   candidates.
6. Updated one-step CLI:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
   with `--path_cost_mode`.
7. Updated rollout CLI:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py`
   with `--path_cost_mode`, `--scene_variant minimal|medium_three_rooms`,
   `--scene_seed`, map-bound controls, and medium-scene construction through
   `scene_factory`.
8. Updated rollout utilities and visualizations:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py`
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
9. Added simulator A* validator:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_astar.py`
10. Ran py_compile for Stage 4A-3.5 files.
11. Ran one-step medium A* expert:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_medium_expert_step_astar.log`
12. One-step medium A* result:
    traversable/blocked/unknown cells `4316 / 1907 / 8177`,
    reachable/unreachable candidates `12 / 52`, best score
    `51.651363679237036`, best gain_exp `110.0`, gain_sc `0.0`,
    best path_cost `2.129663036258191`, best A* path length
    `1.2656854249492382m`, best grid `[64, 91, 13]`.
13. Ran medium A* rollout:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_medium_rollout_astar_empty_pred.log`
14. Medium A* rollout result:
    episode `medium_three_rooms_astar_empty_pred_000`, `steps_completed=5`,
    `done_reason=no_valid_candidate`, observed_ratio
    `0.0 -> 0.04308796296296296`, final unknown/free/occupied
    `413386 / 15863 / 2751`, average reachable candidates `18.4`, average
    best path_cost `0.9421159585855353`.
15. Recorded the main blocker:
    at expert step 5 all 64 sampled candidates were unreachable under
    conservative observed-free A* traversability
    (`traversable=338`, `blocked=918`, `unknown=13144`). No Euclidean fallback
    was used.
16. Generated one-step visualizations:
    `expert_topdown.png`, `expert_score_bar.png`, and
    `traversability_topdown.png`.
17. Generated rollout visualizations:
    `rollout_topdown_path.png`, `observed_ratio_curve.png`,
    `frontier_count_curve.png`, `reachable_candidates_curve.png`, and
    step topdowns.
18. Ran Stage 4A-3.5 tests:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_astar_planner_test.log`
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_sim_expert_astar_test.log`
19. Ran Euclidean regression tests:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_regression_sim_paper_expert_euclidean_test.log`
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_regression_sim_expert_rollout_test.log`
20. Tests verified A* empty-grid/obstacle/unreachable behavior, medium
    traversability, one-step reachable candidates, rollout >=2 transitions,
    observed_ratio non-decreasing, EmptyPredictionLayer `gain_sc=0`,
    prediction did not write observed_map, no target/ground-truth fields, and
    no RL/optimizer/BC/IL training.
21. Updated simulator notes, SSCNet notes, project context, and decisions.

Completed Stage 4A-3.6:

1. Re-read simulator notes, SSCNet notes, and project context.
2. Confirmed Stage 4A-3.5 is complete and the current task is
   reachability-aware candidate generation for the A* expert.
3. Confirmed Stage 4A-3.6 must not run RL, PPO, behavior-cloning training,
   imitation-learning training, SSCNet training, SSCNet inference on Isaac
   depth, PredictionLayer connection, prediction writeback, target labels, or
   ground truth.
4. Updated the observed-free A* planner:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/astar_planner.py`
5. Added `connected_component_from_start(...)`,
   `nearest_traversable_cell(...)`, and
   `frontier_reachable_candidate_mask(...)`.
6. Updated simulator expert candidate generation:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
7. Added `compute_reachable_frontier_candidate_cells(...)`, reachable
   component diagnostics, candidate source labels, snapped-current support,
   and current/snap start-cell exclusion when other candidates exist.
8. Added `candidate_sampling_mode=frontier|reachable_frontier|auto`; A*
   `auto` resolves to `reachable_frontier`, while Euclidean `auto` preserves
   old frontier sampling.
9. Kept UNKNOWN non-traversable and still ran A* for exact path cost on each
   sampled candidate.
10. Updated one-step CLI:
    `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
    with `--candidate_sampling_mode`, `--snap_start_to_traversable`, and
    `--max_snap_radius_cells`.
11. Updated rollout CLI:
    `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py`
    with the same flags plus reachable rollout diagnostics and summaries.
12. Updated rollout utilities and visualizations:
    `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
    `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py`
    `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
13. Added reachable sampling tests:
    `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_reachable_candidate_sampling.py`
14. Ran py_compile for Stage 4A-3.6 files.
15. Ran one-step medium reachable A* expert:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_medium_expert_step_astar_reachable.log`
16. One-step medium reachable A* result:
    reachable/unreachable candidates `64 / 0`, reachable component count
    `1196`, reachable frontier-adjacent count `1196`, candidate source
    `reachable_frontier`, `top_n=16`, best score `88.24634362636618`,
    best gain_exp `66.0`, gain_sc `0.0`, best path_cost
    `0.7479063413600806`, best A* path length `0.28284271247461906m`, best
    grid `[58, 82, 11]`.
17. Ran medium reachable A* rollout:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_medium_rollout_astar_reachable_empty_pred.log`
18. Medium reachable A* rollout result:
    episode `medium_three_rooms_astar_reachable_empty_pred_000`,
    `steps_completed=10`, `done_reason=max_steps`, observed_ratio
    `0.0 -> 0.10147453703703704`, final unknown/free/occupied
    `388163 / 36017 / 7820`, average reachable candidates `64.0`, average
    reachable component count `238.8`, average reachable frontier-adjacent
    count `238.8`, `no_valid_candidate_steps=[]`.
19. Generated one-step visualizations:
    `expert_topdown.png`, `expert_score_bar.png`, and
    `traversability_topdown.png`.
20. Generated rollout visualizations:
    `rollout_topdown_path.png`, `observed_ratio_curve.png`,
    `frontier_count_curve.png`, `reachable_candidates_curve.png`,
    `reachable_component_count_curve.png`, step topdowns, and
    `rollout_index.html`.
21. Ran Stage 4A-3.6 tests:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_reachable_candidate_sampling_test.log`
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_astar_planner_regression_test.log`
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_sim_expert_astar_reachable_test.log`
22. Ran Euclidean regression smoke checks:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_regression_sim_paper_expert_euclidean_test.log`
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_regression_sim_expert_rollout_test.log`
23. Tests verified connected component behavior, snap-to-traversable, reachable
    candidate sampling, medium traversability, one-step reachable candidates,
    rollout `max_steps`, observed_ratio non-decreasing, EmptyPredictionLayer
    `gain_sc=0`, no prediction writeback, no target/ground-truth fields, and
    no RL/optimizer/BC/IL training.
24. Updated simulator notes, SSCNet notes, project context, and decisions.

Next plan:

Completed Stage 4A-4:

1. Re-read simulator notes, SSCNet notes, and project context.
2. Confirmed Stage 4A-3.6 is complete and Stage 4A-4 is only a
   multi-episode EmptyPredictionLayer A* expert rollout dataset stage.
3. Confirmed no RL, PPO, behavior-cloning training, imitation-learning
   training, SSCNet training, SSCNet inference on Isaac depth, map_predict
   connection, PredictionLayer connection, prediction writeback, target labels,
   ground-truth scoring, UNKNOWN traversability shortcut, or Euclidean fallback
   should be used.
4. Added batch runner:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_batch.py`
5. Added dataset summarizer:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_rollout_dataset.py`
6. Added batch dataset validator:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_rollout_dataset_batch.py`
7. Updated rollout runner:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py`
   with `--start_variant`, `--obstacle_jitter_m`, start-pose metadata, and
   optional `--no_manifest` for batch-controlled manifests.
8. Updated rollout utilities:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
   with explicit selected expert fields: `gain_exp`, `gain_sc`,
   `gain_hybrid`, `path_cost`, and `final_score`.
9. Ran py_compile for updated/new Stage 4A-4 files.
10. Ran lightweight regressions:
    `test_reachable_candidate_sampling.py`, `test_astar_planner.py`, and
    `test_sim_expert_rollout.py`.
11. Ran the 9-episode headless Isaac batch:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_batch_empty_pred_astar.log`
12. Batch setup:
    scene_variant `medium_three_rooms`, seeds `0,1,2`, starts
    `start_room_a,start_corridor,start_room_b`, max_steps `10`,
    num_candidates `64`, top_n `16`, gain_mode `hybrid`, prediction_mode
    `empty`, path_cost_mode `astar`, candidate_sampling_mode
    `reachable_frontier`.
13. Batch result:
    output root
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar`,
    ok episodes `9`, failed episodes `0`, total transitions `90`, all
    episodes ended by `max_steps`.
14. Ran dataset summarizer:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_dataset_summary.log`
15. Summary result:
    steps min/mean/max `10 / 10 / 10`, observed_ratio_end min/mean/max
    `0.08587037037037037 / 0.11455118312757204 / 0.16534722222222223`,
    average reachable candidates `64.0`, average gain_sc `0.0`,
    no_valid_candidate episodes `0`.
16. Generated:
    `manifest.jsonl`, `dataset_summary.json`, `dataset_summary.md`,
    `rollout_dataset_index.html`, aggregate observed-ratio/reachable/done/step
    plots, and per-episode rollout outputs.
17. Ran batch dataset validator:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_dataset_test.log`
18. Validator confirmed at least 6 successful episodes, observed_ratio
    non-decreasing, EmptyPredictionLayer `gain_sc=0`, prediction did not write
    observed_map, no forbidden target/ground-truth fields, no RL/optimizer/BC/IL
    training, no UNKNOWN traversability shortcut, and no Euclidean fallback.
19. Updated simulator notes, SSCNet notes, project context, and decisions.

Next plan:

Completed Stage 4A-5:

1. Re-read simulator notes, SSCNet notes, and project context.
2. Confirmed Stage 4A-4 is complete and Stage 4A-5 is only a single-frame
   Isaac map_predict preprocessing, inference, and shape-alignment smoke test.
3. Confirmed no RL, PPO, behavior-cloning training, imitation-learning
   training, SSCNet training, checkpoint modification, target labels,
   ground-truth scoring, full rollout, expert decision with prediction,
   prediction writeback, or prediction-based traversability/collision/A*
   should be used.
4. Inspected `projection_layer.py` and `dataloaders/dataloader.py`.
5. Checked real NYU `position` samples and wrote:
   `/home/ubuntu22/sc_explorer_ws/logs/stage4a5_position_convention_check.log`
6. Implemented Isaac depth preprocessing:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_sscnet_preprocess.py`
7. Implemented simulator-native read-only prediction layer:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_prediction_layer.py`
8. Implemented single-frame runner:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_map_predict_single.py`
9. Implemented visualization:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_isaac_prediction_alignment.py`
10. Implemented smoke test:
    `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_map_predict_single.py`
11. Selected the first ok Stage 4A-4 episode from `manifest.jsonl`:
    `medium_three_rooms_seed0_start_room_a_empty_astar`, step `0`.
12. Preprocessed `depth_000.npy` to SSCNet depth `(480,640)`, position
    `(480,640)`, and valid position mask with `166888` valid pixels.
13. Loaded the best checkpoint with strict state dict loading:
    `/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar`
14. Ran one SSCNet inference:
    logits `(1,12,60,36,60)`, local prediction `(60,36,60)`.
15. Aligned local `(z_forward,y_up,x_right)` prediction to global
    `(world_x,world_y,world_z)` observed map shape `(120,120,30)`.
16. Wrote read-only global arrays:
    `global_pred_class`, `global_confidence`, `global_free_prob`,
    `global_occupied_prob`, and `global_prediction_valid`.
17. Saved output under:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke`
18. Generated:
    `sscnet_input_debug.npz`, `local_prediction.npz`,
    `global_prediction_layer.npz`, `prediction_alignment_summary.json`,
    `isaac_depth_input.png`, `local_prediction_slices.png`,
    `global_prediction_topdown.png`, `observed_vs_prediction_topdown.png`,
    and `prediction_not_measured_topdown.png`.
19. Run result:
    global valid prediction voxels `56602`, predicted occupied voxels `15664`,
    predicted_unmeasured voxels `39400`, observed_state hash unchanged.
20. Ran py_compile for all new Stage 4A-5 files.
21. Ran Stage 4A-5 smoke test:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a5_isaac_map_predict_single_test.log`
22. Smoke test verified preprocessing shapes, position range, logits shape,
    probability ranges, global shape alignment, SimPredictionLayer API,
    predicted_unmeasured count, no target/ground-truth artifact fields,
    no observed_state modification, no prediction writeback, no
    traversability/collision/A* prediction use, no RL/optimizer/BC/IL training,
    and no expert/rollout prediction use.
23. Updated simulator notes, SSCNet notes, project context, and decisions.

Next plan:

Completed Stage 4A-5.1:

1. Re-read simulator notes, SSCNet notes, and project context.
2. Confirmed Stage 4A-5 is complete and Stage 4A-5.1 should only connect the
   read-only `SimPredictionLayer` to one-step simulator expert scoring.
3. Confirmed no rollout, RL, PPO, behavior-cloning training,
   imitation-learning training, SSCNet training, checkpoint modification,
   target labels, ground-truth scoring, prediction writeback, or
   prediction-based traversability/collision/A* should be used.
4. Updated `sim_paper_expert.py` to accept a prediction-layer-compatible
   object for `prediction_mode=sim_npz`.
5. Kept `prediction_mode=empty` as the default and forced it to use
   `EmptyPredictionLayer`.
6. Preserved measured-only candidate generation, measured-only conservative
   raycast, and observed-free A* traversability/path-cost logic.
7. Added prediction diagnostics proving prediction is used only for
   information gain and not for candidate sampling, traversability, collision,
   A*, ray blocking, or observed_state writeback.
8. Updated `run_sim_expert_step.py` with:
   `--prediction_mode empty|sim_npz`, `--prediction_npz`, `--tau`, and
   `--episode_summary`.
9. Added observed_state SHA-256 before/after diagnostics to the one-step
   runner.
10. Updated `visualize_sim_expert_step.py` with prediction-valid,
    predicted-unmeasured, predicted-occupied, and predicted-unmeasured-visible
    overlays.
11. Added `test_sim_expert_with_prediction.py`.
12. Ran py_compile for:
    `sim_prediction_layer.py`, `sim_paper_expert.py`,
    `run_sim_expert_step.py`, `visualize_sim_expert_step.py`, and
    `test_sim_expert_with_prediction.py`.
13. Ran empty baseline one-step expert:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_empty_baseline.log`
14. Ran SC prediction one-step expert:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_sc_prediction.log`
15. Ran Stage 4A-5.1 smoke test:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_with_prediction_test.log`
16. Saved comparison outputs under:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke`
17. Verified:
    empty `gain_sc=0`, prediction `64/64` candidates with `gain_sc>0`,
    `gain_hybrid=gain_exp+gain_sc`, finite `gain_occ/gain_conf`,
    observed_state hash unchanged, no target/ground-truth leakage, no
    RL/optimizer/BC/IL training, and no rollout.

Next plan:

1. Stage 4A-6: run a short multi-step rollout with read-only map_predict.
2. Either recompute prediction at each step or explicitly document a
   static-prediction ablation.
3. Compare against the Stage 4A-4 measured-only baseline.
4. Keep prediction separate from observed_map and out of traversability,
   collision, A* validity, and ray blocking unless a later staged ablation says
   otherwise.

Completed Stage 4A-6:

1. Re-read simulator notes, SSCNet notes, and project context.
2. Confirmed Stage 4A-5.1 is complete and Stage 4A-6 should run only a short
   multi-step SC-aware rollout, not RL, PPO, behavior cloning,
   imitation-learning training, SSCNet training, checkpoint modification,
   target labels, or ground-truth scoring.
3. Added `isaac_map_predictor.py` to load the SSCNet checkpoint once and reuse
   it for per-step depth preprocessing, inference, global alignment, and
   `SimPredictionLayer` construction.
4. Added `run_sim_expert_rollout_sc_pred.py` for the dynamic Isaac headless
   rollout loop:
   depth capture -> measured-only observed_state update -> read-only
   map_predict -> SC-aware expert scoring -> planar teleport.
5. Kept prediction out of observed_state writes, traversability, collision
   checking, A*, candidate reachability, and ray blocking.
6. Added `compare_sc_pred_rollout.py` to compare the 5-step SC rollout against
   the matching Stage 4A-4 empty baseline.
7. Added `test_sim_sc_aware_rollout.py` for output, safety, leakage, and
   comparison checks.
8. Updated `visualize_sim_rollout.py` with gain, best_score, map_predict
   timing, prediction-valid, and predicted-unmeasured curves plus per-step
   prediction overlays.
9. Extended transition serialization in `sim_rollout_utils.py` with SC gain,
   prediction count, timing, hash, and safety fields while preserving empty
   rollout defaults.
10. Fixed an initial Stage 4A-6 visualization failure by adding
    `observed_state_source` and related source fields to the per-step
    prediction summary.
11. Ran the 5-step dynamic rollout:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a6_sc_pred_dynamic_rollout.log`
12. Rollout result:
    steps_completed `5`, done_reason `max_steps`, observed_ratio
    `0.0 -> 0.05899768518518519`, final unknown/free/occupied
    `406513 / 21226 / 4261`, model_loaded_once `true`.
13. map_predict performance:
    average inference `0.020522295199771178s`, average total prediction
    `0.14326694260016665s`, average expert `1.026360238399866s`,
    GPU peak `794354176` bytes on RTX 5080.
14. SC gain result:
    average gain_exp `49.6`, gain_sc `49.4`, gain_hybrid `99.0`,
    gain_occ `8.8`, gain_conf `16.96283725500107`, candidates with
    gain_sc > 0 min/mean/max `63 / 63.6 / 64`.
15. Ran comparison against:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar`
16. Comparison result:
    compared_steps `5`, empty final observed_ratio `0.06896296296296296`,
    SC final observed_ratio `0.05899768518518519`, SC-empty delta
    `-0.009965277777777774`, changed selected actions `5`.
17. Ran py_compile for the Stage 4A-6 files and dependent simulator modules.
18. Ran `test_sim_sc_aware_rollout.py`:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a6_sc_aware_rollout_test.log`
19. Test verified observed_ratio non-decreasing, gain_sc nonzero,
    gain_hybrid identity, observed_state hash unchanged by prediction,
    prediction not used for traversability/collision/A*/ray blocking,
    prediction did not write back, checkpoint was not modified, comparison
    exists, and no RL/optimizer/BC/IL/SSCNet training ran.

Next plan:

1. Stage 4A-6.1: analyze/ablate/tune before longer rollout:
   static-prediction ablation, gain_sc weighting sweep, tau sweep, optional
   prediction-gain cap, and prediction-overlay/action inspection.
2. After tuning, run a 10-step SC-aware rollout on the same seed/start and
   compare against the measured-only Stage 4A-4 baseline.
3. Stage 4A-7: add simple prediction fusion across steps only after
   Stage 4A-6.1 behavior is understood.
4. Do not jump to RL, PPO, behavior cloning, or imitation-learning training.

Completed Stage 4A-6.1:

1. Re-read simulator notes, SSCNet notes, and project context.
2. Confirmed Stage 4A-6 is complete and Stage 4A-6.1 is analysis/ablation
   only, not RL, PPO, behavior cloning, imitation-learning training, SSCNet
   training, checkpoint modification, target use, or ground-truth scoring.
3. Added `analyze_sc_rollout_behavior.py` to compare the existing Stage 4A-6
   SC rollout against the Stage 4A-4 measured-only baseline step by step.
4. Extended `sim_paper_expert.py` with optional `sc_gain_weight`,
   `sc_gain_cap`, and `score_gain_mode` while preserving raw paper gains and
   default Stage 4A-6-compatible behavior.
5. Updated `run_sim_expert_step.py` and `run_sim_expert_rollout_sc_pred.py`
   with the weighted/capped scoring CLI.
6. Updated `sim_rollout_utils.py` and `visualize_sim_rollout.py` to serialize
   and visualize weighted gain fields.
7. Added `run_sc_pred_ablation_sweep.py` to run a small sequential Isaac
   ablation sweep without launching parallel Isaac instances.
8. Added `summarize_sc_pred_ablation.py` to aggregate ablation results,
   compare to empty/original SC, write plots/tables, and generate qualitative
   inspection sheets.
9. Added `test_sc_pred_ablation.py` to validate output structure, at least two
   completed configs, observed_ratio monotonicity, safety flags, checkpoint
   status, no forbidden target arrays, no training flags, and weighted gain
   formula.
10. Ran existing SC-vs-empty analysis:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a61_existing_sc_analysis.log`
11. Existing analysis result:
    empty final observed_ratio `0.06896296296296296`, original SC final
    observed_ratio `0.05899768518518519`, delta
    `-0.009965277777777774`, first SC lag step `1`, changed actions `5/5`,
    mean SC gain_sc `49.4`, mean SC path_cost `0.2768163156997422`.
12. Ran five 5-step ablations:
    `dynamic_w025_tau01`, `dynamic_w05_tau01`, `dynamic_w1_tau03`,
    `dynamic_w1_tau01_cap50`, and `static_step0_weight_1p0_tau_0p1`.
13. Ablation result:
    all five completed, failed configs `[]`, all final observed_ratio
    `0.05899768518518519`, all delta vs empty `-0.009965277777777774`, and
    all selected the same `5/5` actions as the original SC rollout.
14. Ran ablation summary:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a61_ablation_summary.log`
15. Summary outputs:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_ablation/summary`
    and qualitative sheets under
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/qualitative_inspection`.
16. Ran py_compile and `test_sc_pred_ablation.py`:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a61_ablation_test.log`
17. Validation passed:
    observed_ratio non-decreasing, weighted gain formula correct, prediction
    read-only, no prediction traversability/collision/A*/ray blocking, no
    prediction writeback, checkpoint not modified, no target/ground-truth
    leakage, and no RL/optimizer/BC/IL/SSCNet training.

Next plan:

1. Stage 4A-6.3 should fix/reconcile map_predict alignment convention and
   rerun Stage 4A-5/5.1/6 smoke.
2. If alignment is fixed but dense calibration remains, implement calibrated
   or capped confidence-based I_sc and rerun one-step plus 5-step smoke.
3. If domain shift dominates after alignment/calibration checks, collect
   Isaac-domain validation/synthetic supervised data before relying on
   SC-aware rollout.
4. Still do not jump to RL, PPO, behavior cloning, or imitation-learning
   training.

Completed Stage 4A-6.2:

1. Re-read simulator notes, SSCNet notes, and project context.
2. Confirmed Stage 4A-6 and Stage 4A-6.1 are complete and the current task is
   map_predict diagnostics, not rollout tuning, not RL/IL, and not training.
3. Implemented Isaac-vs-NYU preprocessing diagnostics:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_isaac_sscnet_preprocess.py`
4. Implemented global alignment/frustum diagnostics:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_prediction_global_alignment.py`
5. Implemented future observed sensor validation:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/evaluate_prediction_against_future_observed.py`
   Future observations are post-hoc evaluation only, not planning or expert
   scoring.
6. Implemented diagnostic alignment variant sweep without rerunning SSCNet:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_alignment_variant_sweep.py`
7. Implemented diagnostic summary and candidate-score decomposition:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_map_predict_diagnostics.py`
8. Implemented diagnostics validator:
   `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_map_predict_diagnostics.py`
9. Ran preprocessing diagnostics:
   `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_preprocess_stats.log`
   Output:
   `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/preprocess_stats`
10. Ran global alignment diagnostics:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_global_alignment.log`
    Output:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/global_alignment`
11. Ran future observed evaluation:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_future_observed_eval.log`
    Output:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/future_observed_eval`
12. Ran alignment variant sweep:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_alignment_variant_sweep.log`
    Output:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/alignment_variant_sweep`
13. Ran summary:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_diagnostic_summary.log`
    Output:
    `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/summary`
14. Ran py_compile and diagnostics validation:
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_py_compile.log`
    `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_diagnostics_test.log`
15. Key result:
    direct frustum sanity mostly passes, but alignment variant sweep ranks
    `xz_swap_variant` best with default rank `7` and Brier improvement
    `0.0735458940774611`, making alignment convention the primary suspect.
16. Secondary result:
    tau `0.1` prediction is dense/unselective with mean predicted_unmeasured
    `35118.2`, later measured fraction `0.059004217437215026`, occupied Brier
    `0.2786559495144023`, and ECE-like calibration `0.3405436085907938`.
17. Candidate scoring result:
    gain_exp/gain_sc correlation `0.9647202023737985` and final_score vs
    inverse path_cost correlation `0.9713818732156227`; gain_sc duplicates
    gain_exp and low path cost dominates logged top-N ranking.
18. Validation passed:
    observed_state hashes unchanged, checkpoint not modified, prediction
    read-only, no traversability/collision/A*/ray blocking/candidate
    reachability use, future observations evaluation-only, and no
    RL/optimizer/BC/IL/SSCNet training.
