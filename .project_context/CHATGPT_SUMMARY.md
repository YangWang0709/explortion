Continuation Summary
Updated: 2026-05-30

SSCNet training is complete and the best checkpoint is:

```text
/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar
```

Stage 2A is complete. The repository now has a reusable offline inference tool and read-only prediction wrapper:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/offline_infer_npz.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/prediction_layer.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_offline_prediction_layer.py
```

Single-sample inference output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/sscnet_inference/NYU0670_0000_voxels_prediction.npz
logits shape: (1, 12, 60, 36, 60)
pred_class shape: (60, 36, 60)
confidence mean: 0.793621
occupied_prob mean: 0.276600
free_prob mean: 0.723400
```

PredictionLayer smoke test passed and validated shapes, class range, probability ranges, and confidence threshold behavior.

Stage 2B strict paper-faithful expert scoring is complete. Files:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/sc_explorer_paper_expert.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/run_paper_expert_offline.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_paper_expert.py
```

Strict Stage 2B rules:

```text
target_lr/target_hr are not used for expert scoring.
Measured S is approximated only from sensor-derived tsdf_lr and/or position.
Default measured_mode is tsdf_lr.
P = PredictionLayer.confidence >= tau and not measured S.
Default raycast mode is non_blocking, so SC prediction does not block rays.
Default gain_mode is hybrid.
Cost is approximate position/yaw time; full RRT tree utility Eq. 12 is future work.
```

Single sample result:

```text
sample: NYU0670_0000_voxels
candidates: 64
best score: 872.624436
best gain_exp: 473.0
best gain_sc: 473.0
best gain_hybrid: 946.0
best gain_occ: 352.0
best gain_conf: 137.822484
best path_cost: 1.084086
expert_action: 0
output: /home/ubuntu22/sc_explorer_ws/outputs/paper_expert/paper_expert_decision_NYU0670_0000_voxels.npz
jsonl: /home/ubuntu22/sc_explorer_ws/outputs/paper_expert/paper_expert_decisions.jsonl
```

Logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage2b_paper_expert_test.log
/home/ubuntu22/sc_explorer_ws/logs/stage2b_paper_expert_single.log
/home/ubuntu22/sc_explorer_ws/logs/stage2b_paper_expert_batch5.log
```

Stage 2B handoff:

```text
Stage 2C: convert strict paper expert outputs into imitation-learning-ready dataset format, still without training IL.
```

Continue to avoid RL/PPO/imitation learning training, planner integration, Unreal/AirSim, retraining, robot execution, and observed_map writes unless explicitly requested.

Stage 2C paper expert dataset generation is complete. New files:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/generate_paper_expert_dataset.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_paper_expert_dataset.py
```

Smoke dataset output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/samples
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/manifest.jsonl
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/metadata.json
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/combined_smoke.npz
```

Stage 2C smoke result:

```text
total samples: 5
ok: 5
failed: 0
sample npz count: 5
candidate_features shape in first sample: (16, 15)
forbidden target fields check: passed
```

Logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage2c_dataset_smoke.log
/home/ubuntu22/sc_explorer_ws/logs/stage2c_dataset_test.log
```

Stage 2C saved format:

```text
candidate_features float32 [N, D]
feature_names string [D]
candidate_positions int32 [N, 3]
candidate_yaws float32 [N]
valid_mask bool [N]
expert_action int64 scalar
expert_scores float32 [N]
top_candidate_ids int64 [N]
gain_mode / measured_mode / raycast_mode
sample_npz / prediction_npz / sample_id
strict_no_target_note / tree_limitation_note
```

For the smoke dataset, `N=16`, `D=15`, `num_candidates=64`,
`top_n=16`, `tau=0.1`, `measured_mode=tsdf_lr`,
`raycast_mode=non_blocking`, and `gain_mode=hybrid`.

Stage 2C still uses the offline NYU measured-mask approximation. It has no
online measured map and no full RRT tree Eq. 12 implementation. No imitation
learning training was run.

Next recommended stage:

```text
Stage 3A: create an imitation-learning Dataset/DataLoader and behavior cloning
training script, but first run only a data-loading smoke test.
```

Stage 3A IL Dataset/DataLoader smoke is complete. New files:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/__init__.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/paper_expert_dataset.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/policy.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/train_bc.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/test_dataset.py
```

Stage 3A APIs:

```text
PaperExpertDataset
collate_paper_expert_batch
compute_feature_stats
save_feature_stats
CandidateMLPPolicy
```

Dataset rules:

```text
Only Stage 2C expert sample npz files are loaded.
Original sample_npz and prediction_npz are not opened.
metadata strict_no_target_lr must be true by default.
forbidden fields target_lr, target_hr, gt, and ground_truth are rejected.
expert_action must be valid and finite feature/score tensors are required.
```

Dataset smoke result:

```text
log: /home/ubuntu22/sc_explorer_ws/logs/stage3a_il_dataset_test.log
dataset size: 5
first candidate_features shape: (16, 15)
batch candidate_features shape: (2, 16, 15)
feature_names length: 15
feature_stats mean shape: (15,)
feature_stats std min: 1.053888
logits shape: (2, 16)
cross_entropy loss: 0.000060
optimizer step performed: no
forbidden target fields: none
```

BC dry-run result:

```text
log: /home/ubuntu22/sc_explorer_ws/logs/stage3a_bc_dry_run.log
dataset size: 5
B,N,D: 2,16,15
expert_action: [0, 0]
logits shape: (2, 16)
loss: 0.165347
optimizer step performed: no
model saved: no
```

Feature stats were saved to:

```text
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/feature_stats.npz
```

No imitation-learning training was run in Stage 3A. No optimizer step was
performed. No model was saved.

Stage 4A-1 Isaac depth observation smoke is complete. New files:

```text
/home/ubuntu22/sc_explorer_ws/simulator_notes.md
/home/ubuntu22/sc_explorer_ws/sim_explorer/README.md
/home/ubuntu22/sc_explorer_ws/sim_explorer/minimal_depth_scene.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_depth_to_voxel.py
```

Isaac status:

```text
env: env_isaaclab
Isaac Lab path: /home/ubuntu22/IsaacLab
Isaac Lab commit: 090aed18163b2194d5551c7919f7539283677743
Isaac Lab package: 0.54.3
Isaac Sim path: /home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim
Isaac Sim package metadata: 5.1.0.0
Isaac Sim VERSION: 5.1.0-rc.19+release.26219.9c81211b.gl
```

Official smoke tests:

```text
empty scene: passed, [INFO]: Setup complete...
camera/depth: passed with distance_to_image_plane
official depth shape: torch.Size([2, 480, 640, 1])
```

Working headless camera environment:

```text
export TERM=xterm
export PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
unset DISPLAY WAYLAND_DISPLAY XAUTHORITY GNOME_SETUP_DISPLAY
```

Minimal Stage 4A-1 output:

```text
output dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke
depth_000.npy: shape (120, 160), dtype float32, min 1.3499999, max 3.9250004
depth_001.npy: shape (120, 160), dtype float32, min 1.6134452, max 3.9250004
depth_002.npy: shape (120, 160), dtype float32, min 0.3499999, max 2.9250004
```

Observed voxel map:

```text
voxel size: 0.1m
map bounds: x [-4, 4], y [-4, 4], z [0, 3]
observed map shape: (80, 80, 30)
unknown_count: 143335
free_count: 44435
occupied_count: 4230
observed_ratio: 0.2534635416666667
```

Logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a1_env_status.log
/home/ubuntu22/sc_explorer_ws/logs/isaac_empty_scene_smoke.log
/home/ubuntu22/sc_explorer_ws/logs/isaac_sensor_smoke.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a1_minimal_depth_scene.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a1_depth_to_voxel.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a1_depth_to_voxel_test.log
```

Stage 4A-1 decision:

```text
Stage 4A-1 only validates simulator depth observation and observed_map update.
Prediction and expert integration are Stage 4A-2.
```

Continue to avoid RL/PPO, behavior-cloning training, imitation-learning
training, SSCNet retraining, AirSim/Unreal, target label use, ground-truth map
use, and prediction writes into observed_map.

Next recommended stage:

```text
Stage 4A-2: connect one simulator observed voxel map step to PredictionLayer
and the paper expert scorer, while keeping prediction separate from observed_map.
```

Stage 4A-1-viz visualization is complete. New file:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_observed_map.py
```

Visualization output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke_viz
```

Generated files:

```text
depth_000.png
depth_001.png
depth_002.png
depth_grid.png
observed_topdown_step0.png
observed_topdown_step1.png
observed_topdown_step2.png
observed_topdown_compare.png
occupied_voxels_3d_step2.png
free_occupied_voxels_3d_step2.png
slices_step2.png
index.html
viz_summary.json
```

Observed map visualization counts:

```text
shape: (80, 80, 30)
step0: unknown=170910, free=19335, occupied=1755, observed_ratio=0.10984375
step1: unknown=143439, free=44515, occupied=4046, observed_ratio=0.252921875
step2: unknown=143335, free=44435, occupied=4230, observed_ratio=0.2534635416666667
```

Open3D PLY export:

```text
skipped because open3d is not installed
```

Log:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a1_visualize_observed_map.log
```

Stage 4A-1-viz remained read-only with respect to observed_state and did not
run RL/PPO, behavior-cloning training, imitation-learning training, SSCNet
inference, PredictionLayer, or expert scoring.

Stage 4A-1-scene-viz scripted Isaac scene visualization is complete. New file:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/render_minimal_scene_views.py
```

Scene visualization output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_scene_viz
```

Generated files:

```text
camera_rgb_000.png
camera_rgb_001.png
camera_rgb_002.png
camera_depth_color_000.png
camera_depth_color_001.png
camera_depth_color_002.png
scene_overview_rgb.png
scene_overview_depth_color.png
scene_layout_topdown.png
scene_metadata.json
scene_viz_summary.json
```

Camera data:

```text
RGB key used: rgb
Depth key used: distance_to_image_plane
Main camera RGB: 160 x 120, true RGB PNG
Overview RGB: 640 x 480, true RGB PNG
Main camera max depth: 5m
Overview camera max depth: 12m
```

Scene:

```text
floor: 8m x 8m
walls: side walls at y=-4 and y=4, back wall at x=4, height 2m
obstacles: center cuboid and back cuboid
camera poses: [0,0,1.2] yaw 0 deg; [0,0,1.2] yaw 90 deg; [1,0,1.2] yaw 0 deg
overview pose: [0,-6.5,5] looking at [1.5,0,0.7]
```

Validation:

```text
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a1_scene_viz.log
required files: present
summary JSON: no NaN
RGB images: non-empty and non-uniform
depth color images: include colorbar and min/max depth title
```

Stage 4A-1-scene-viz remained read-only with respect to observed_map and
observed_state and did not run RL/PPO, behavior-cloning training,
imitation-learning training, SSCNet inference, PredictionLayer, or expert
scoring.

Stage 4A-2 simulator observed-map expert step is complete. New files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py
```

Stage 4A-2 validates only this one-step path:

```text
Isaac observed_state_step2.npy -> frontier/candidate generation
  -> observed-map raycast visibility -> paper gain scoring
  -> best next viewpoint -> top-N output and visualization
```

Input:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step2.npy
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_summary.json
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/camera_info.json
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/pose_002.json
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/scene_metadata.json
```

Stats:

```text
shape: (80, 80, 30)
unknown_count: 143335
free_count: 44435
occupied_count: 4230
observed_ratio: 0.2534635416666667
frontier_count: 5929
frontier_adjacent_free_count: 5876
candidates: 64
top_n: 16
prediction_mode: empty
gain_mode: hybrid
raycast_mode: observed_conservative_unknown_blocking
```

Best candidate:

```text
expert_action: 0
candidate id: 63
best score: 88.83270299135849
gain_exp: 73.0
gain_sc: 0.0
gain_hybrid: 73.0
path_cost: 0.8217694333482268
grid position: [51, 38, 14]
world position: [1.15, -0.15, 1.45]
yaw: -0.7030942394487684 rad
```

Output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke/expert_step_decision.npz
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke/expert_step_decision.json
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke/expert_step_candidates.jsonl
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke/expert_topdown.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke/expert_score_bar.png
```

Logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a2_sim_expert_step.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a2_sim_expert_test.log
```

Smoke test passed:

```text
observed_state_modified: no
gain_sc/gain_occ/gain_conf: 0 under EmptyPredictionLayer
gain_hybrid == gain_exp under EmptyPredictionLayer
rl_or_optimizer_or_policy_training_run: no
```

Stage 4A-2 did not run SSCNet on Isaac depth, did not connect real
PredictionLayer, did not use NYU target labels or ground truth, did not modify
`observed_state_step*.npy`, and did not run RL/PPO/BC/IL training.

Next recommended stage:

```text
Stage 4A-3: turn the one-step expert into a loop:
move camera to best candidate, capture new depth, update observed_map,
run expert again.
Keep prediction empty until Isaac-depth-to-SSCNet preprocessing is solved.
```

Stage 4A-3 empty-prediction expert rollout is complete. New / updated files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py
```

Stage 4A-3 implements this deterministic simulator loop:

```text
current camera pose -> Isaac RGB/depth -> measured-only observed_map update
  -> EmptyPredictionLayer expert scorer -> best candidate
  -> planar teleport camera motion -> repeat
```

Rollout output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000
```

Run/test logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a3_rollout_empty_pred.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a3_rollout_test.log
```

Rollout result:

```text
episode_id: minimal_room_empty_pred_000
steps_completed: 10
done_reason: max_steps
observed_ratio_start: 0.0
observed_ratio_end: 0.21754166666666666
total_delta_observed_ratio: 0.21754166666666666
final unknown/free/occupied: 150232 / 35873 / 5895
final pose: [3.549999952316284, 3.25, 1.2000000476837158]
repeated_pose_count: 2
average frontier_count: 4525.6
average candidates: 64.0
gain_sc min/mean/max: 0.0 / 0.0 / 0.0
```

Generated artifacts include:

```text
step_000.npz ... step_009.npz
transitions.jsonl
observed_state_step000.npy ... observed_state_step009.npy
observed_state_final.npy
episode_summary.json
rollout_topdown_path.png
observed_ratio_curve.png
frontier_count_curve.png
step_topdown_000.png ... step_topdown_009.png
rollout_index.html
manifest.jsonl
```

Smoke test passed:

```text
synthetic_transition_serialization: ok
real_episode_validation: ok, 10 steps
observed_ratio_non_decreasing: yes
gain_sc_empty_prediction: zero
prediction_writes_observed_map: no
rl_optimizer_bc_training_run: no
```

Stage 4A-3 boundaries:

```text
EmptyPredictionLayer only.
No SSCNet on Isaac depth.
No PredictionLayer / map_predict connection.
No prediction writes into observed_map.
No NYU target_lr/target_hr.
No scene or simulator ground truth.
No RL/PPO/BC/IL training or optimizer step.
Motion is planar teleport camera movement, not physical robot execution.
No A* planner and no full SC-Explorer RRT tree planner yet.
```

Next recommended stage:

```text
Stage 4A-4: generate multiple rollout episodes with randomized scripted rooms
or randomized start poses, still using EmptyPredictionLayer, to create a real
sequential expert dataset.

Alternative Stage 4A-3.5: add A* over observed FREE before multi-episode
generation.
```

Stage 4A-3.2 medium-complexity scripted scene is complete. New / updated files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/medium_complex_depth_scene.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/render_medium_complex_scene_views.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_medium_complex_scene_metadata.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py
```

Stage 4A-3.2 scene:

```text
scene_id: medium_complex_three_rooms
variant: three_rooms
seed: 0
bounds: x/y [-6,6], z [0,3]
floor: 12m x 12m
wall height: 2.2m
rooms: 3
corridors: 1
openings: 3
walls: 13
obstacles: 13
camera poses: 5
camera height: 1.2m
main camera: 160 x 120, max depth 8m
overview camera: 640 x 480
```

Smoke output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_smoke
depth_000.npy ... depth_004.npy
rgb_000.png ... rgb_004.png
pose_000.json ... pose_004.json
camera_info.json
scene_metadata.json
observed_state_step0.npy ... observed_state_step4.npy
observed_state_final.npy
observed_summary.json
```

Observed map:

```text
shape: (120, 120, 30)
unknown/free/occupied: 339813 / 86064 / 6123
observed_ratio: 0.21339583333333334
```

Visualization output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_viz
scene_overview_rgb.png
scene_overview_depth_color.png
scene_layout_topdown.png
camera_rgb_grid.png
camera_depth_grid.png
observed_topdown_compare.png
free_occupied_voxels_3d_final.png
slices_final.png
scene_viz_summary.json
```

Optional one-step expert smoke:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_smoke
prediction_mode: empty
gain_mode: hybrid
frontier_count: 20919
candidates: 64
best score: 53.62160777611031
best grid: [64, 91, 13]
best world: [0.45, 3.15, 1.35]
gain_sc: 0.0
```

Logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_metadata_test.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_depth.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_depth_to_voxel.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_viz.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_expert_step.log
```

Stage 4A-3.2 boundaries:

```text
No RL/PPO.
No behavior cloning or imitation-learning training.
No SSCNet training or SSCNet inference on Isaac depth.
No PredictionLayer / map_predict connection.
No prediction writes into observed_map.
No target_lr/target_hr.
No scene or simulator ground truth for exploration.
No large-scale rollout dataset.
Only scripted scene construction, fixed-pose depth/RGB capture, measured-only
observed-map fusion, visualization, and optional one-step EmptyPredictionLayer
expert smoke were run.
```

Stage 4A-3.2 handoff, now completed by Stage 4A-3.5:

```text
Stage 4A-3.5: add A* over observed FREE.
Then Stage 4A-4: run multi-step rollout on the medium scene.
```

Stage 4A-3.5 observed-free A* path-cost is complete. New / updated files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/astar_planner.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_astar_planner.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_astar.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
```

A* rules:

```text
Traversability is derived only from observed_state.
FREE in the robot-height band is traversable support.
OCCUPIED in the band is blocked and inflated by robot radius.
UNKNOWN is not traversable.
No scene metadata, prediction, target labels, or ground truth are used for A*.
A* affects path-cost scoring only; motion still teleports.
Default path_cost_mode remains euclidean.
```

One-step medium A* smoke:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_astar_smoke
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a35_medium_expert_step_astar.log
traversable/blocked/unknown cells: 4316 / 1907 / 8177
reachable/unreachable candidates: 12 / 52
best score: 51.651363679237036
best gain_exp: 110.0
best gain_sc: 0.0
best path_cost: 2.129663036258191
best A* path length: 1.2656854249492382m
best grid/world: [64, 91, 13] / [0.45, 3.15, 1.35]
visualizations: expert_topdown.png, expert_score_bar.png,
traversability_topdown.png
```

Medium A* rollout smoke:

```text
output root: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred
episode dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred/episodes/medium_three_rooms_astar_empty_pred_000
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a35_medium_rollout_astar_empty_pred.log
steps_completed: 5
done_reason: no_valid_candidate
observed_ratio: 0.0 -> 0.04308796296296296
final unknown/free/occupied: 413386 / 15863 / 2751
final pose: [0.6500000000000004, 0.6500000000000004, 1.2]
average reachable candidates: 18.4
average best path_cost: 0.9421159585855353
```

Important blocker:

```text
At expert step 5 of the medium A* rollout, all 64 sampled candidates were
unreachable under conservative observed-free traversability
(traversable=338, blocked=918, unknown=13144). The code did not silently fall
back to Euclidean. Stage 4A-4 should address this before scaling to many
episodes.
```

Validation:

```text
py_compile: passed
test_astar_planner.py: passed
test_sim_expert_astar.py: passed
Euclidean one-step regression: passed
Euclidean rollout regression: passed
observed_ratio non-decreasing: yes
gain_sc with EmptyPredictionLayer: zero
prediction writes observed_map: no
target/ground-truth fields: none
RL/optimizer/BC/IL training: no
```

Next recommended stage:

```text
Stage 4A-4: run multi-step EmptyPredictionLayer rollouts on multiple medium
scene seeds/start poses using A* cost, after addressing the no-valid-candidate
failure mode.
Then Stage 4A-5: carefully begin adding map_predict / prediction gain.
```

Stage 4A-3.6 reachability-aware A* candidate sampling is complete. Updated /
new files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/astar_planner.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_reachable_candidate_sampling.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_astar_planner.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_astar.py
```

Stage 4A-3.6 added:

```text
connected_component_from_start(...)
nearest_traversable_cell(...)
frontier_reachable_candidate_mask(...)
compute_reachable_frontier_candidate_cells(...)
--candidate_sampling_mode frontier|reachable_frontier|auto
--snap_start_to_traversable
--max_snap_radius_cells
```

Rules:

```text
A* auto candidate sampling uses reachable_frontier.
Euclidean auto candidate sampling keeps old frontier behavior.
UNKNOWN remains non-traversable.
No target_lr/target_hr, scene ground truth, simulator ground truth, or
prediction output is used for reachability.
No Euclidean fallback is used.
EmptyPredictionLayer remains the only prediction layer.
```

One-step medium reachable A* output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_astar_reachable_smoke
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a36_medium_expert_step_astar_reachable.log
reachable/unreachable candidates: 64 / 0
reachable component count: 1196
reachable frontier-adjacent count: 1196
candidate source: reachable_frontier
top_n: 16
best score: 88.24634362636618
best gain_exp: 66.0
best gain_sc: 0.0
best path_cost: 0.7479063413600806
best A* path length: 0.28284271247461906m
best grid/world: [58, 82, 11] / [-0.15, 2.25, 1.15]
```

Medium reachable A* rollout output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_reachable_empty_pred/episodes/medium_three_rooms_astar_reachable_empty_pred_000
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a36_medium_rollout_astar_reachable_empty_pred.log
steps_completed: 10
done_reason: max_steps
observed_ratio: 0.0 -> 0.10147453703703704
final unknown/free/occupied: 388163 / 36017 / 7820
final pose: [0.550000011920929, -0.05000000074505806, 1.2000000476837158]
average reachable candidates: 64.0
average reachable component count: 238.8
average reachable frontier-adjacent count: 238.8
no_valid_candidate_steps: []
```

Validation:

```text
py_compile: passed
test_reachable_candidate_sampling.py: passed
test_astar_planner.py: passed
test_sim_expert_astar.py: passed
Euclidean one-step regression: passed
Euclidean rollout regression: passed
observed_ratio non-decreasing: yes
gain_sc with EmptyPredictionLayer: zero
prediction writes observed_map: no
target/ground-truth fields: none
RL/optimizer/BC/IL training: no
```

Next recommended stage:

```text
Stage 4A-4: run multiple medium-scene EmptyPredictionLayer A* rollout
episodes with different seeds/start poses using Stage 4A-3.6 reachable
sampling.
```

Stage 4A-4 multi-episode EmptyPredictionLayer A* rollout dataset is complete.
New files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_batch.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_rollout_dataset.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_rollout_dataset_batch.py
```

Updated files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
```

Dataset output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar
manifest: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/manifest.jsonl
summary json: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/dataset_summary.json
summary md: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/dataset_summary.md
HTML index: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/rollout_dataset_index.html
```

Batch setup:

```text
scene_variant: medium_three_rooms
scene_seeds: 0,1,2
start_variants: start_room_a,start_corridor,start_room_b
intended episodes: 9
actual ok episodes: 9
failed episodes: 0
max_steps: 10
prediction_mode: empty
path_cost_mode: astar
candidate_sampling_mode: reachable_frontier
motion_mode: planar
```

Dataset result:

```text
total transitions: 90
steps_completed min/mean/max: 10 / 10 / 10
done_reason counts: max_steps=9
observed_ratio_end min/mean/max:
  0.08587037037037037 / 0.11455118312757204 / 0.16534722222222223
total_delta_observed_ratio min/mean/max:
  0.08587037037037037 / 0.11455118312757204 / 0.16534722222222223
average reachable candidates: 64.0
average reachable component count: 570.3444444444444
average best_score: 163.2387554327081
average gain_exp: 49.15555555555556
average gain_sc: 0.0
average path_cost: 0.45623051832594874
no_valid_candidate episodes: 0
```

Logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_batch_empty_pred_astar.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_dataset_summary.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_dataset_test.log
```

Validation:

```text
test_rollout_dataset_batch.py: passed
observed_ratio non-decreasing: yes
gain_sc with EmptyPredictionLayer: zero
prediction writes observed_map: no
target/ground-truth fields: none
RL/optimizer/BC/IL training: no
UNKNOWN traversability shortcut: no
Euclidean fallback: no
```

Stage 4A-4 still did not run RL/PPO/BC/IL training, SSCNet training, SSCNet
inference on Isaac depth, PredictionLayer/map_predict connection, prediction
writeback, target label use, ground-truth scoring, physical path execution, or
full RRT planning.

Next recommended stage:

```text
Stage 4A-5: carefully connect map_predict / PredictionLayer as a read-only
Isaac prediction layer, starting with a single-frame preprocessing and
shape-alignment smoke test.
```

Stage 4A-5 single-frame Isaac map_predict alignment smoke is complete. New
files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_sscnet_preprocess.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_prediction_layer.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_map_predict_single.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_isaac_prediction_alignment.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_map_predict_single.py
```

Position convention check:

```text
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a5_position_convention_check.log
NYU position: (480,640) int32 flat indices in [0,240*144*240)
Project2Dto3D output axes: (z_forward,y_up,x_right)
Isaac local volume: x_right [-2.4,2.4], y_up [-1.44,1.44], z_forward [0,4.8]
```

Run:

```text
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a5_isaac_map_predict_single.log
test log: /home/ubuntu22/sc_explorer_ws/logs/stage4a5_isaac_map_predict_single_test.log
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke
episode: medium_three_rooms_seed0_start_room_a_empty_astar
step: 0
depth input shape: (480,640)
position shape: (480,640)
valid position pixels: 166888
logits shape: (1,12,60,36,60)
local prediction shape: (60,36,60)
global prediction shape: (120,120,30)
global valid prediction voxels: 56602
predicted occupied voxels: 15664
predicted_unmeasured voxels: 39400
inference time: 0.1617s
observed_state modified: false
```

Artifacts:

```text
sscnet_input_debug.npz
local_prediction.npz
global_prediction_layer.npz
prediction_alignment_summary.json
isaac_depth_input.png
local_prediction_slices.png
global_prediction_topdown.png
observed_vs_prediction_topdown.png
prediction_not_measured_topdown.png
```

Validation:

```text
py_compile: passed
test_isaac_map_predict_single.py: passed
SimPredictionLayer API: ok
prediction read-only: yes
prediction writeback: no
prediction used by expert/rollout: no
prediction used for traversability/collision/A*: no
RL/optimizer/BC/IL training: no
SSCNet checkpoint modified/trained: no
target/ground-truth artifact fields: none
```

Next recommended stage:

```text
Stage 4A-5.1: use the read-only SimPredictionLayer in one-step expert scoring
on the same frame, verifying I_sc / I_hybrid / I_occ / I_conf become nonzero
while observed_map and A* traversability remain measured-only.
```

Stage 4A-5.1 one-step SC-aware expert scoring is complete.

Updated/added:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_with_prediction.py
```

Input:

```text
observed_state: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar/observed_state_step000.npy
pose: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar/pose_000.json
prediction: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke/global_prediction_layer.npz
tau: 0.1
```

Runs:

```text
empty baseline output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/empty_baseline
prediction output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/sc_prediction
comparison: /home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/comparison_summary.json
logs:
  /home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_empty_baseline.log
  /home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_sc_prediction.log
  /home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_with_prediction_test.log
```

Result:

```text
empty best id: 11
empty score: 331.3448560321166
empty gain_exp/gain_sc/gain_hybrid: 55.0 / 0.0 / 55.0
empty path_cost: 0.1659902032541859

prediction best id: 11
prediction score: 662.6897120642332
prediction gain_exp/gain_sc/gain_hybrid: 55.0 / 55.0 / 110.0
prediction gain_occ: 13.0
prediction gain_conf: 19.406008422374725
prediction path_cost: 0.1659902032541859
candidates with gain_sc > 0: 64 / 64
max/mean gain_sc: 174.0 / 71.59375
total predicted_unmeasured visible count: 4582
best candidate changed: false
top-N overlap: 16 / 16
```

Validation:

```text
py_compile: passed
test_sim_expert_with_prediction.py: passed
observed_state hash unchanged: yes
prediction layer shape matches observed_state: yes
empty gain_sc zero: yes
prediction gain_sc nonzero: yes
gain_hybrid = gain_exp + gain_sc: yes
gain_occ/gain_conf finite: yes
prediction used for traversability/collision/A*: no
prediction blocks rays: no
prediction writeback: no
target/ground-truth leakage: no
RL/optimizer/BC/IL training: no
rollout run: no
```

Next recommended stage:

```text
Stage 4A-6: run a short multi-step rollout with read-only map_predict,
recomputing prediction at each step or using a clearly documented static
prediction ablation, and compare against the Stage 4A-4 measured-only
baseline.
```

Stage 4A-6 short multi-step SC-aware rollout is complete. New / updated files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_map_predictor.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/compare_sc_pred_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_sc_aware_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
```

The Stage 4A-6 dynamic loop is:

```text
Isaac current camera pose
  -> capture RGB/depth
  -> update measured-only observed_state from depth
  -> run map_predict on current depth/pose
  -> align prediction to global observed_state shape
  -> create read-only SimPredictionLayer
  -> run SC-aware expert scoring
  -> select next viewpoint
  -> planar teleport camera
  -> repeat
```

Run:

```text
episode: medium_three_rooms_seed0_start_room_a_sc_pred_dynamic_000
scene: medium_three_rooms
seed/start: 0 / start_room_a
max_steps: 5
prediction_mode: sim_dynamic
path_cost_mode: astar
candidate_sampling_mode: reachable_frontier
motion_mode: planar
checkpoint: /home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke
```

Result:

```text
steps_completed: 5
done_reason: max_steps
observed_ratio: 0.000000 -> 0.05899768518518519
final unknown/free/occupied: 406513 / 21226 / 4261
final pose: [-4.25, -4.150000095367432, 1.2000000476837158]
average gain_exp: 49.6
average gain_sc: 49.4
average gain_hybrid: 99.0
average gain_occ: 8.8
average gain_conf: 16.96283725500107
average best_score: 441.9845465468916
candidates_with_gain_sc_positive min/mean/max: 63 / 63.6 / 64
```

Performance:

```text
model_loaded_once: true
average map_predict preprocess_time: 0.05369079960000818 s
average map_predict inference_time: 0.020522295199771178 s
average map_predict alignment_time: 0.03251961960013432 s
average map_predict total_time: 0.14326694260016665 s
average expert_time: 1.026360238399866 s
total_wall_time: 19.86559214800036 s
GPU memory peak: 794354176 bytes
hardware: AMD Ryzen 9 9950X3D, 32 CPU threads, 32GB RAM, NVIDIA RTX 5080
```

Comparison to Stage 4A-4 empty baseline:

```text
baseline: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar
comparison: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke/comparison_to_empty_baseline
compared_steps: 5
empty final observed_ratio: 0.06896296296296296
SC final observed_ratio: 0.05899768518518519
SC-empty observed_ratio delta: -0.009965277777777774
changed selected actions: 5
mean score delta SC-empty: 233.79287700349096
mean gain_exp delta SC-empty: -5.199999999999996
mean SC gain_sc: 49.4
```

Observed_ratio at the compared 5-step horizon is lower for the SC-aware
rollout than the measured-only baseline. Stage 4A-6 should therefore be read
as a dynamic prediction integration and safety milestone, not a performance
improvement result.

Validation:

```text
py_compile: passed
test_sim_sc_aware_rollout.py: passed
observed_ratio non-decreasing: yes
gain_sc nonzero: yes
gain_hybrid = gain_exp + gain_sc: yes
observed_state hash unchanged by prediction: yes
prediction writeback: no
prediction used for traversability/collision/A*: no
prediction blocks rays: no
checkpoint modified: no
target/ground-truth leakage: no
RL/optimizer/BC/IL/SSCNet training: no
required episode/comparison files present: yes
final log blockers: none found
```

Fixed issue:

```text
first run failed at step 0 visualization because prediction_alignment_summary.json
was missing observed_state_source.
fix: isaac_map_predictor.py writes observed_state_source, depth_source,
pose_source, and camera_info_source.
final status: rerun completed and test passed.
```

Next recommended stage:

```text
Stage 4A-6.1: analysis/ablation/tuning before longer rollout:
static prediction ablation, gain_sc weighting, tau sweep, maybe cap prediction
gain, inspect prediction overlays and selected actions, then run a 10-step
SC-aware rollout compared against measured-only baseline.
Do not jump to RL yet.
```

Stage 4A-6.1 SC-aware rollout analysis/ablation is complete. New files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/analyze_sc_rollout_behavior.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sc_pred_ablation_sweep.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_sc_pred_ablation.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sc_pred_ablation.py
```

Updated files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
```

The simulator expert now supports optional scoring ablations:

```text
sc_gain_weight: default 1.0
sc_gain_cap: default None
score_gain_mode: hybrid_raw | hybrid_weighted
```

Default behavior remains compatible with Stage 4A-6: `hybrid_raw`,
`sc_gain_weight=1.0`, and no cap.

Existing SC-vs-empty analysis:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/existing_sc_vs_empty
empty final observed_ratio at 5 steps: 0.06896296296296296
original SC final observed_ratio: 0.05899768518518519
delta SC-empty: -0.009965277777777774
first SC lag step: 1
changed selected actions: 5 / 5
mean gain_exp empty/SC: 54.8 / 49.6
mean gain_sc SC: 49.4
mean path_cost empty/SC: 0.36998367643136965 / 0.2768163156997422
mean best_score empty/SC: 208.19166954340062 / 441.9845465468916
dense gain_sc candidate steps: 0,1,2,3,4
```

Completed ablation configs:

```text
dynamic_w025_tau01
dynamic_w05_tau01
dynamic_w1_tau03
dynamic_w1_tau01_cap50
static_step0_weight_1p0_tau_0p1
```

All ablations completed successfully:

```text
steps_completed: 5 for each config
failed configs: none
final observed_ratio: 0.05899768518518519 for each config
delta vs empty baseline: -0.009965277777777774 for each config
delta vs original SC: 0.0 for each config
same selected actions as original SC: 5 / 5 for each config
```

Performance:

```text
dynamic wall time: 27.75568118299998s to 30.416639261s per ablation
dynamic avg map_predict inference: 0.02070385000006354s to 0.03013971940004012s
dynamic GPU peak: 794296320 bytes
static step0 wall time: 22.448690754999916s
static step0 avg map_predict inference: 0.0s
```

Outputs:

```text
analysis: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/existing_sc_vs_empty
ablation: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_ablation
summary: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_ablation/summary
qualitative: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/qualitative_inspection
```

Logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a61_existing_sc_analysis.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a61_ablation_sweep.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a61_ablation_summary.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a61_ablation_test.log
```

Validation:

```text
py_compile: passed
test_sc_pred_ablation.py: passed
observed_ratio non-decreasing: yes
weighted gain formula: yes
prediction writeback: no
prediction used for traversability/collision/A*/ray blocking: no
UNKNOWN traversable: no
Euclidean fallback: no
target/ground-truth leakage: no
RL/PPO/BC/IL/optimizer/SSCNet training: no
checkpoint modified: no
```

Stage 4A-6.1 conclusion:

```text
The underperformance is not a plumbing failure. The prediction path is active
and safe, but gain_sc is dense across nearly all reachable candidates and did
not alter selected actions under weight/tau/cap ablations. Current map_predict
scoring appears too uncalibrated or insufficiently discriminative for this
Isaac rollout. The next step should inspect Isaac-to-SSCNet preprocessing,
global alignment, confidence calibration, and NYU-to-Isaac domain shift before
longer SC-aware rollouts. Still do not jump to RL/IL.
```

Stage 4A-6.2 map_predict preprocessing / alignment / calibration diagnostics
are complete.

New scripts:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_isaac_sscnet_preprocess.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_prediction_global_alignment.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/evaluate_prediction_against_future_observed.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_alignment_variant_sweep.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_map_predict_diagnostics.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_map_predict_diagnostics.py
```

Outputs:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/preprocess_stats
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/global_alignment
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/future_observed_eval
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/alignment_variant_sweep
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/candidate_score_decomposition
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/summary
```

Key numbers:

```text
Isaac mean depth: 2.532350206375122 m
NYU mean depth: 2.8481276988983155 m
Isaac valid position ratio: 0.565763671875
NYU position nonzero proxy ratio: 0.74495458984375
mean valid in-front ratio: 0.9977230302200312
mean inside global bounds ratio: 0.8669629629629629
best alignment variant: xz_swap_variant
default alignment rank: 7
Brier improvement vs default: 0.0735458940774611
tau=0.1 predicted_unmeasured mean: 35118.2
tau=0.1 later measured fraction: 0.059004217437215026
tau=0.1 occupied precision: 0.25632042463242544
tau=0.1 free precision: 0.9323112341592195
tau=0.1 occupied Brier: 0.2786559495144023
ECE-like occupied calibration: 0.3405436085907938
gain_exp/gain_sc correlation: 0.9647202023737985
final_score vs inverse path_cost correlation: 0.9713818732156227
```

Stage 4A-6.2 diagnosis:

```text
Primary suspected issue: local SSCNet prediction to global map alignment
convention. The direct frustum sanity check is mostly forward-facing, but the
diagnostic xz_swap_variant fits delayed sensor observations materially better
than current_default.

Secondary issue: confidence / prediction valid mask is too dense and
unselective on Isaac at tau=0.1. gain_sc largely duplicates gain_exp and low
path_cost dominates top-N score ranking.
```

Validation:

```text
py_compile: passed
test_map_predict_diagnostics.py: passed
future observations: post-hoc evaluation only, not planning/scoring
observed_state modified: no
prediction writeback: no
prediction traversability/collision/A*/ray blocking/candidate reachability: no
checkpoint modified: no
RL/PPO/BC/IL/optimizer/SSCNet training: no
diagnostics launched Isaac: no
log scan blockers: none found
```

Next:

```text
Stage 4A-6.3 should fix/reconcile the map_predict alignment convention and
rerun Stage 4A-5/5.1/6 smoke. If alignment is fixed but dense calibration
remains, implement calibrated/capped confidence-based I_sc smoke. If domain
shift remains dominant, collect Isaac-domain validation or synthetic supervised
data before relying on SC-aware rollout. Do not jump to RL/IL.
```

Stage 4A-6.3 SSCNet alignment convention fix is complete.

New scripts:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/document_sscnet_axis_convention.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/fix_prediction_alignment_convention.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_alignment_convention_fix.py
```

Updated scripts:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_sscnet_preprocess.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_map_predictor.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_map_predict_single.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_isaac_prediction_alignment.py
```

Axis convention result:

```text
Project2Dto3D: scatter flat -> view(W,H,D) -> permute(D,H,W)
raw Python dataloader position flatten: np.ravel_multi_index((x,y,z),(240,144,240))
C++/ROS projection flatten: z*(240*144)+y*240+x
dataloader target_lr.T: reverses stored target axes before loss flattening
old current_default_v0: output axes (z_forward,y_up,x_right)
fixed code_consistent_v1: position flatten (z_forward,y_up,x_right),
  output axes (x_right,y_up,z_forward)
```

Outputs:

```text
axis audit:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a63_alignment_fix/axis_convention_audit

convention eval:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a63_alignment_fix/convention_eval

fixed single-frame:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke_alignment_fixed

fixed one-step expert:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_alignment_fixed_smoke

fixed rollout:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_alignment_fixed_smoke
```

Key metrics:

```text
best diagnostic convention: xz_swap_diagnostic
recommended fixed convention: code_consistent_v1
current_default_v0 occupied Brier: 0.2786559495144023
code_consistent_v1 occupied Brier: 0.20511005543694122
Brier improvement: 0.0735458940774611
current_default_v0 ECE-like: 0.3405436085907937
code_consistent_v1 ECE-like: 0.22427722861569463
later measured fraction default/fixed: 0.059004217437215026 / 0.059004217437215026
```

Fixed smoke result:

```text
single-frame global_valid_prediction_count: 56602
single-frame global_predicted_occupied_count: 16792
single-frame predicted_unmeasured_count: 39400
one-step best candidate id: 11
one-step best gain_exp/gain_sc/gain_hybrid: 55.0 / 55.0 / 110.0
5-step fixed observed_ratio: 0.0 -> 0.05899768518518519
empty baseline 5-step observed_ratio: 0.06896296296296296
original SC 5-step observed_ratio: 0.05899768518518519
changed actions vs original SC: 0
```

Validation:

```text
py_compile: passed
test_alignment_convention_fix.py: passed
observed_state modified: no
prediction writeback/traversability/collision/A*/ray blocking/candidate reachability: no
future observations: post-hoc evaluation only
checkpoint modified: no
RL/PPO/BC/IL/optimizer/SSCNet training: no
```

Next:

```text
Use code_consistent_v1 for future Isaac map_predict runs. Since fixed
alignment improves prediction diagnostics but not the 5-step rollout action
sequence or observed_ratio, Stage 4A-6.4 should implement
calibrated/confidence-gated I_sc before any 10-step SC-aware rollout scaling.
Still do not jump to RL/IL.
```
