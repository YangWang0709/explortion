Decisions
Updated: 2026-05-30

- Use best checkpoint only:
  `/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar`
- Do not use the latest checkpoint for Stage 2A inference.
- Use official `make_model("sscnet", num_classes=12)` and strict checkpoint loading.
- Treat class 0 as empty/free based on `config.py` and `ssc_metrics.py`.
- Compute `occupied_prob = 1 - prob[class 0]`.
- Keep prediction output in standalone `.npz` files.
- Do not write prediction into observed_map.
- Stage 2B is strict paper-faithful deterministic expert candidate scoring using PredictionLayer, not RL.
- Decision: `target_lr` and `target_hr` must not be used for expert scoring.
- For Stage 2B offline NYU smoke tests, approximate measured set S only from sensor-derived fields:
  `tsdf_lr`, `position`, or their union.
- Default measured mode is `tsdf_lr`: measured if `tsdf_lr >= 0.001`; measured surface blocking if `tsdf_lr == 1.0`.
- Prediction set P is `PredictionLayer.confidence >= tau` and not in measured S.
- Default raycast mode is `non_blocking`: scene-completion predictions do not block rays.
- `sc_blocking` is allowed only as an explicit paper-ablation mode.
- Stage 2B utility is per-candidate gain divided by approximate position/yaw time cost.
- Full SC-Explorer RRT tree utility Eq. 12 is future work.
- Stage 4A-1 only validates simulator depth observation and observed_map update.
- Prediction and expert integration are Stage 4A-2.
- Stage 4A-1 observed_map is measured-only from simulator depth ray marching;
  no SSCNet prediction, target_lr, target_hr, ground truth, or expert score is
  written into it.
- Decision: Stage 4A-3 uses teleport planar camera motion for rollout smoke,
  not physical robot path planning.
- Stage 4A-3 keeps EmptyPredictionLayer as the only prediction layer and keeps
  observed_map measured-only from Isaac depth.
- Decision: before scaling rollouts or datasets, increase scripted scene
  complexity beyond the minimal room.
- Stage 4A-3.2 uses a deterministic medium-complexity cuboid-only Isaac scene
  with 3 rooms, 1 corridor, 3 openings, and at least 10 obstacles.
- Stage 4A-3.2 remains a scene/depth/observed-map smoke stage only:
  no RL, PPO, behavior-cloning training, imitation-learning training, SSCNet
  training, SSCNet inference on Isaac depth, PredictionLayer connection, or
  prediction writes into observed_map.
- Use explicit medium-scene voxel bounds `x/y=[-6,6]`, `z=[0,3]` for
  depth_to_voxel smoke outputs.
- Decision: before IL/RL or large-scale datasets, construct the paper expert
  step by step in simulator.
- Stage 4A-3.5 adds observed-free A* path-cost scoring after medium-scene
  validation.
- Stage 4A-3.5 A* traversability is derived only from measured
  `observed_state`: FREE is traversable support, OCCUPIED is blocked and
  inflated, UNKNOWN is not traversable.
- Stage 4A-3.5 does not use scene metadata, target labels, ground truth, or
  prediction output for A* traversability or path cost.
- Stage 4A-3.5 keeps EmptyPredictionLayer only and keeps observed_map
  measured-only from Isaac depth.
- A* is currently only for expert scoring; simulator motion remains planar
  teleport and no physical path execution or full SC-Explorer RRT tree planner
  is implemented.
- If all A* candidates are unreachable, do not silently switch to Euclidean;
  report `no_valid_candidate` and fix traversability/candidate sampling in a
  documented follow-up.
- Decision: before scaling rollout datasets, A* candidate generation must
  sample from the current reachable observed-free component.
- Stage 4A-3.6 A* `candidate_sampling_mode=auto` resolves to
  `reachable_frontier`; Euclidean `auto` preserves the old frontier sampler.
- Stage 4A-3.6 may snap the A* start to the nearest observed traversable cell
  within an explicit local radius, but UNKNOWN remains non-traversable and no
  scene/simulator ground truth or prediction output may be used for
  reachability.
- Do not use Euclidean fallback to hide A* no-valid-candidate failures.
- Decision: Stage 4A-4 generates a multi-episode measured-only expert rollout
  dataset before connecting map_predict.
- Stage 4A-4 uses EmptyPredictionLayer only, observed-free A* cost,
  reachable-frontier candidate sampling, measured-only Isaac depth updates, and
  planar teleport motion.
- Stage 4A-4 dataset generation must not use target_lr/target_hr, scene ground
  truth, simulator ground truth, SSCNet inference on Isaac depth, PredictionLayer
  output, prediction writeback, RL, PPO, behavior cloning training,
  imitation-learning training, optimizer steps, UNKNOWN traversability, or
  Euclidean fallback.
- Stage 4A-5 should connect map_predict / PredictionLayer only as a read-only
  Isaac prediction layer, beginning with single-frame preprocessing and
  shape-alignment validation.
- Decision: Stage 4A-5 introduces map_predict only as a read-only simulator
  prediction layer. It must not write into observed_map and must not affect
  traversability, collision, A*, expert decisions, or rollout decisions.
- Stage 4A-5 local SSCNet output is interpreted as `(z_forward, y_up,
  x_right)` because `Project2Dto3D` scatters into `(240,144,240)` and then
  permutes to `(D,H,W)`; the global simulator layer is aligned back to
  observed_state axis order `(world_x, world_y, world_z)`.
- Stage 4A-5 Isaac-to-SSCNet preprocessing is smoke-only and provisional with
  local volume `x_right=[-2.4,2.4]`, `y_up=[-1.44,1.44]`,
  `z_forward=[0,4.8]`; it documents the NYU-to-Isaac domain shift and should
  be revalidated before rollout-scale prediction use.
- Decision: Stage 4A-5.1 allows read-only `SimPredictionLayer` output to
  affect one-step simulator expert information gain only.
- Stage 4A-5.1 prediction may contribute `gain_sc`, `gain_hybrid`, `gain_occ`,
  and `gain_conf` over visible voxels that are predicted and not measured.
- Stage 4A-5.1 prediction must not affect `observed_state`, observed_map
  writes, candidate sampling, A* traversability, A* path validity, collision
  checking, or ray blocking.
- Stage 4A-5.1 remains a one-step expert-scoring smoke only:
  no rollout, RL, PPO, behavior cloning training, imitation-learning training,
  SSCNet training, checkpoint modification, target labels, or ground-truth
  scoring.
- Decision: Stage 4A-6 uses dynamic per-step map_predict for a short
  SC-aware rollout on `medium_three_rooms` seed `0`, start `start_room_a`.
- Stage 4A-6 prediction is information-gain-only and read-only.
- Stage 4A-6 prediction must not affect observed_map, `observed_state`,
  traversability, collision checking, A* validity/path cost, candidate
  reachability, or ray blocking.
- Stage 4A-6 keeps UNKNOWN non-traversable and does not use Euclidean fallback.
- Stage 4A-6 loads the SSCNet checkpoint once per rollout and reuses the model
  at every step; do not reload the checkpoint per step.
- Stage 4A-6 hardware context:
  AMD Ryzen 9 9950X3D, 32 CPU threads, 32GB RAM, NVIDIA RTX 5080. Use GPU
  inference and reasonable CPU threading, but avoid parallel Isaac runs.
- Stage 4A-6 still does not use target_lr/target_hr, scene ground truth,
  simulator ground truth, RL, PPO, behavior cloning training,
  imitation-learning training, optimizer steps, SSCNet training, checkpoint
  modification, physical path execution, or a full RRT tree planner.
- Decision: after Stage 4A-6, the next step is Stage 4A-6.1
  analysis/ablation/tuning, not RL. Analyze the 5-step underperformance
  against measured-only baseline with static-prediction ablation, gain_sc
  weighting, tau sweep, optional prediction-gain cap, and prediction
  overlay/action inspection before any 10-step SC-aware rollout.
- Decision: do not scale SC-aware rollout length until Stage 4A-6.1
  underperformance is explained; the first five tuning/static ablations did not
  improve measured coverage or change selected actions.
- Stage 4A-6.1 introduces `sc_gain_weight`, `sc_gain_cap`, and
  `score_gain_mode` as scoring ablation controls only. Raw paper-style gains
  remain logged as `gain_exp`, `gain_sc`, and `gain_hybrid=gain_exp+gain_sc`.
- Stage 4A-6.1 keeps prediction read-only and information-gain-only.
  Prediction must not affect observed_map, `observed_state`, traversability,
  A*, collision, candidate reachability, or ray blocking.
- Stage 4A-6.1 found gain_sc to be dense across nearly all reachable
  candidates in the analyzed seed/start; weight, tau, and cap changed scores
  but not selected actions.
- Do not jump to RL/IL after a single underperforming SC rollout. Next inspect
  map_predict preprocessing, alignment, confidence calibration, and
  NYU-to-Isaac domain shift before longer SC-aware rollout scaling.
- Decision: Stage 4A-6.2 uses future measured observations only for post-hoc
  prediction diagnostics, not expert scoring or planning.
- Stage 4A-6.2 does not use future observations for candidate scoring,
  traversability, collision, A*, reachability, ray blocking, or observed_map
  updates.
- Do not scale SC-aware rollouts until map_predict preprocessing, alignment,
  and confidence calibration are understood and fixed/reconciled.
- Prediction remains read-only and information-gain-only.
- Prediction must not affect observed_map, `observed_state`, traversability,
  A*, collision checking, candidate reachability, or ray blocking.
- Stage 4A-6.2 diagnostic recommendation is to treat alignment convention as
  the primary suspect because an `xz_swap_variant` projection fit delayed
  sensor measurements better than `current_default`; rerun Stage 4A-5/5.1/6
  smoke only after a deliberate alignment fix/reconciliation.
- Stage 4A-6.2 also found confidence calibration to be dense and unselective
  on Isaac at tau `0.1`; if alignment is fixed but this remains, implement
  calibrated/capped I_sc before longer rollouts.
- Do not jump to RL/IL after SC-aware underperformance or after Stage 4A-6.2
  diagnostics.
- Decision: Stage 4A-6.3 fixes/reconciles SSCNet output-axis alignment before
  any more SC-aware rollout scaling.
- Stage 4A-6.3 supersedes the Stage 4A-5 local-output-axis interpretation for
  Isaac map_predict. The old `current_default_v0` convention followed the raw
  Python dataloader flatten path and interpreted local output as
  `(z_forward,y_up,x_right)`. The code-consistent Isaac convention is now
  `code_consistent_v1`: input position flatten follows the C++/ROS projection
  formula `z*(240*144)+y*240+x`, and local output axes are
  `(x_right,y_up,z_forward)`.
- `xz_swap_diagnostic` is retained only as a diagnostic reprojection name. The
  convention to use for future Isaac map_predict smoke runs is
  `code_consistent_v1`, because it is supported by code audit and improves
  post-hoc future-observed Brier versus `current_default_v0`.
- Stage 4A-6.3 fixed alignment improves post-hoc prediction diagnostics but
  does not improve the 5-step SC-aware rollout in the seed/start smoke; the
  fixed rollout selected the same actions as original SC and remained below
  the empty baseline.
- Future observed maps are only post-hoc evaluation, not planning.
- Prediction remains read-only and information-gain-only.
- Prediction must not affect observed_map, `observed_state`, traversability,
  A*, collision, candidate reachability, or ray blocking.
- Do not jump to RL/IL until the remaining confidence/selectivity issue is
  resolved. The next stage should calibrate/gate `I_sc`, not run PPO, behavior
  cloning, imitation-learning training, or optimizer-based policy learning.
