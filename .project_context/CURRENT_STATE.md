Project: SC-Explorer SSCNet Stage 4A-6.6c-home-like-scene-v1-validation
Updated: 2026-06-04

Current state:

- Stage 4A-7.0 BC dataset design/preparation is complete and validated.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_design_preparation`.
- This was BC dataset schema/converter/audit preparation only: no real Isaac
  startup, no capture, no map_predict, no SSCNet inference, no action
  execution, no rollout, no long rollout, no BC training, no optimizer step,
  no model checkpoint, and no RL/GDPO/PPO.
- Primary label policy:
  `stage4a613_uncertainty_bonus_executed_primary`, using Stage 4A-6.13
  executed `uncertainty_bonus_composite_beta8` primary actions as candidate-set
  classification labels.
- Primary BC dataset:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_design_preparation/bc_dataset_primary_short_rollout.npz`.
  Counts: primary samples `30`, starts `10`, sequence steps `[0, 1, 2]`,
  padded candidate rows `1920`, `D_raw=91`, `D_model=16`, and valid primary
  labels `30`.
- Quality/split results: `strict_keep=30`, `moderate_keep=30`,
  `analysis_only=30`; split policy is leave-one-start-out plus
  split-by-start-variant (`train=21`, `val=6`, `test=3`) with `10`
  leave-one-start-out folds.
- Additional outputs include shadow multilabel, one-action reference, and
  combined research-view datasets; feature stats at
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_design_preparation/normalization_stats.npz`;
  visual QA HTML at
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_design_preparation/bc_dataset_index.html`;
  dataset card at
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_design_preparation/bc_dataset_card.md`.
- Forbidden field audit passed: no target/ground-truth/future-observed fields
  are used as BC features, labels, scores, or filters. Prediction and
  uncertainty remain recorded candidate features only and were not written
  into observed_state.
- Forward-only smoke ran with CE loss `4.157931327819824`; optimizer step,
  backward, model save, and checkpoint creation were all false.
- Source USD, fixed USD, checkpoint, Stage 4A-6.13 dataset/manifest, and prior
  6.7/6.8/6.11/6.12 datasets were unchanged. Git large artifact policy
  remained clean.
- Recommended next:
  review the Stage 4A-7.0 dataset QA package, then explicitly approve either
  Stage 4A-7.1 BC dry-run/tiny training design or a second bounded short
  rollout with small variations. Do not jump directly to long rollout, full BC
  training, or RL/GDPO/PPO.

- Stage 4A-6.13a Isaac close timeout lifecycle hardening is complete and
  validated. Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a613a_isaac_close_guard_hardening`.
- This was lifecycle hardening only: no real Isaac startup, no capture, no
  map_predict, no SSCNet inference, no action execution, no rollout, no long
  rollout, no training, and no BC/IL/RL/GDPO/PPO.
- Implemented reusable guard utilities in
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_lifecycle_guard.py`, a
  process-level supervisor wrapper in
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_with_isaac_close_guard.py`,
  fake child coverage in
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/fake_hanging_isaac_child.py`,
  and validation/report generation in
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_lifecycle_guard.py`.
- Patched
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a613_uncertainty_bonus_short_rollout_pilot.py`
  for future finalization-sentinel support before `simulation_app.close()`.
  The close point is now after output finalization when the optional guard
  arguments are used; rollout scoring behavior was not changed and Stage
  4A-6.13 was not rerun.
- Fake child validation passed:
  clean exit returned `0` with `close_status=clean_exit`; hang after safe
  finalization returned `0` with
  `close_status=forced_terminated_after_finalization` and
  `success_with_close_hang=true`; hang before finalization returned nonzero
  with `close_status=failed_before_finalization`.
- Process and GPU snapshots are emitted before/after both the top-level
  hardening test and each fake child supervisor run. Orphan scan is report-only
  by default, and no unrelated process kill list was produced.
- Source USD, fixed USD, checkpoint, Stage 4A-6.13 dataset, and Stage
  4A-6.13 manifest hashes were unchanged. Prediction and uncertainty writeback
  remained false.
- Recommended next:
  review the 6.13 visual/audit package, then choose either BC dataset
  design/preparation or a second explicitly approved short rollout with small
  variations. Do not jump directly to long rollout. Any future long rollout
  must use the close guard and include expert data quality visualization/audit
  outputs.

- Stage 4A-6.13 uncertainty-bonus bounded short rollout pilot is complete and
  validated. Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a613_uncertainty_bonus_short_rollout_pilot`.
- This was the explicitly approved short rollout, not a long rollout and not
  a full expert dataset. Counts: `start_count=10`,
  `max_decision_steps_per_start=3`, `decision_frame_count=30`,
  `terminal_frame_count=10`, `capture_count=40`, `map_predict_calls=30`,
  and `executed_action_count=30`. Isaac startup count was `1`; after all
  files were finalized, `simulation_app.close()` hung and the process was
  terminated without a second Isaac startup.
- Primary expert formula was `uncertainty_bonus_composite_beta8`:
  `gain_exp / cost + 48 * minmax(source_occ_free) + 8 * uncertainty_composite`,
  where `uncertainty_composite =
  0.4*minmax(candidate_uncertain_fraction) +
  0.4*minmax(candidate_entropy_mean) +
  0.2*minmax(1 - candidate_margin_mean)`. Minmax scope was per decision
  frame over measured-valid candidates only.
- Rollout result: observed ratio mean moved from `0.2444367224367224` to
  `0.262136036036036`; total newly observed voxels were `412571`, mean
  `41257.1` per start. Done reasons were `max_steps_reached: 10`;
  `no_valid_candidate_count=0`, `stuck_revisit_count=0`, and
  `candidate_all_local_count=0`.
- Decision comparison over all 30 decision frames:
  action changed count `17` vs measured-only, `3` vs lambda48, `6` vs
  confidence-gated, and `0` vs the Stage 4A-6.12 decision on step 0.
  Branch counts vs measured were `same_as_measured=7`, `local_jitter=17`,
  and `distinct_nonmeasured_branch=6`.
- Selected uncertainty health: confidence mean/min
  `0.8360315690272979 / 0.6457599577356558`, entropy mean/max
  `0.23500558781373931 / 0.4559122721354167`, and margin mean/min
  `0.7639459535016347 / 0.46364005667264346`. Low-cost artifact,
  historical-prior basin, and formula-dominated-by-uncertainty counts were
  all `0`.
- Gates stayed closed: no long rollout, no full expert dataset, no
  BC/IL/RL/GDPO/PPO, no training, no replay buffer, no policy checkpoint, no
  prediction writeback, no uncertainty writeback, and no prediction/uncertainty
  use for traversability, collision, ray blocking, candidate validity, or edge
  validity. Source USD, fixed USD, checkpoint, and prior 6.7/6.8/6.9/6.10a/
  6.11/6.12 datasets were unchanged.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a613_py_compile.log` and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a613_uncertainty_bonus_short_rollout_pilot_test.log`.
  Expert data quality, prediction safety, uncertainty safety, rollout safety,
  runtime quality, and dataset integrity audits all passed with no warnings or
  blockers.
- Main artifacts: dataset
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a613_uncertainty_bonus_short_rollout_pilot/short_rollout_dataset_uncertainty_bonus.npz`,
  HTML
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a613_uncertainty_bonus_short_rollout_pilot/short_rollout_uncertainty_bonus_index.html`,
  and MP4
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a613_uncertainty_bonus_short_rollout_pilot/short_rollout_flythrough.mp4`.
- Recommended next:
  review the 6.13 visual/audit package, then choose either BC dataset
  design/preparation or a second explicitly approved short rollout with small
  variations. Do not jump directly to long rollout; any future long rollout
  must include expert data quality visualization/audit outputs.

- Stage 4A-6.12 uncertainty-as-exploration-bonus decision pilot is complete
  and validated. Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a612_uncertainty_exploration_bonus_pilot`.
- This was decision-only: `action_execution_count_this_stage=0`,
  `rollout_executed=false`, no Isaac startup, no capture, no map_predict, no
  SSCNet inference, no action, no second action, no third frame, no long
  rollout, no training, and no BC/IL/RL/GDPO/PPO.
- Inputs loaded:
  Stage 4A-6.10a dense uncertainty artifacts and Stage 4A-6.11
  uncertainty-aware candidate features. `start_count=10`,
  `candidate_rows_loaded=469`, beta values `{2,4,8,16,32}`, formulas
  `fraction`, `entropy`, `low_margin`, and `composite`.
- Recommended formula:
  `uncertainty_bonus_composite_beta8`, i.e.
  `gain_exp / cost + 48 * minmax(source_occ_free) + 8 * uncertainty_composite`.
  `uncertainty_bonus_runtime_ready=true`.
- Decision comparison for the recommended formula:
  action changed count `6` vs measured-only, `1` vs lambda48, and `1` vs the
  Stage 4A-6.11 confidence-gated primary. Candidate-all-local count was `5`,
  matching lambda48 (`increased_vs_lambda48=false`).
- Selected uncertainty health:
  confidence mean/min `0.8623173562170562 / 0.7047213040865384`, entropy
  mean/max `0.20959808035924596 / 0.3686438927283654`, and margin mean/min
  `0.8020552913679388 / 0.5739912766676682`.
- Risk and quality:
  `uncertainty_bonus_quality_audit.passed=true`,
  `uncertainty_bonus_risk_audit.passed=true`, with no warnings or blockers.
  `source_occ_free` and uncertainty metrics were kept separate, and
  prediction/uncertainty were not used for traversability, collision, ray
  blocking, candidate validity, edge validity, target/ground-truth scoring, or
  future-observed scoring.
- Main artifacts:
  decision dataset
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a612_uncertainty_exploration_bonus_pilot/expert_decision_dataset_uncertainty_bonus.npz`,
  HTML
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a612_uncertainty_exploration_bonus_pilot/uncertainty_bonus_index.html`,
  and future short rollout sketch
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a612_uncertainty_exploration_bonus_pilot/future_short_rollout_with_uncertainty_bonus_sketch.md`.
- Validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a612_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a612_uncertainty_exploration_bonus_pilot.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a612_uncertainty_exploration_bonus_pilot_test.log`.
  Test reported `all_passed=true`.
- Recommended next:
  review the 6.12 visual/audit package. If accepted, the next step can be an
  explicitly approved short rollout design using the recommended formula. Do
  not jump directly to long rollout.

- Stage 4A-6.11 uncertainty-aware lambda one-action pilot is complete and
  validated. Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a611_uncertainty_aware_lambda_one_action_pilot`.
- Primary formula:
  `confidence_gated_lambda48_v1`, i.e.
  `gain_exp / cost + 48 * minmax(source_occ_free * candidate_confidence_mean)`.
  Shadow formulas saved: measured-only, lambda48 baseline,
  confidence-margin gated, uncertainty bonus beta8, uncertainty penalty beta8,
  and entropy penalty beta8.
- Runtime/result:
  `start_count=10`, `frame_count=10`, `capture_count=10`,
  `map_predict_calls=10`, `dense_uncertainty_artifacts=10`,
  `executed_action_count=10`, `exactly_one_action_per_start=true`,
  `second_action_count=0`, `third_frame_count=0`,
  `continuous_rollout_executed=false`, and `long_rollout_executed=false`.
  No new Isaac process was started for this finalizer; it reused the validated
  Stage 4A-6.8 one-frame captures and Stage 4A-6.10a dense uncertainty
  artifacts. No second Isaac startup, Frame2, or rollout was introduced.
- Candidate uncertainty:
  469 measured-valid candidate rows were reconstructed across the 10 starts
  from the 64 requested reachable-frontier candidates, each with real
  candidate-visible dense confidence/entropy/margin features. Candidate
  confidence/entropy/margin means are `0.8604786937920237`,
  `0.1888305449472253`, and `0.8094706315352114`. Selected primary
  confidence/entropy/margin means are `0.8796395396528542`,
  `0.19162816140892908`, and `0.8282547902721005`.
- Decision comparison:
  primary vs measured produced `same_as_measured=2`, `local_jitter=6`,
  `distinct_nonmeasured_branch=2`, and `no_valid_candidate=0`.
  Action changed counts are `6` vs measured-only, `0` vs the 6.11 lambda48
  baseline shadow, and `2` vs Stage 4A-6.8 / Stage 4A-6.9 Frame1. Mean action
  distance vs lambda48 is `0.009999999999999787m`; mean yaw delta is
  `0.06747409422235524rad`.
- Safety and quality:
  `prediction_safety_audit.passed=true`,
  `uncertainty_safety_audit.passed=true`,
  `dataset_integrity_report.passed=true`, and
  `expert_data_quality_audit.passed=true`. Gates remain closed for rollout,
  second action, third frame, long rollout, BC/IL/RL/GDPO/PPO, training,
  replay buffer, policy checkpoint, prediction writeback, uncertainty
  writeback, and prediction/uncertainty traversability, collision, ray
  blocking, or candidate-validity use. Source USD, fixed USD, checkpoint,
  source observed_state, and prior 6.8/6.9 datasets were unchanged.
- Main artifacts:
  dataset
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a611_uncertainty_aware_lambda_one_action_pilot/expert_dataset_uncertainty_lambda.npz`,
  manifest
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a611_uncertainty_aware_lambda_one_action_pilot/expert_dataset_manifest.jsonl`,
  HTML
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a611_uncertainty_aware_lambda_one_action_pilot/expert_uncertainty_lambda_index.html`,
  and MP4
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a611_uncertainty_aware_lambda_one_action_pilot/expert_uncertainty_lambda_flythrough.mp4`.
- Validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a611_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a611_uncertainty_aware_lambda_one_action_pilot.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a611_uncertainty_aware_lambda_one_action_pilot_test.log`.
  Test reported `all_passed=true`.
- Recommended next:
  review the uncertainty-aware visual package and choose BC dataset
  design/preparation or an explicitly approved short rollout. Do not jump
  directly to long rollout; any future short/long rollout must include expert
  data quality visualization and audit outputs.

- Stage 4A-6.10a dense prediction uncertainty artifact regeneration is
  complete and validated. Primary output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a610a_dense_prediction_uncertainty_artifacts`.
  Dense rerun audit output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a610a_uncertainty_audit_rerun_dense`.
- Input limited audit was Stage 4A-6.10
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a610_prediction_uncertainty_offline_audit`.
  The map_predict artifact-saving contract was updated so future prediction
  calls can save compact dense uncertainty artifacts. Full `class_prob` remains
  off by default.
- Dense result:
  `logical_frame_count=30`, `physical_map_predict_regeneration_calls=30`,
  dense compact fields generated for all 30 frames, and
  `candidate_uncertainty_rows=480`. Confidence/entropy/margin candidate means
  are `0.8564193916817506`, `0.2142625541271021`, and
  `0.7945013785113891`.
- Dense audit readiness:
  `candidate_level_uncertainty_ready=true` and
  `uncertainty_aware_expert_pilot_ready=true`. Stage 4A-6.11 was not executed.
  Source-occ-free vs uncertainty was computed in dense mode:
  Stage 6.8/6.9 frame1 Pearson `0.037934232555910705`; Stage 6.9 frame2
  Pearson `-0.109843280729622`. Branch-class uncertainty is now available;
  dense frame2 `distinct_nonmeasured_branch` had mean uncertainty
  `0.2738123838789761`, `local_jitter` `0.14163060652624285`, and
  `same_as_measured` `0.07121249474585056`.
- Safety stayed closed for Stage 4A-6.10a:
  no Isaac startup, no capture, no action, no rollout, no long rollout, no
  training, no BC/IL/RL/GDPO/PPO, no prediction writeback, and no source/fixed
  USD, checkpoint, prior dataset, old 6.10 output, or observed_state
  modification. Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a610a_dense_uncertainty_artifacts_test.log`.
- Recommended next:
  Stage 4A-6.11 uncertainty-aware lambda pilot design, bounded one-action only,
  not rollout. Do not jump to long rollout.

- Stage 4A-6.10 prediction uncertainty offline audit is complete in
  `summary_only_limited` mode. Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a610_prediction_uncertainty_offline_audit`.
- Inputs loaded:
  Stage 4A-6.8
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot`
  and Stage 4A-6.9
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot`.
  Optional Stage 4A-6.4 calibration and Stage 4A-6.2 diagnostics context were
  present and recorded. Fixed USD and checkpoint hashes matched prior reports.
- Artifact inventory:
  `prediction_artifacts_found=189`, dense probability/confidence/logit
  artifacts `0`, frames analyzed `30`, top-candidate rows analyzed `480`,
  and candidate-level uncertainty rows `0`.
- Uncertainty result:
  confidence, entropy, and margin were not available in the existing 6.8/6.9
  artifacts. `map_predict_summary` records `prediction_summary_only=true` and
  the dense prediction NPZ paths were removed after summary. Confidence
  mean/range, entropy mean/range, margin mean/range, low-confidence fractions,
  and high-entropy fractions are therefore `not_available_summary_only`.
- Available score/count summaries:
  candidate `source_occ_free` mean/range `48.32083333333333 / 13..85`;
  selected lambda48 `source_occ_free` mean/range `53.6 / 36..78`;
  prediction density mean/range
  `0.025579121979121978 / 0.021895323895323896..0.02678078078078078`.
  Frame2 minus Frame1 predicted-unmeasured count delta mean/range was
  `1467.9 / -9142..13754`.
- Relationship audits:
  `source_occ_free` vs uncertainty, branch class vs uncertainty,
  `candidate_all_local` vs low-confidence/high-entropy, local_jitter vs
  uncertainty, distinct_nonmeasured_branch vs uncertainty, and shadow
  uncertainty scores are blocked by missing dense confidence/probability
  fields. The audit keeps `source_occ_free` separate from uncertainty.
- Readiness:
  `uncertainty_feature_extraction_complete=true`,
  `candidate_level_uncertainty_ready=false`, and
  `uncertainty_aware_expert_pilot_ready=false`. Main blocker:
  `blocked_missing_dense_prediction_probability_fields`; candidate visibility
  voxel probability lists are also absent.
- Safety stayed closed for Stage 4A-6.10:
  no Isaac startup, no capture, no map_predict, no SSCNet inference, no action,
  no rollout, no long rollout, no training, no BC/IL/RL/GDPO/PPO, no
  prediction writeback, and no source/fixed USD, checkpoint, source
  observed_state, or 6.8/6.9 dataset modification. Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a610_prediction_uncertainty_offline_audit_test.log`.
- Recommended next:
  update a future map_predict artifact-saving contract to persist dense
  probability/confidence/entropy/margin fields and candidate-visible voxel
  probability references, then rerun the offline uncertainty audit. Do not
  jump to long rollout.

- Stage 4A-6.9 bounded two-frame lambda48 pilot is complete and validated.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot`.
- Runtime/result:
  `start_count=10`, `frame_count=20`, `capture_count=20`,
  `executed_action_count=10`, `map_predict_calls=20`,
  `sscnet_inference_called=true`, `predictor_loaded_once=true`,
  `exactly_one_action_per_start=true`, `second_action_count=0`,
  `third_frame_count=0`, `continuous_rollout_executed=false`, and
  `long_rollout_executed=false`.
  Isaac startup count is recorded as `1`: the first run completed all 20
  frame captures, then `simulation_app.close()` hung. The process was
  terminated only after the captures were finalized on disk; the recovery run
  reused those captures and did not start Isaac a second time.
- Lambda48 formula stayed:
  `gain_exp / cost + 48 * minmax(source_occ_free)`, with `lambda_sc=48`.
  Prediction stayed read-only and was used only for the lambda48 information
  bonus, not for traversability, collision, ray blocking, candidate validity,
  edge validity, target/ground-truth scoring, or future-observed scoring.
- Frame1 lambda48 vs measured shadow:
  `same_as_measured=4`, `local_jitter=4`,
  `distinct_nonmeasured_branch=2`, `no_valid_candidate=0`,
  `low_cost_artifact=0`, and `historical_prior_basin=0`.
  Stage 4A-6.9 Frame1 reproduced Stage 4A-6.8 lambda48 decisions for all
  10 starts: `frame1_reproduced_stage4a68_count=10`,
  mean action delta `0.0m`, mean yaw delta `0.0rad`.
- Frame2 diagnostic lambda48 vs measured shadow:
  `same_as_measured=1`, `local_jitter=7`,
  `distinct_nonmeasured_branch=2`, `no_valid_candidate=0`,
  `low_cost_artifact=0`, and `historical_prior_basin=0`.
  Frame2 remained diagnostic only; no second action was executed.
- Observed delta and stability:
  total newly observed voxels from Frame1 to Frame2 were `132834`, mean
  `13283.4`, min `3894`, and max `22129`. Frame2 candidate health passed with
  min candidate count `17`, mean candidate count `58.8`, and
  `no_valid_candidate_count=0`. Two-frame stability audit passed and reported
  `unsafe_extension_suggested=false` and `frame2_regression=false`.
- Safety and quality:
  `prediction_safety_audit.passed=true`, `dataset_integrity_report.passed=true`,
  `safety_audit.passed=true`, `expert_data_quality_audit.passed=true`, and
  `two_frame_stability_audit.passed=true`. The only quality warning class is
  `candidate_all_local`. Checkpoint, source USD, fixed USD, and source
  observed_state are unchanged.
- Main artifacts:
  dataset `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot/expert_dataset_two_frame.npz`,
  manifest `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot/expert_dataset_manifest.jsonl`,
  HTML `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot/expert_two_frame_index.html`,
  flythrough `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot/expert_two_frame_flythrough.mp4`,
  and comparison reports
  `stage4a69_vs_stage4a68_comparison.*` and
  `stage4a69_vs_stage4a67_comparison.*`.
- Stage 4A-6.9 vs Stage 4A-6.7:
  Frame1 lambda48 action changed count remains `4`, mean action distance is
  `0.3074937611088073m`, and mean yaw delta is `0.6706520898196431rad`.
- Gates remain closed:
  no long rollout, no second action, no third frame, no full expert dataset,
  no RL/GDPO/PPO/BC/IL, no training, no replay buffer, no policy checkpoint,
  no prediction writeback, and no USD/source observed_state modification.
- Recommended next:
  if this 6.9 evidence is accepted, choose BC dataset design/preparation or an
  explicitly approved short rollout. Do not jump directly to long rollout
  unless explicitly approved; when long rollout starts, it must include expert
  data quality visualization and audit outputs.

- Stage 4A-6.8 map_predict/lambda48 expert pilot is complete and validated.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot`.
- Runtime/result:
  `sample_count=10`, `capture_count=10`, `map_predict_calls=10`,
  `sscnet_inference_called=true`, `predictor_loaded_once=true`,
  `exactly_one_headless_capture_per_start=true`, and
  `exactly_one_action_per_start=true`.
  Isaac startup count is recorded as `1`: the first run completed all 10
  start captures, then `simulation_app.close()` hung. The process was
  terminated only after the captures were on disk; the recovery run reused
  those captures and did not start Isaac a second time.
- Inputs:
  fixed USD
  `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd`,
  corrected camera/start package
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_camera_pose_fix`,
  and Stage 4A-6.7 measured-only pilot
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a67_measured_only_expert_pilot`.
  The same 10 interior start variant IDs were used.
- Lambda48 formula:
  `gain_exp / cost + 48 * minmax(source_occ_free)`, with `lambda_sc=48`.
  Min-max scope is per start over valid candidate/yaw scored rows.
  `source_occ_free` is the raw count of visible predicted-unmeasured voxels
  from the read-only map_predict layer; prediction is not used for
  traversability, collision, ray blocking, candidate validity, or edge
  validity.
- Lambda48 vs measured shadow:
  `same_as_measured=4`, `local_jitter=4`,
  `distinct_nonmeasured_branch=2`, `no_valid_candidate=0`,
  `low_cost_artifact=0`, and `historical_prior_basin=0`.
  Stage 4A-6.8 vs Stage 4A-6.7 action changed count is `4`, mean action
  distance is `0.3074937611088073m`, and mean yaw delta is
  `0.6706520898196431rad`.
- Safety and quality:
  `prediction_safety_audit.passed=true`, `dataset_integrity_report.passed=true`,
  `safety_audit.passed=true`, `expert_data_quality_audit.passed=true`.
  The only quality warning class is `candidate_all_local`; all 10 samples are
  explainable. Checkpoint, source USD, fixed USD, and source observed_state are
  unchanged.
- Main artifacts:
  dataset `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot/expert_dataset.npz`,
  manifest `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot/expert_dataset_manifest.jsonl`,
  HTML `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot/expert_pilot_index.html`,
  flythrough `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot/expert_action_flythrough.mp4`,
  and comparison reports
  `lambda48_vs_measured_comparison.*` and
  `stage4a68_vs_stage4a67_comparison.*`.
- Gates remain closed:
  no long rollout, no second action, no third frame, no full expert dataset,
  no RL/GDPO/PPO/BC/IL, no training, no replay buffer, no policy checkpoint,
  no prediction writeback, and no USD/source observed_state modification.
- Recommended next:
  inspect the 6.7 vs 6.8 comparison and the expert data quality visual package,
  then decide whether to run a bounded two-frame pilot or prepare BC dataset
  design. Do not jump directly to long rollout unless explicitly approved.

- Stage 4A-6.6c-usd-download-official-isaac-deps is complete as a blocked
  dependency-download/localization + one-retry attempt. New script:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/download_stage4a66c_official_isaac_deps.py`.
  New validation script:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66c_official_isaac_deps.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_download_official_isaac_deps`.
- Download/local package result: all 67 exact initial official Isaac URLs from
  `dependency_package_request.md` are present/valid. The final local package
  under
  `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment/dependencies`
  contains 278 files total: 67 USD, 23 MDL, 187 PNG, and 1 DDS, with package
  size 511,952,260 bytes. No random assets, substitutes, procedural fallback,
  cuboid fallback, or old `larger_complex_scene_v1` assets were used.
- Localized package:
  `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized`.
  Localized USD:
  `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized/home_like_scene_v1.usd`.
  The root USD was exported/patched via PXR/Sdf text export; source USD and
  original staged USD hashes stayed unchanged:
  `11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b`.
  Post-patch dependency scan reports remote official refs `0`,
  `omniverse://` refs `0`, unsupported external deps `0`, and unresolved local
  deps `0`.
- Isaac retry gate became allowed by dependency closure and was attempted
  exactly once. The retry failed before RGB/depth capture because the localized
  USD has no `defaultPrim`; Kit reported an unresolved reference prim path
  `@.../current_environment_localized/home_like_scene_v1.usd@<defaultPrim>`
  for `/World/HomeLikeSceneV1`. No validation RGB/depth,
  `observed_state_final.npy`, visual HTML/MP4, or flythrough success package
  was fabricated.
- Validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_download_official_isaac_deps_test.log`
  reports `all_passed: true` for context loading, previous dependency blocker,
  67 initial URLs, manifests, hash preservation, localized package/patch
  reports, blocked retry evidence, and negative scope.
- Gates remain closed:
  `human_visual_inspection_done=false`,
  `formal_expert_sampling_ready=false`,
  `full_expert_dataset_ready=false`, `stage4a66d_executed=false`, and
  `stage4a67_executed=false`. No rollout, selected action, expert sampling,
  map_predict, SSCNet inference, prediction NPZ, replay buffer, checkpoint
  modification, or RL/GDPO/PPO/BC/IL was run.
- Current main blocker: dependency closure is now complete, but Isaac loading
  is blocked by the localized USD missing a `defaultPrim` / concrete spawn
  target. Next faithful step is a localized USD defaultPrim or spawn-path fix
  followed by a separately authorized validation attempt. Do not enter
  Stage 4A-6.6d or Stage 4A-6.7 yet.

- Stage 4A-6.6c-usd-dependency-fix-env-corrected re-ran the dependency
  localization search using the user-corrected conda environment
  `env_isaaclab` (the task/report namespace still says `env_isaacsim`).
  New script:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/localize_stage4a66c_usd_dependencies_env_isaacsim.py`.
  New validation script:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66c_usd_env_isaacsim_dependency_fix.py`.
  New reports are in:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_env_isaacsim_dependency_fix`.
- Probe result: `env_isaaclab` Python is
  `/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python`, Python 3.11.15.
  `isaacsim` import succeeds, `omni` namespace import succeeds, direct `pxr`
  import fails in the bare environment, and `pxr` succeeds only when
  `omni.usd.libs` is added through `PYTHONPATH`/`LD_LIBRARY_PATH`.
  IsaacSim install roots found:
  `/home/ubuntu22/miniconda3/envs/env_isaaclab` and
  `/home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311`.
- Re-search result: all 67 unique remote
  `https://omniverse-content-production.../Assets/Isaac/4.5/Isaac/...` USD
  dependencies remain unresolved. No trusted local
  `Assets/Isaac/4.5/Isaac/...` asset root was found in `env_isaaclab`,
  IsaacSim install roots, or Omniverse/cache roots. There were 0 exact local
  matches, 0 copied dependencies, 0 localized USD patches, and no Isaac retry.
  Source/staged SHA256 remained unchanged:
  `11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b`.
- Validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_usd_env_isaacsim_dependency_fix_test.log`
  reports `all_passed: true` for the 36 required blocker/search outputs,
  previous blocked context, hash preservation, retry gate closure, and
  negative scope. No validation RGB/depth, no `observed_state_final.npy`, no
  MP4, no rollout, no expert sampling, no map_predict, no prediction NPZ, and
  no RL/GDPO/PPO/BC/IL artifacts were produced.
- Current main blocker: `env_isaaclab` also does not contain the required
  local Isaac assets. The USD is still not self-contained. Stage 4A-6.6d and
  Stage 4A-6.7 remain blocked until a complete dependency package is provided
  or the user explicitly allows downloading the exact missing URLs.

- Stage 4A-6.6c-usd-dependency-fix completed a local-only dependency
  localization audit for the staged USD. New script:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/localize_stage4a66c_usd_dependencies.py`.
  New reports are in:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_home_like_scene_validation`.
  The expanded audit found 99 remote reference occurrences, 67 unique remote
  Omniverse/S3 USD assets, 0 sublayers, 0 payloads, 0 material/texture asset
  paths, 0 absolute paths, and 0 local dependency candidates after searching
  the requested roots with allowed USD/mesh/image extensions.
- No dependency files were copied, no staged USD path patches were applied,
  and source/staged SHA256 still match:
  `11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b`.
  Because dependencies remain missing, Isaac headless retry was not allowed
  and was not executed. Package request:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_home_like_scene_validation/dependency_package_request.md`.
  Stage 4A-6.6d and Stage 4A-6.7 remain blocked.
- Stage 4A-6.6c-usd-import attempted the user-provided project candidate
  environment `/home/ubuntu22/sc_explorer_ws/building_scene.usd` as the active
  replacement for the deleted generated `home_like_scene_v1` package.
- The USD was staged without modification to:
  `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment/home_like_scene_v1.usd`.
  Source and staged sha256 match:
  `11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b`.
- `home_like_scene_v1` is registered in
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py` as a staged-USD
  loader via `build_home_like_scene_v1_from_usd` /
  `build_home_like_scene_v1`. It uses
  `HOME_LIKE_SCENE_V1_STAGED_USD`, does not procedurally generate a scene, and
  has no cuboid fallback or asset-download fallback. The old
  `larger_complex_scene_v1` builder remains a disabled `RuntimeError` stub.
- Offline USD inspection succeeded without starting Isaac: Z-up,
  `metersPerUnit=1.0`, 758 prims, 99 references, 67 unresolved/missing
  external dependencies, 1 mesh prim, 2 materials, and 0 texture references.
  Semantic name guesses include home/interior categories such as
  `sofa_couch`, `table`, `bed`, `bathroom`, `cabinet_shelf`,
  `chair_armchair`, `door_window`, `hallway_corridor_stair_elevator`,
  `room`, `floor`, and `wall`.
- Isaac headless was started exactly once for Stage 4A-6.6c-usd-import. It
  failed while resolving/loading the staged USD with:
  `LLVM ERROR: out of memory`. The Kit log showed unresolved remote
  Omniverse HTTPS dependencies from the USD. No second Isaac validation attempt
  was made.
- Because the single Isaac validation was blocked, no validation RGB/depth,
  inspection RGB/depth, `observed_state_final.npy`, measured observed-state
  fusion, or `usd_scene_flythrough.mp4` was produced. The output directory
  contains offline USD reports, start/pose proposals, topdown PNGs, blocker
  reports, a blocked HTML package, and closed manual-review gates:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_home_like_scene_validation`.
- Logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_usd_home_like_scene_validation.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_usd_home_like_scene_py_compile.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_usd_home_like_scene_validation_test.log`.
  The updated blocked-mode validation test reports `all_passed: true` for the
  blocker evidence, staging, scene-factory registration, offline reports,
  manual gates, and negative scope.
- Current gates remain closed:
  `human_visual_inspection_done=false`,
  `user_needs_to_review_visuals=true`,
  `formal_expert_sampling_ready=false`,
  `full_expert_dataset_ready=false`, and `stage4a67_executed=false`.
- Safety/negative scope stayed clean:
  no procedural scene generation, no asset download, no rollout, no expert
  sampling, no selected action execution, no map_predict, no SSCNet inference,
  no prediction NPZ, no replay buffer, no checkpoint creation/modification,
  and no RL/GDPO/PPO/BC/IL.
- Next faithful step:
  provide a self-contained/local dependency package for the USD, or a lighter
  USD with all dependencies resolved locally, then rerun Stage 4A-6.6c import
  and the one-Isaac validation before any Stage 4A-6.6d audit. Stage 4A-6.7
  remains blocked.

- Stage 4A-6.6c-build-v2 generated `home_like_scene_v1` scene package was
  deleted at user request after the IsaacSim top-down render was shown.
  The output directory is absent:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation`.
- The temporary IsaacSim top-down renderer script was also deleted:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/render_stage4a66c_home_like_sim_topdown.py`.
- The deleted 6.6c-build-v2 visual package, RGB/depth captures,
  `observed_state_final.npy`, downloaded/converted Kenney assets, manifests,
  HTML, MP4, and top-down render are no longer available on disk and must not
  be used as inputs for later stages.
- The old `larger_complex_scene_v1` is rejected and disabled. Its old 6.6 /
  6.6a / 6.6b output directories are absent, and
  `build_larger_complex_scene_v1` in
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py` is a disabled
  stub that raises `RuntimeError`. Stage 4A-6.7 must not use the old larger
  scene or its retired output bundles.
- Deletion negative scope stayed clean:
  no rollout, no expert sampling, no selected action execution, no
  map_predict, no SSCNet inference, no prediction NPZ, no replay buffer, no
  policy checkpoint, no checkpoint modification, and no RL/GDPO/PPO/BC/IL.
- Manual review gate remains closed because the scene package was deleted:
  `human_visual_inspection_done=false`,
  `formal_expert_sampling_ready=false`. A replacement scene package must be
  built before any Stage 4A-6.6d audit or later formal expert sampling.

- Stage 4A-6.6b `larger_complex_scene_v1` GUI / visual inspection setup is
  complete and validated. Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection`
- Stage 4A-6.6b created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/inspect_stage4a66b_gui_visual_environment.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66b_gui_visual_environment.py`.
- GUI status:
  DISPLAY was present as `:1`, WAYLAND_DISPLAY was `wayland-0`, and
  XAUTHORITY was `/run/user/1000/.mutter-Xwaylandauth.IRQ0P3`. A bounded
  GUI-mode IsaacSim probe was attempted once, but it did not confirm a visible
  user-inspectable GUI (`gui_attempt_status: failed`; user visibility
  unconfirmed). The log recorded DRI3 presentation warnings, so the run fell
  back to the visual package path.
- Fallback visual package:
  Isaac headless rendering succeeded for 24 inspection views. Validation
  confirmed 24/24 nonblank RGB views and 24/24 finite-positive depth views,
  plus topdown maps, labeled room/corridor/opening/obstacle/start/validation
  plots, warning-region plots, closeups, and an MP4 flythrough:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection/larger_complex_scene_v1_flythrough.mp4`.
  HTML index:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection/visual_inspection_index.html`.
  Human review checklist:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection/human_visual_review_checklist.md`.
- Manual review gate:
  `human_visual_inspection_done: false`,
  `user_needs_to_review_visuals: true`, and visual approval is required before
  Stage 4A-6.7.
- Stage 4A-6.6b safety/negative scope:
  no expert sampling, no expert dataset, no selected action execution, no
  rollout, no open-ended loop, no map_predict, no SSCNet inference, no
  prediction NPZ, no replay buffer, no policy checkpoint, no checkpoint
  modification, no observed_state modification, no target/ground-truth/future
  observed scoring, no external source build, and no RL/GDPO/PPO/BC/IL.
  Stage 4A-6.7 was not executed, and full expert dataset collection remains
  blocked.
- Recommended next step:
  if the user approves the visual package, Stage 4A-6.7 should be a bounded
  formal expert sampling pilot, measured-only first. If the user rejects the
  visuals, run Stage 4A-6.6c scene visual revision / scene editing. Long-term
  GDPO remains future direction only.

- Stage 4A-6.6a `larger_complex_scene_v1` offline scene complexity audit is
  complete and validated. Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66a_scene_complexity_audit`
- Stage 4A-6.6a created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/audit_stage4a66a_scene_complexity.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66a_scene_complexity_audit.py`.
- Audit decision:
  `scene_complexity_audit_passed: true`,
  `scene_ready_for_formal_expert_sampling_pilot: true`, and
  `formal_expert_sampling_ready_full_dataset: false`.
  All nine audit categories passed: scale, topology, starts, fixed views,
  observed_state, frontier/reachability, obstacle/occlusion, expert usability,
  and safety/negative-scope. Hard blockers: none.
- Main warnings:
  some starts share close topology regions despite good Euclidean spread;
  fixed validation views do not directly name `corridor_east_spur` and
  `room_j`; fixed-view observed_ratio is intentionally low with unknown space
  remaining; measured-only fixed views create multiple observed-free
  components; spur rooms have higher obstacle density; Stage 4A-6.7 should
  start measured-only before any lambda48 read-only map_predict pilot.
- Validation:
  `python sim_explorer/test_stage4a66a_scene_complexity_audit.py --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66a_scene_complexity_audit --stage4a66_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66_larger_complex_scene_v1_validation --expect_no_isaac --expect_no_capture --expect_no_rollout --expect_no_formal_expert_sampling --expect_no_map_predict --expect_no_rl_gdpo`
  reported `all_passed: true`.
- Stage 4A-6.6a was offline audit only:
  no Isaac startup, no RGB/depth capture, no selected action execution, no
  rollout, no open-ended loop, no formal expert sampling, no expert dataset,
  no transitions.jsonl, no map_predict, no SSCNet inference, no prediction
  NPZ, no prediction writeback/fusion, no checkpoint changes, no replay
  buffer, and no RL/GDPO/PPO/BC/IL.
- Recommended next faithful step:
  Stage 4A-6.7 bounded formal expert sampling pilot design/execution,
  measured-only first, small/qualified start subset or all qualified starts,
  not full dataset. Long-term GDPO remains future direction only.

- Stage 4A-6.6 `larger_complex_scene_v1` construction and fixed-view
  validation is complete and validated. This was scene construction and
  validation only: exactly one successful Isaac headless startup in the clean
  validation run, 14 fixed validation RGB/depth captures, measured-only
  observed_state integration, no action execution, no rollout, no open-ended
  loop, no formal expert sampling, no map_predict call, no SSCNet inference,
  no prediction NPZ, no prediction writeback/fusion, no target/ground-truth/
  future-observed planning or scoring, no replay buffer, no policy checkpoint,
  and no training/RL/GDPO/PPO/BC/IL.
- Stage 4A-6.6 output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66_larger_complex_scene_v1_validation`
- Stage 4A-6.6 created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/build_stage4a66_larger_complex_scene_v1.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66_larger_complex_scene_v1.py`.
  It also extended
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py` with
  `build_larger_complex_scene_v1`.
- Larger scene construction summary:
  deterministic `scene_seed=0`, bounds x/y `[-12, 12]`, z `[0, 3]`,
  voxel size `0.1`, expected observed_state shape `(240, 240, 30)`,
  10 rooms, 7 corridors, 21 openings, 69 wall cuboids, 52 obstacle cuboids,
  9 start variants, 14 fixed validation camera poses, metadata graph cycle
  rank 5, and preliminary complexity targets met.
- Fixed capture validation:
  14/14 RGB views were nonblank and 14/14 depth views had finite positive
  depth. The final measured-only observed_state shape was `(240, 240, 30)`,
  with observed_ratio `0.09458275462962963`, observed_count `163439`,
  free_count `154672`, occupied_count `8767`, unknown_count `1564561`, and
  invalid_label_count `0`.
- Validation passed:
  `python sim_explorer/test_stage4a66_larger_complex_scene_v1.py --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66_larger_complex_scene_v1_validation`
  reported `all_passed: true`.
- Stage 4A-6.6 outcome:
  `larger_complex_scene_v1_constructed_and_fixed_capture_validated`. This is
  not a scene complexity audit pass and does not make expert sampling ready.
  Formal expert sampling remains blocked until Stage 4A-6.6a
  `scene_complexity_audit` passes. The recommended next faithful step is
  Stage 4A-6.6a scene complexity audit using the 6.6 audit input bundle.

- Stage 4A-6.5av `start_room_b` tree_seed `0` bounded two-frame one-action
  lambda48 runtime smoke is complete and validated. This was a real bounded
  runtime smoke: exactly one Isaac startup in the clean run, exactly two
  frames, exactly two map_predict calls, exactly one selected action execution,
  no second action, no third frame, no rollout, no open-ended loop, no formal
  expert sampling, no training/RL/GDPO/PPO/BC/IL, no checkpoint changes, and
  no existing observed_state or prediction NPZ modification.
- Stage 4A-6.5av output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65av_start_room_b_bounded_smoke`
- Stage 4A-6.5av created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65av_start_room_b_bounded_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65av_start_room_b_bounded_smoke.py`.
- Stage 4A-6.5av logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65av_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65av_start_room_b_bounded_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65av_start_room_b_bounded_smoke_test.log`.
- Runtime setup matched Stage 4A-6.5au design and metadata:
  `medium_three_rooms`, scene seed `0`, start variant `start_room_b`, pose
  `[2.75, -2.55, 1.2]`, yaw `2.7052603405912112`, repeat variant
  `start_room_b_tree_seed0`, formula
  `gain_exp / cost + 48 * minmax(source_occ_free)`, measured-only shadow, and
  lambda32 shadow.
- Frame 1:
  measured-only shadow `n0001 -> n0053`, lambda48 primary `n0001 -> n0053`,
  lambda32 shadow `n0001 -> n0053`; lambda48 classification
  `same_as_measured`, low-cost artifact `false`, historical prior basin
  `false`, and all pre-action safety gates passed. The single action executed
  to `[2.25, -2.4499999999999997, 1.2]`, yaw `0.588002603547567`.
- Frame 2:
  measured-only shadow `n0167 -> n0167`, lambda48 diagnostic
  `n0002 -> n0200`, lambda32 shadow `n0002 -> n0200`; lambda48
  classification `local_jitter`, low-cost artifact `false`, historical prior
  basin `false`. No second action was executed.
- observed_state/map_predict:
  observed_ratio delta `0.04165740740740741`, newly observed `17996`;
  Frame 1 prediction valid/OCC+FREE `60060 / 53080`, Frame 2
  `52286 / 33383`, density ratio `0.6289186134137151`, both
  `code_consistent_v1`, no explosion/collapse.
- Stage 4A-6.5av outcome:
  `spatially_consistent_healthy_start_room_b`. Prediction stayed read-only and
  information-gain-only with no writeback/fusion, no traversability/collision/
  ray blocking, no candidate sampling or edge-validity use, and no target,
  ground-truth, or future-observed scoring. Validation passed with
  `all_passed: true`.
- Current evidence is clean but still not rollout-ready and not formal expert
  sampling-ready. The required next gate is Stage 4A-6.6
  `larger_complex_scene_v1` construction/validation followed by Stage 4A-6.6a
  scene complexity audit before any formal expert sampling pilot. Long-term
  GDPO remains future direction only.

- Stage 4A-6.5at start_corridor seed0/seed1 repeat-comparison diagnosis and
  next-start design is complete and validated. This was diagnosis/design only:
  no Isaac startup, no RGB/depth capture, no map_predict call, no SSCNet
  inference, no selected action execution, no two-frame runtime execution, no
  rollout, no open-ended loop, no training/RL/GDPO/PPO/BC/IL, no checkpoint
  changes, and no existing observed_state or prediction NPZ modification.
- Stage 4A-6.5at output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65at_start_corridor_seed01_review_next_start_design`
- Stage 4A-6.5at created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/review_stage4a65at_start_corridor_seed01_next_start_design.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65at_start_corridor_seed01_next_start_design.py`.
- Stage 4A-6.5at logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65at_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65at_start_corridor_seed01_review_next_start_design.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65at_start_corridor_seed01_review_next_start_design_test.log`.
- Reverified Stage 4A-6.5aq tree_seed `0` and Stage 4A-6.5as tree_seed `1`:
  both used `start_corridor`, pose `[0.0, -4.45, 1.2]`, yaw
  `1.5707963267948966`, exactly two frames, exactly two map_predict calls,
  exactly one selected action, no second action, no third frame, and no
  rollout. The formula remained
  `gain_exp / cost + 48 * minmax(source_occ_free)`.
- Seed comparison:
  aq Frame1/Frame2 lambda48 stayed `same_as_measured`; as Frame1/Frame2
  lambda48 was `distinct_nonmeasured_branch`. Frame1 selected/best deltas were
  `0.2m` / `1.6881943016134136m`; Frame2 selected/best deltas were
  `0.458257569495584m` / `2.4103941586387903m`. Action pose/yaw deltas were
  `0.20000000000000018m` / `2.7504672066207645rad`.
- observed_state and map_predict stayed sane:
  aq observed_ratio delta `0.012087962962962964`, newly observed `5222`; as
  observed_ratio delta `0.006354166666666667`, newly observed `2745`, so
  as-aq observed_ratio delta was `-0.005733796296296297`. map_predict Frame1
  matched exactly at `61152 / 49164`; Frame2 was aq `52988 / 43828` and as
  `47866 / 41937`, density ratios `0.8914652998128713` vs
  `0.8530021967293141`, both `code_consistent_v1`, no explosion/collapse.
- lambda32/lambda48 agreement:
  Frame1 matched selected/best for both seeds. Frame2 aq matched selected child
  only, while as lambda48 diverged from lambda32/measured; this was classified
  as healthy lambda/tree-seed sensitivity, not a safety regression.
- No low-cost artifact and no historical prior basin appeared. Prediction
  stayed read-only / information-gain-only with no writeback/fusion, no
  traversability/collision/ray blocking, no candidate sampling or edge-validity
  use, and no target/ground-truth/future-observed scoring.
- Combined Stage 4A-6.5at outcome:
  `healthy_distinct_seed1_after_conservative_seed0` (also
  start_corridor seed-sensitive but clean). start_corridor tree_seed `2` was
  not executed and is not automatically next. Current evidence is still not
  rollout-ready and not RL/GDPO-ready.
- Selected future Stage 4A-6.5au next-start design:
  `start_room_b`, pose `[2.75, -2.55, 1.2]`, yaw `2.7052603405912112`, source
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_b_empty_astar/scene_metadata.json`,
  future tree_seed `0`. Future 6.5au should stay bounded: exactly two frames,
  exactly two map_predict calls if action executes, exactly one selected
  action, no second action, no third frame, no rollout, formula
  `gain_exp / cost + 48 * minmax(source_occ_free)`, and `--max_workers 32`.
  The future command sketch is marked `DO NOT RUN IN STAGE 4A-6.5at.` and was
  not executed.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65at_start_corridor_seed01_review_next_start_design_test.log`
  reported `all_passed: true` with 76 required files and 10 plots present.
- Long-term NVIDIA GDPO-style multi-reward decoupled policy optimization
  remains future direction only. RL/GDPO/PPO/BC/IL remains explicitly not next
  until bounded repeats and rollout data are ready.

- Stage 4A-6.5as start_corridor tree_seed `1` bounded repeat-safety runtime
  smoke is complete and validated. This was a real bounded runtime smoke:
  exactly one Isaac startup in the clean run, exactly two frames, exactly two
  map_predict calls, exactly one selected action execution, no second action,
  no third frame, no rollout, no open-ended loop, no training/RL/GDPO/PPO/BC/IL,
  no checkpoint changes, and no existing observed_state or prediction NPZ
  modification.
- Stage 4A-6.5as output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65as_start_corridor_tree_seed1_bounded_smoke`
- Stage 4A-6.5as created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65as_start_corridor_seed1_bounded_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65as_start_corridor_seed1_bounded_smoke.py`.
- Stage 4A-6.5as logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65as_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65as_start_corridor_tree_seed1_bounded_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65as_start_corridor_tree_seed1_bounded_smoke_test.log`.
- Runtime setup:
  `medium_three_rooms`, scene seed `0`, start variant `start_corridor`, pose
  `[0.0, -4.45, 1.2]`, yaw `1.5707963267948966`, pose source
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_corridor_empty_astar/scene_metadata.json`,
  repeat variant `alternate_start_corridor_tree_seed1`, reference Stage
  4A-6.5aq tree_seed `0`, current tree_seed `1`.
- Frame 1:
  measured-only shadow `n0018 -> n0022`, lambda48 primary `n0001 -> n0135`,
  lambda32 shadow `n0001 -> n0135`; lambda48 classification
  `distinct_nonmeasured_branch`, low-cost artifact `false`, historical prior
  basin `false`, and all pre-action gates passed. The single action executed
  to `[0.15000000000000036, -3.9499999999999997, 1.2]`, yaw
  `-0.29145679447786677`.
- Frame 2:
  measured-only shadow `n0036 -> n0106`, lambda48 diagnostic `n0008 -> n0137`,
  lambda32 shadow `n0036 -> n0106`; lambda48 classification
  `distinct_nonmeasured_branch`, low-cost artifact `false`, historical prior
  basin `false`.
- Comparison vs Stage 4A-6.5aq tree_seed `0`:
  Frame1 selected/best deltas `0.2m` / `1.6881943016134136m`, Frame2
  selected/best deltas `0.458257569495584m` / `2.4103941586387903m`,
  action pose delta `0.20000000000000018m`, action yaw delta
  `2.7504672066207645rad`. observed_ratio delta was
  `0.006354166666666667`, which is `-0.005733796296296297` vs aq; newly
  observed `2745`. map_predict remained stable: Frame1 valid/OCC+FREE
  `61152 / 49164`, Frame2 `47866 / 41937`, density ratio
  `0.8530021967293141`, no explosion/collapse, both `code_consistent_v1`.
- lambda32/lambda48 agreement:
  Frame1 matched selected/best. Frame2 diverged: lambda48 `n0008 -> n0137`
  while lambda32/measured-only stayed `n0036 -> n0106`.
- Stage 4A-6.5as outcome:
  `spatially_consistent_healthy_repeat`. It is clean and seed-sensitive in a
  healthy way, but still not rollout-ready and not coverage-improvement
  evidence.
- Prediction stayed read-only and information-gain-only, with no writeback or
  fusion, no traversability/collision/ray blocking, no candidate sampling or
  edge-validity use, no target/ground-truth/future-observed scoring, no
  over-cost runtime primary, and no external source modification/build.
- Hardware:
  `os_cpu_count=32`, requested/actual workers `32/32`,
  OMP/OPENBLAS/MKL/NUMEXPR/VECLIB threads `1/1/1/1/1`, GPU
  `NVIDIA GeForce RTX 5080`, total runtime wall time
  `37.23983290199976s`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65as_start_corridor_tree_seed1_bounded_smoke_test.log`
  reported `all_passed: true`.
- Recommended next small task:
  Stage 4A-6.5at start_corridor seed0/seed1 repeat-comparison diagnosis and
  next-start design only, not rollout. Long-term NVIDIA GDPO-style
  multi-reward decoupled policy optimization remains future direction only;
  RL/GDPO/PPO/BC/IL remains explicitly not next until bounded repeats and
  rollout data are ready.

- Stage 4A-6.5ar alternate-start post-action/two-frame diagnosis and
  repeat-safety review is complete and validated. This was diagnosis/design
  only: no Isaac startup, no RGB/depth capture, no map_predict call, no
  SSCNet inference, no selected action execution, no two-frame runtime
  execution, no rollout, no open-ended loop, no training/RL/GDPO/PPO/BC/IL,
  no checkpoint changes, and no existing observed_state or prediction NPZ
  modification.
- Stage 4A-6.5ar output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ar_alternate_start_post_action_diagnosis`
- Stage 4A-6.5ar created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_stage4a65ar_alternate_start_post_action.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ar_alternate_start_post_action.py`.
- Stage 4A-6.5ar logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ar_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ar_alternate_start_post_action_diagnosis.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ar_alternate_start_post_action_diagnosis_test.log`.
- Stage 4A-6.5ar reverified Stage 4A-6.5aq:
  exactly two frames, exactly two map_predict calls, exactly one selected
  action, no second action, no third frame, and no rollout in the completed
  6.5aq clean run.
- start_corridor pose/yaw consistency passed: pose `[0.0, -4.45, 1.2]`,
  yaw `1.5707963267948966`, matched Stage 4A-6.5ap design and the
  Stage 4A-4 metadata source; distance from canonical start remained
  `4.654299087940095m`.
- Action pose consistency passed: the single 6.5aq action pose
  `[-0.04999999999999982, -3.9499999999999997, 1.2]`, yaw
  `-3.0419240010986313`, matched the Frame2 pose and the Frame1 lambda48
  selected child XY.
- observed_state post-action diagnosis stayed measured-only and sane:
  observed_ratio `0.03149537037037037 -> 0.043583333333333335`, delta
  `0.012087962962962964`, newly observed `5222`, unknown->free `4876`,
  unknown->occupied `346`, occupied->free `0`, invalid labels `0`.
- map_predict stability rechecked cleanly:
  Frame1 valid/OCC+FREE `61152 / 49164`, Frame2 `52988 / 43828`,
  density ratio `0.8914652998128713`, no explosion/collapse, both
  `code_consistent_v1`.
- Tree/branch diagnosis:
  Frame1 lambda48 matched measured-only exactly (`n0001 -> n0104`).
  Frame2 lambda48 shared the measured selected child `n0001` and remained
  `same_as_measured`, while the best descendant differed (`n0127` vs measured
  `n0126`). lambda32/lambda48 matched selected/best on Frame1 and matched
  selected child but not best descendant on Frame2.
- No low-cost artifact and no historical prior basin were found. Prediction
  stayed read-only/information-gain-only, with no prediction writeback/fusion,
  traversability/collision/ray blocking, candidate sampling, edge-validity
  use, target/ground-truth/future-observed scoring, over-cost runtime primary,
  or coverage-improvement claim.
- Stage 4A-6.5ar outcome:
  `clean_same_as_measured`, interpreted as conservative but safe. This is not
  coverage-improvement evidence, not rollout-ready, and not RL/GDPO-ready.
- Stage 4A-6.5ar validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ar_alternate_start_post_action_diagnosis_test.log`
  reported `passed: true` for 68 required files, 10 plots, sequence/safety,
  pose/observed/prediction, tree/branch/outcome, future command, and input
  hash checks.
- Selected future Stage 4A-6.5as:
  start_corridor tree_seed `1` bounded repeat-safety smoke, exactly two
  frames, exactly two map_predict calls if action executes, exactly one
  selected action, no second action, no third frame, no rollout, formula
  `gain_exp / cost + 48 * minmax(source_occ_free)`, and `--max_workers 32`.
  The future command sketch is design-only and begins
  `DO NOT RUN IN STAGE 4A-6.5ar.` It was not executed in 6.5ar.
- Long-term NVIDIA GDPO-style multi-reward decoupled policy optimization
  remains a future direction only. RL/GDPO/PPO/BC/IL remains explicitly not
  next until bounded repeats and rollout data are ready.

- Stage 4A-6.5aq alternate-start bounded two-frame/one-action lambda48 smoke
  at `start_corridor`, `tree_seed=0`, is complete and validated.
- Stage 4A-6.5aq output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65aq_alternate_start_corridor_bounded_smoke`
- Stage 4A-6.5aq created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65aq_alternate_start_bounded_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65aq_alternate_start_bounded_smoke.py`.
- Stage 4A-6.5aq logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aq_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aq_alternate_start_corridor_bounded_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aq_alternate_start_corridor_bounded_smoke_test.log`
- Stage 4A-6.5aq used `medium_three_rooms`, scene seed `0`, start variant
  `start_corridor`, pose `[0.0, -4.45, 1.2]`, yaw
  `1.5707963267948966`, from
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_corridor_empty_astar/scene_metadata.json`.
  The pose matched Stage 4A-6.5ap design and metadata, and is
  `4.654299087940095m` from the canonical start.
- Runtime sequence:
  exactly one Isaac startup in the clean run, exactly two frames, exactly two
  map_predict calls, exactly one selected action execution, no second action,
  no third frame, and no rollout.
- Frame 1:
  measured-only shadow `n0001 -> n0104`, lambda48 primary
  `n0001 -> n0104`, lambda32 shadow `n0001 -> n0104`;
  classification `same_as_measured`, low-cost artifact `false`, historical
  prior basin `false`, and all pre-action gates passed. The single action
  executed to `[-0.04999999999999982, -3.9499999999999997, 1.2]`, yaw
  `-3.0419240010986313`.
- Frame 2:
  measured-only shadow `n0001 -> n0126`, lambda48 diagnostic
  `n0001 -> n0127`, lambda32 shadow `n0001 -> n0126`;
  classification `same_as_measured`, low-cost artifact `false`, historical
  prior basin `false`.
- Observed_state delta was sane and measured-only:
  observed_ratio `0.03149537037037037 -> 0.043583333333333335`, delta
  `0.012087962962962964`, newly observed `5222`, unknown->free `4876`,
  unknown->occupied `346`, occupied->free `0`, invalid labels `0`.
- map_predict remained stable:
  Frame 1 valid/OCC+FREE `61152 / 49164`, Frame 2 valid/OCC+FREE
  `52988 / 43828`, density ratio `0.8914652998128713`, no
  explosion/collapse, both `code_consistent_v1`.
- lambda32/lambda48 agreement:
  Frame 1 matched selected child and best descendant; Frame 2 matched selected
  child but lambda32 best descendant `n0126` differed from lambda48 best
  descendant `n0127`, with both still `same_as_measured`.
- Comparison to Stage 4A-6.5ap design passed. Comparison to canonical-start
  seed0/1/2 was recorded as context only; exact branch/position match is not
  required because the start pose changed.
- Stage 4A-6.5aq outcome:
  `clean_same_as_measured`. Prediction stayed read-only and
  information-gain-only, with no writeback/fusion, no traversability/collision
  or ray blocking use, no candidate sampling or edge-validity use, no
  target/ground-truth/future-observed scoring, no checkpoint changes, no
  external source build, no over-cost runtime primary, and no coverage
  improvement claim.
- Hardware:
  `os_cpu_count=32`, requested/actual workers `32/32`,
  OMP/OPENBLAS/MKL/NUMEXPR/VECLIB threads `1/1/1/1/1`, GPU
  `NVIDIA GeForce RTX 5080`, total runtime wall time
  `39.62141864500154s`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aq_alternate_start_corridor_bounded_smoke_test.log`
  reported `all_passed: true`.
- Current evidence is still not rollout-ready. Long-term NVIDIA GDPO-style
  multi-reward decoupled policy optimization remains future direction only;
  no RL/GDPO/PPO/BC/IL is next until bounded repeats and rollout data are
  ready.
- Recommended next small task:
  Stage 4A-6.5ar alternate-start post-action/two-frame diagnosis and
  repeat-safety review only, not rollout.

- Stage 4A-6.5ap seed0/1/2 repeat-comparison review and alternate-start
  bounded-repeat design is complete and validated.
- Stage 4A-6.5ap was review/design only: no Isaac startup, no RGB/depth
  capture, no map_predict call, no SSCNet inference, no selected action
  execution, no two-frame runtime execution, no rollout, no training/RL/
  GDPO/PPO/BC/IL, no checkpoint changes, and no existing observed_state or
  prediction NPZ modification.
- Stage 4A-6.5ap output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ap_seed012_repeat_review_alternate_start_design`
- Stage 4A-6.5ap created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/review_stage4a65ap_seed012_alternate_start_design.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ap_seed012_alternate_start_design.py`.
- Stage 4A-6.5ap logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ap_seed012_repeat_review_alternate_start_design.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ap_seed012_repeat_review_alternate_start_design_test.log`.
- Stage 4A-6.5ap reverified Stage 4A-6.5ak tree_seed `0`, Stage 4A-6.5am
  tree_seed `1`, and Stage 4A-6.5ao tree_seed `2`: each had exactly `2`
  frames, exactly `2` map_predict calls, exactly `1` selected action, no
  second action, no third frame, and no rollout.
- Seed0/1/2 Frame 1 lambda48 comparison:
  seed0 `n0001 -> n0228` (`distinct_nonmeasured_branch`), seed1
  `n0001 -> n0157` (`same_as_measured`), and seed2 `n0001 -> n0248`
  (`same_as_measured`).
- Seed0/1/2 Frame 2 lambda48 comparison:
  seed0 `n0002 -> n0158`, seed1 `n0001 -> n0214`, and seed2
  `n0003 -> n0227`; all three are `distinct_nonmeasured_branch`.
- Action pose deltas were plausible sub-meter tree_seed variation:
  seed0-vs-seed1 `0.20000000000000018m`, seed0-vs-seed2
  `0.22360679774997896m`, and seed1-vs-seed2 `0.4123105625617663m`.
- observed_state deltas remained positive and measured-only:
  seed0 `0.026840277777777775` / `11595` newly observed, seed1
  `0.015152777777777777` / `6546`, and seed2
  `0.013023148148148148` / `5626`.
- map_predict remained stable:
  Frame 1 valid/OCC+FREE matched exactly at `57382 / 40328`; Frame 2 was
  seed0 `47814 / 30133`, seed1 `37258 / 27254`, and seed2
  `32890 / 24936`, with density ratios `0.7471979765919461`,
  `0.6758083713548899`, and `0.6183296964887919`; all stayed
  `code_consistent_v1` with no explosion/collapse.
- lambda32/lambda48 agreement:
  seed0 matched on both frames, seed1 matched on Frame 1 but diverged on
  Frame 2, and seed2 matched on both frames.
- Stage 4A-6.5ap found no low-cost artifact and no historical prior basin in
  any seed/frame. Prediction remained read-only and information-gain-only,
  with no writeback/fusion, traversability/collision/ray blocking, candidate
  sampling, edge-validity use, target/ground-truth/future-observed scoring,
  over-cost runtime promotion, or coverage-improvement claim.
- Combined seed0/1/2 outcome:
  `seed_sensitive_but_clean`. Seed2 is spatially consistent with seed1 and
  closer than seed1 to seed0 on the Frame 2 selected child, so tree_seed
  sensitivity is reduced but not eliminated. Current evidence is still not
  rollout-ready.
- Stage 4A-6.5ap selected future alternate start `start_corridor` from:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_corridor_empty_astar/scene_metadata.json`
  with pose `[0.0, -4.45, 1.2]`, yaw `1.5707963267948966`, distance
  `4.654299087940095m` from the current canonical start. Future Stage
  4A-6.5aq should use tree_seed `0` first so the next variable is start pose,
  not another tree_seed.
- Future Stage 4A-6.5aq command sketch is written at
  `future_stage4a65aq_command_sketch.md` and begins with
  `DO NOT RUN IN STAGE 4A-6.5ap.` It was not executed in 6.5ap.
- Long-term NVIDIA GDPO-style multi-reward policy optimization is recorded
  only as a future direction. No RL/GDPO/PPO/BC/IL is part of Stage 4A-6.5ap.
- Recommended next small task:
  Stage 4A-6.5aq alternate-start bounded two-frame/one-action lambda48 repeat
  smoke at `start_corridor`, tree_seed `0`, still no rollout.

- Stage 4A-6.5ao bounded repeat-safety smoke, same scene/start with
  `tree_seed=2`, is complete and validated.
- Stage 4A-6.5ao output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ao_bounded_repeat_safety_smoke_tree_seed2`
- Stage 4A-6.5ao created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65ao_bounded_repeat_safety_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ao_bounded_repeat_safety_smoke.py`.
- Stage 4A-6.5ao logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ao_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ao_bounded_repeat_safety_smoke_tree_seed2.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ao_bounded_repeat_safety_smoke_tree_seed2_test.log`
- Stage 4A-6.5ao runtime sequence:
  exactly `1` Isaac startup, exactly `2` frames, exactly `2` map_predict
  calls, exactly `1` selected action execution, no second action, no third
  frame, and no rollout. A first launch attempt failed at GLX initialization
  before capture; the clean run used the validated headless NVIDIA/Vulkan env.
- Stage 4A-6.5ao repeat variant:
  same `medium_three_rooms` scene seed `0`, same start pose
  `[-4.65, -4.65, 1.2]`, yaw `0.38710316317995463`, reference tree_seeds
  `0` and `1`, current tree_seed `2`; only the tree seed was intentionally
  changed.
- Stage 4A-6.5ao Frame 1:
  measured-only shadow `n0001 -> n0248`, lambda48 primary `n0001 -> n0248`,
  lambda32 shadow `n0001 -> n0248`; lambda48 classification
  `same_as_measured`, low-cost artifact `false`, historical prior basin
  `false`, and all pre-action safety gates passed.
- Stage 4A-6.5ao executed exactly one action:
  pose `[-4.25, -4.35, 1.2]`, yaw `2.2142974355881817`.
- Stage 4A-6.5ao Frame 2:
  measured-only shadow `n0126 -> n0186`, lambda48 diagnostic
  `n0003 -> n0227`, lambda32 shadow `n0003 -> n0227`; lambda48
  classification `distinct_nonmeasured_branch`, low-cost artifact `false`,
  historical prior basin `false`.
- Stage 4A-6.5ao observed_state delta was sane:
  observed_ratio `0.0425462962962963 -> 0.05556944444444444`, delta
  `0.013023148148148148`, newly observed `5626`, unknown->free `5078`,
  unknown->occupied `548`, occupied->free `0`, invalid labels `0`.
- Stage 4A-6.5ao map_predict stability passed:
  Frame 1 valid/OCC+FREE `57382 / 40328`, Frame 2 valid/OCC+FREE
  `32890 / 24936`, density ratio `0.6183296964887919`, no
  explosion/collapse, both `code_consistent_v1`.
- Stage 4A-6.5ao repeat comparison against seed0/seed1:
  Frame 1 selected deltas were `0.223606797749979m` vs seed0 and
  `0.41231056256176607m` vs seed1; Frame 2 selected deltas were `0.5m` vs
  seed0 and `0.632455532033676m` vs seed1. Frame 2 best-descendant deltas
  were `4.036087214122113m` vs seed0 and `1.2083045973594573m` vs seed1.
- Stage 4A-6.5ao validation passed:
  required outputs, dual-reference comparison, formula contract, hardware,
  hashes, prediction safety, no-rollout checks, and forbidden-output absence
  all passed.
- Stage 4A-6.5ao outcome:
  repeat classification `spatially_consistent_healthy_repeat`, spatially
  consistent with seed1, seed-sensitive but clean, with no safety regression.
  Current evidence is still not enough for rollout.
- Stage 4A-6.5ao recommendation:
  next small task is Stage 4A-6.5ap repeat-comparison review / alternate-start
  design only. Do not recommend rollout directly.
- Still no rollout, open-ended loop, RL/PPO/BC/IL/GDPO, prediction
  writeback/fusion, prediction traversability/collision/ray blocking,
  target/ground-truth/future-observed scoring, checkpoint changes, external
  source build, runtime planner implementation, over-cost runtime promotion,
  or coverage-improvement claim.

- Stage 4A-6.5an repeat-comparison review and next bounded-repeat design is
  complete and validated. This was review/design only: no Isaac startup, no
  RGB/depth capture, no map_predict call, no SSCNet inference, no selected
  action execution, no two-frame runtime execution, and no rollout.
- Stage 4A-6.5an output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65an_repeat_comparison_and_next_design`
- Stage 4A-6.5an created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/review_stage4a65an_repeat_comparison.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65an_repeat_comparison.py`.
- Stage 4A-6.5an logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65an_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65an_repeat_comparison_and_next_design.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65an_repeat_comparison_and_next_design_test.log`
- Stage 4A-6.5an reverified Stage 4A-6.5ak tree_seed `0` and Stage
  4A-6.5am tree_seed `1` as safety-clean bounded two-frame/one-action smokes:
  exactly two frames, exactly two map_predict calls, exactly one action, no
  second action, no third frame, and no rollout in each run.
- Stage 4A-6.5an Frame 1 comparison:
  tree_seed `0` lambda48 `n0001 -> n0228`
  (`distinct_nonmeasured_branch`) versus tree_seed `1` lambda48
  `n0001 -> n0157` (`same_as_measured`); selected-child delta `0.2m`,
  best-descendant delta `1.2529964086141674m`.
- Stage 4A-6.5an Frame 2 comparison:
  tree_seed `0` lambda48 `n0002 -> n0158` versus tree_seed `1` lambda48
  `n0001 -> n0214`; both are `distinct_nonmeasured_branch`; selected-child
  delta `1.0816653826391969m`, best-descendant delta `4.548626166217664m`.
- Stage 4A-6.5an action/observed comparison:
  action pose delta `0.20000000000000018m`; observed_state delta was
  `0.026840277777777775` / newly observed `11595` for tree_seed `0` and
  `0.015152777777777777` / newly observed `6546` for tree_seed `1`. The lower
  tree_seed `1` second-frame observed delta is plausible from the different
  single action pose and not a label-safety regression.
- Stage 4A-6.5an map_predict comparison:
  Frame 1 valid/OCC+FREE matched exactly at `57382 / 40328`. Frame 2 was
  `47814 / 30133` for tree_seed `0` and `37258 / 27254` for tree_seed `1`;
  density ratios `0.7471979765919461` and `0.6758083713548899`; both remained
  `code_consistent_v1`, read-only, and free of density explosion/collapse.
- Stage 4A-6.5an branch health:
  no low-cost artifact and no historical prior basin in either run/frame.
  Lambda32 matched lambda48 on both tree_seed `0` frames and on tree_seed `1`
  Frame 1; tree_seed `1` Frame 2 lambda32 stayed measured-like while lambda48
  selected the distinct nonmeasured branch.
- Stage 4A-6.5an outcome:
  repeat classification `divergent_but_healthy`, with tree_seed sensitivity
  but no safety regression. Current evidence is still not enough for rollout.
- Stage 4A-6.5an next chosen bounded repeat:
  Stage 4A-6.5ao should be same scene/start with `tree_seed=2`, before moving
  to alternate start, because Frame 2 selected-child delta exceeded `1m`.
- Future Stage 4A-6.5ao design:
  exactly two frames if safety gates pass, exactly two map_predict calls if
  action executes, exactly one selected action, no second action, no third
  frame, no rollout, formula
  `gain_exp / cost + 48 * minmax(source_occ_free)`, measured-only shadow,
  lambda32 shadow, prediction read-only/information-gain-only, and
  `--max_workers 32`. The future command sketch is explicitly marked
  `DO NOT RUN IN STAGE 4A-6.5an`.
- Still no rollout, open-ended loop, RL/PPO/BC/IL, prediction
  writeback/fusion, prediction traversability/collision/ray blocking,
  target/ground-truth/future-observed scoring, checkpoint changes, external
  source build, runtime planner implementation, over-cost runtime promotion,
  or coverage-improvement claim.
- Stage 4A-6.5am bounded repeat-safety smoke, same scene/start with
  `tree_seed=1`, is complete and validated.
- Stage 4A-6.5am used the Stage 4A-6.5ak two-frame/one-action runtime path as
  the delegated execution core, and added repeat-variant/reference comparison
  bookkeeping for Stage 4A-6.5am.
- Stage 4A-6.5am output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65am_bounded_repeat_safety_smoke_tree_seed1`
- Stage 4A-6.5am created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65am_bounded_repeat_safety_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65am_bounded_repeat_safety_smoke.py`.
- Stage 4A-6.5am logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65am_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65am_bounded_repeat_safety_smoke_tree_seed1.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65am_bounded_repeat_safety_smoke_tree_seed1_test.log`
- Stage 4A-6.5am runtime sequence:
  exactly `1` Isaac startup, exactly `2` frames, exactly `2` measured-only
  observed_state updates, exactly `2` map_predict calls, exactly `1`
  selected action execution, no second action, no third frame, and no rollout.
- Stage 4A-6.5am repeat variant:
  same `medium_three_rooms` scene seed `0`, same start pose
  `[-4.65, -4.65, 1.2]`, yaw `0.38710316317995463`, reference tree_seed
  `0`, current tree_seed `1`; only the tree seed was intentionally changed.
- Stage 4A-6.5am Frame 1:
  measured-only shadow `n0001 -> n0157`, lambda48 primary `n0001 -> n0157`,
  lambda32 shadow `n0001 -> n0157`; lambda48 classification
  `same_as_measured`, low-cost artifact `false`, historical prior basin
  `false`, and all pre-action safety gates passed.
- Stage 4A-6.5am executed exactly one action:
  pose `[-4.15, -4.75, 1.2]`, yaw `2.1587989303424653`.
- Stage 4A-6.5am Frame 2:
  measured-only shadow `n0003 -> n0255`, lambda48 diagnostic
  `n0001 -> n0214`, lambda32 shadow `n0003 -> n0179`; lambda48
  classification `distinct_nonmeasured_branch`, low-cost artifact `false`,
  historical prior basin `false`.
- Stage 4A-6.5am observed_state delta was sane:
  observed_ratio `0.0425462962962963 -> 0.057699074074074076`,
  delta `0.015152777777777777`, newly observed `6546`, unknown->free `5893`,
  unknown->occupied `653`, occupied->free `0`, invalid labels `0`.
- Stage 4A-6.5am map_predict stability passed:
  Frame 1 valid/OCC+FREE `57382 / 40328`, Frame 2 valid/OCC+FREE
  `37258 / 27254`, density ratio `0.6758083713548899`, no
  explosion/collapse, both `code_consistent_v1`.
- Stage 4A-6.5am repeat comparison against Stage 4A-6.5ak:
  Frame 1 selected delta `0.2m`, Frame 2 selected delta
  `1.0816653826391969m`, action pose delta `0.20000000000000018m`.
  Repeat outcome is `divergent_but_healthy`.
- Stage 4A-6.5am validation passed:
  required artifacts, formula contract, hardware, hashes, prediction safety,
  no-rollout checks, repeat outputs, and forbidden-output absence all passed.
- Stage 4A-6.5am hash/safety audit passed:
  checkpoint unchanged; referenced Stage 4A-6.5ak/6.5al observed_state and
  prediction NPZ inputs unchanged; generated prediction NPZs unchanged after
  creation; prediction remained read-only and information-gain-only; no
  prediction writeback/fusion, traversability/collision/ray blocking,
  candidate sampling, or edge-validity use.
- Stage 4A-6.5am hardware:
  `os_cpu_count=32`, requested/actual workers `32/32`,
  OMP/OPENBLAS/MKL/NUMEXPR/VECLIB threads `1/1/1/1/1`,
  GPU `NVIDIA GeForce RTX 5080`.
- Stage 4A-6.5am recommendation:
  next small task is another bounded repeat review, likely alternate start or
  `tree_seed=2` design, still exactly two frames/one action. Do not recommend
  rollout directly.
- Still no rollout, open-ended loop, RL/PPO/BC/IL, prediction
  writeback/fusion, prediction traversability/collision/ray blocking,
  target/ground-truth/future-observed scoring, checkpoint changes, external
  source build, runtime planner implementation, over-cost runtime promotion,
  or coverage-improvement claim.
- SSCNet full 50 epoch training is complete.
- Best checkpoint is epoch 26:
  `/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar`
- Best checkpoint was verified with `test.py` before Stage 2A.
- Stage 2A offline inference wrapper is implemented:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/offline_infer_npz.py`
- PredictionLayer wrapper is implemented:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/prediction_layer.py`
- Smoke test passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage2a_prediction_layer_test.log`
- Batch5 offline inference passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage2a_batch5_inference.log`
- Stage 2B strict paper-faithful expert scorer is implemented:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/sc_explorer_paper_expert.py`
- Stage 2B CLI is implemented:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/run_paper_expert_offline.py`
- Stage 2B smoke test is implemented:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_paper_expert.py`
- Stage 2B single-sample and batch5 smoke tests passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage2b_paper_expert_single.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage2b_paper_expert_test.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage2b_paper_expert_batch5.log`
- Stage 2C paper expert dataset generator is implemented:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/generate_paper_expert_dataset.py`
- Stage 2C dataset validator is implemented:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_paper_expert_dataset.py`
- Stage 2C batch5 dataset smoke passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage2c_dataset_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage2c_dataset_test.log`
- Stage 2C smoke output:
  `/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke`
- Stage 2C smoke result:
  5 ok samples, 0 failed samples, 5 per-sample expert `.npz` files,
  `manifest.jsonl`, `metadata.json`, and `combined_smoke.npz`.
- Stage 3A IL Dataset/DataLoader module is implemented:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/`
- Stage 3A dataset smoke test passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage3a_il_dataset_test.log`
- Stage 3A behavior cloning skeleton dry-run passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage3a_bc_dry_run.log`
- Stage 3A smoke result:
  dataset size 5, first item candidate_features `(16, 15)`, batch
  candidate_features `(2, 16, 15)`, logits `(2, 16)`, CE loss computed,
  feature stats saved to
  `/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/feature_stats.npz`.
- Stage 4A-1 Isaac depth observation smoke test is implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/`
- Isaac Lab was found at:
  `/home/ubuntu22/IsaacLab`
- Isaac Sim was found in `env_isaaclab` at:
  `/home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim`
- Official headless empty scene and USD camera/depth tutorial smoke tests passed.
- Minimal depth scene smoke passed and saved depth outputs to:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke`
- Stage 4A-1 depth outputs:
  `depth_000.npy`, `depth_001.npy`, `depth_002.npy`, pose JSON files,
  `camera_info.json`, and `scene_metadata.json`.
- Stage 4A-1 observed voxel map smoke passed:
  shape `(80, 80, 30)`, voxel size `0.1m`, bounds `x,y=[-4,4]`,
  `z=[0,3]`, `unknown_count=143335`, `free_count=44435`,
  `occupied_count=4230`, `observed_ratio=0.2534635416666667`.
- Stage 4A-1 logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a1_env_status.log`
  `/home/ubuntu22/sc_explorer_ws/logs/isaac_empty_scene_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/isaac_sensor_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a1_minimal_depth_scene.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a1_depth_to_voxel.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a1_depth_to_voxel_test.log`
- Stage 4A-1-viz observed map visualization is implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_observed_map.py`
- Stage 4A-1-viz output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke_viz`
- Stage 4A-1-viz generated:
  `depth_grid.png`, `observed_topdown_compare.png`,
  `occupied_voxels_3d_step2.png`, `free_occupied_voxels_3d_step2.png`,
  `slices_step2.png`, `index.html`, and `viz_summary.json`.
- Stage 4A-1-viz observed map counts:
  step0 `unknown=170910`, `free=19335`, `occupied=1755`;
  step1 `unknown=143439`, `free=44515`, `occupied=4046`;
  step2 `unknown=143335`, `free=44435`, `occupied=4230`,
  `observed_ratio=0.2534635416666667`.
- Open3D PLY export was skipped because `open3d` is not installed.
- Stage 4A-1-scene-viz scripted Isaac scene visualization is implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/render_minimal_scene_views.py`
- Stage 4A-1-scene-viz output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_scene_viz`
- Stage 4A-1-scene-viz generated:
  `camera_rgb_000.png`, `camera_rgb_001.png`, `camera_rgb_002.png`,
  `camera_depth_color_000.png`, `camera_depth_color_001.png`,
  `camera_depth_color_002.png`, `scene_overview_rgb.png`,
  `scene_overview_depth_color.png`, `scene_layout_topdown.png`,
  `scene_metadata.json`, and `scene_viz_summary.json`.
- Stage 4A-1-scene-viz camera keys:
  RGB `rgb`, depth `distance_to_image_plane`.
- Stage 4A-1-scene-viz validation:
  required files exist, camera RGB PNGs are true RGB mode, summary JSON has no
  NaN, and RGB/depth images are non-empty.
- Stage 4A-1-scene-viz log:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a1_scene_viz.log`
- Stage 4A-2 simulator observed-map expert step is implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py`
- Stage 4A-2 input observed map:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step2.npy`
- Stage 4A-2 uses `EmptyPredictionLayer` only and keeps prediction out of
  `observed_state`.
- Stage 4A-2 expert output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke`
- Stage 4A-2 generated:
  `expert_step_decision.npz`, `expert_step_decision.json`,
  `expert_step_candidates.jsonl`, `expert_topdown.png`, and
  `expert_score_bar.png`.
- Stage 4A-2 observed map stats:
  shape `(80, 80, 30)`, `unknown_count=143335`, `free_count=44435`,
  `occupied_count=4230`, `observed_ratio=0.2534635416666667`.
- Stage 4A-2 frontier/candidate stats:
  `frontier_count=5929`, `frontier_adjacent_free_count=5876`,
  `candidates=64`, `top_n=16`.
- Stage 4A-2 best candidate:
  `expert_action=0`, candidate id `63`, score `88.83270299135849`,
  `gain_exp=73.0`, `gain_sc=0.0`, `gain_hybrid=73.0`,
  `path_cost=0.8217694333482268`, grid `[51, 38, 14]`,
  world `[1.15, -0.15, 1.45]`, yaw `-0.7030942394487684`.
- Stage 4A-2 logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a2_sim_expert_step.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a2_sim_expert_test.log`
- Stage 4A-2 smoke test passed and verified `observed_state_step2.npy` was not
  modified. No RL, PPO, optimizer, policy training, behavior cloning, IL
  training, SSCNet inference on Isaac depth, NYU target label use, or ground
  truth use was performed.
- Stage 4A-3 multi-step EmptyPredictionLayer expert rollout is implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
- Stage 4A-3 also added a rollout-facing measured-only depth update wrapper:
  `update_observed_state_from_depth(...)` in
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py`.
- Stage 4A-3 rollout output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000`
- Stage 4A-3 generated:
  `step_000.npz` through `step_009.npz`, `transitions.jsonl`,
  `observed_state_step000.npy` through `observed_state_step009.npy`,
  `observed_state_final.npy`, `episode_summary.json`, per-step RGB/depth/pose
  files, `rollout_topdown_path.png`, `observed_ratio_curve.png`,
  `frontier_count_curve.png`, `step_topdown_000.png` through
  `step_topdown_009.png`, `rollout_index.html`, and `viz_summary.json`.
- Stage 4A-3 global manifest:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/manifest.jsonl`
- Stage 4A-3 rollout result:
  episode `minimal_room_empty_pred_000`, `steps_completed=10`,
  `done_reason=max_steps`, `observed_ratio_start=0.0`,
  `observed_ratio_end=0.21754166666666666`,
  `total_delta_observed_ratio=0.21754166666666666`.
- Stage 4A-3 final observed map stats:
  shape `(80, 80, 30)`, `unknown_count=150232`, `free_count=35873`,
  `occupied_count=5895`, `observed_count=41768`.
- Stage 4A-3 final pose:
  `[3.549999952316284, 3.25, 1.2000000476837158]`, final yaw
  `-2.8477112304002925`, repeated pose count `2`.
- Stage 4A-3 expert behavior:
  average `frontier_count=4525.6`, average candidates `64.0`,
  best_score min/mean/max `29.41531522194122 / 105.48766454499457 /
  190.10038228379815`, gain_exp min/mean/max `37.0 / 63.8 / 89.0`,
  gain_sc min/mean/max `0.0 / 0.0 / 0.0`, path_cost min/mean/max
  `0.2 / 0.8726943498167937 / 2.6516798957102083`.
- Stage 4A-3 logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a3_rollout_empty_pred.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a3_rollout_test.log`
- Stage 4A-3 smoke test passed:
  synthetic transition serialization ok, real episode validation ok,
  observed_ratio non-decreasing, EmptyPredictionLayer `gain_sc=0`,
  prediction did not write observed_map, and no RL/optimizer/BC training ran.
- Stage 4A-3.2 medium-complexity scripted Isaac scene is implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/medium_complex_depth_scene.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/render_medium_complex_scene_views.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_medium_complex_scene_metadata.py`
- Stage 4A-3.2 also updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py`
  with CLI bounds arguments `--x_min --x_max --y_min --y_max --z_min --z_max`
  and `observed_state_final.npy` output.
- Stage 4A-3.2 scene summary:
  bounds `x/y=[-6,6]`, `z=[0,3]`, floor `12m x 12m`, wall height `2.2m`,
  rooms `3`, corridors `1`, openings `3`, walls `13`, obstacles `13`,
  fixed camera poses `5`.
- Stage 4A-3.2 smoke output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_smoke`
- Stage 4A-3.2 smoke generated:
  `depth_000.npy` through `depth_004.npy`, `rgb_000.png` through
  `rgb_004.png`, `pose_000.json` through `pose_004.json`,
  `camera_info.json`, `scene_metadata.json`,
  `observed_state_step0.npy` through `observed_state_step4.npy`,
  `observed_state_final.npy`, and `observed_summary.json`.
- Stage 4A-3.2 observed map stats:
  shape `(120, 120, 30)`, unknown/free/occupied
  `339813 / 86064 / 6123`, observed_ratio `0.21339583333333334`.
- Stage 4A-3.2 visualization output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_viz`
- Stage 4A-3.2 visualization generated:
  `scene_overview_rgb.png`, `scene_overview_depth_color.png`,
  `scene_layout_topdown.png`, `camera_rgb_grid.png`,
  `camera_depth_grid.png`, `observed_topdown_compare.png`,
  `free_occupied_voxels_3d_final.png`, `slices_final.png`, and
  `scene_viz_summary.json`.
- Stage 4A-3.2 optional one-step expert smoke passed on
  `observed_state_step4.npy` using EmptyPredictionLayer:
  `frontier_count=20919`, `frontier_adjacent_free_count=21637`,
  `candidates=64`, `top_n=16`, best score `53.62160777611031`,
  best grid `[64, 91, 13]`, best world `[0.45, 3.15, 1.35]`,
  `gain_sc=0.0`.
- Stage 4A-3.2 logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_metadata_test.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_depth.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_depth_to_voxel.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_viz.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_expert_step.log`
- Stage 4A-3.2 validation passed:
  metadata test passed, RGB images nonblank, depth images finite/nonzero,
  observed map has UNKNOWN/FREE/OCCUPIED, observed_ratio > 0.05, obstacle
  count >= 10, room count >= 3, opening count >= 3.
- Stage 4A-3.5 observed-free A* path-cost mode is implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/astar_planner.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_astar_planner.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_astar.py`
- Stage 4A-3.5 also updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
- Stage 4A-3.5 A* design:
  traversability is derived only from `observed_state`; FREE in the
  robot-height band is traversable, OCCUPIED is blocked and inflated by robot
  radius, UNKNOWN is not traversable, and no prediction/ground truth/scene
  metadata is used for path planning.
- Stage 4A-3.5 one-step medium A* expert output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_astar_smoke`
- Stage 4A-3.5 one-step medium A* result:
  traversable/blocked/unknown cells `4316 / 1907 / 8177`,
  reachable/unreachable candidates `12 / 52`, best score
  `51.651363679237036`, best `gain_exp=110.0`, `gain_sc=0.0`,
  best path_cost `2.129663036258191`, best A* path length
  `1.2656854249492382m`, best grid `[64, 91, 13]`, best world
  `[0.45, 3.15, 1.35]`.
- Stage 4A-3.5 one-step visualization generated:
  `expert_topdown.png`, `expert_score_bar.png`, and
  `traversability_topdown.png`.
- Stage 4A-3.5 medium A* rollout output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred/episodes/medium_three_rooms_astar_empty_pred_000`
- Stage 4A-3.5 medium A* rollout result:
  `steps_completed=5`, `done_reason=no_valid_candidate`,
  observed_ratio `0.0 -> 0.04308796296296296`, final unknown/free/occupied
  `413386 / 15863 / 2751`, final pose
  `[0.6500000000000004, 0.6500000000000004, 1.2]`, average reachable
  candidates `18.4`, average best path_cost `0.9421159585855353`.
- Stage 4A-3.5 rollout main blocker:
  at expert step 5 all 64 sampled candidates were unreachable under
  conservative observed-free A* traversability
  (`traversable=338`, `blocked=918`, `unknown=13144`). No Euclidean fallback
  was used.
- Stage 4A-3.5 rollout visualization generated:
  `rollout_topdown_path.png`, `observed_ratio_curve.png`,
  `frontier_count_curve.png`, `reachable_candidates_curve.png`,
  `step_topdown_000.png` through `step_topdown_004.png`, and
  `rollout_index.html`.
- Stage 4A-3.5 tests passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_astar_planner_test.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_sim_expert_astar_test.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_regression_sim_paper_expert_euclidean_test.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_regression_sim_expert_rollout_test.log`
- Stage 4A-3.5 logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_medium_expert_step_astar.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_medium_rollout_astar_empty_pred.log`
- Stage 4A-3.6 reachability-aware A* candidate sampling is implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_reachable_candidate_sampling.py`
- Stage 4A-3.6 also updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/astar_planner.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
- Stage 4A-3.6 adds:
  `connected_component_from_start`, `nearest_traversable_cell`,
  `frontier_reachable_candidate_mask`,
  `compute_reachable_frontier_candidate_cells`,
  `--candidate_sampling_mode frontier|reachable_frontier|auto`,
  `--snap_start_to_traversable`, and `--max_snap_radius_cells`.
- Stage 4A-3.6 behavior:
  A* `auto` candidate sampling resolves to `reachable_frontier`; Euclidean
  `auto` preserves old frontier sampling. UNKNOWN is still not traversable.
  No scene metadata, target labels, simulator ground truth, or prediction
  output is used for reachability.
- Stage 4A-3.6 one-step medium reachable A* output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_astar_reachable_smoke`
- Stage 4A-3.6 one-step medium reachable A* result:
  traversable/blocked/unknown cells `4316 / 1907 / 8177`,
  reachable component count `1196`, reachable frontier-adjacent count `1196`,
  candidate source `reachable_frontier`, reachable/unreachable candidates
  `64 / 0`, `top_n=16`, best score `88.24634362636618`,
  best `gain_exp=66.0`, `gain_sc=0.0`, best path_cost
  `0.7479063413600806`, best A* path length `0.28284271247461906m`,
  best grid/world `[58, 82, 11] / [-0.15, 2.25, 1.15]`.
- Stage 4A-3.6 medium reachable A* rollout output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_reachable_empty_pred/episodes/medium_three_rooms_astar_reachable_empty_pred_000`
- Stage 4A-3.6 medium reachable A* rollout result:
  `steps_completed=10`, `done_reason=max_steps`, observed_ratio
  `0.0 -> 0.10147453703703704`, final unknown/free/occupied
  `388163 / 36017 / 7820`, final pose
  `[0.550000011920929, -0.05000000074505806, 1.2000000476837158]`,
  average reachable candidates `64.0`, average reachable component count
  `238.8`, average reachable frontier-adjacent count `238.8`, and
  `no_valid_candidate_steps=[]`.
- Stage 4A-3.6 visualizations generated:
  one-step `expert_topdown.png`, `expert_score_bar.png`,
  `traversability_topdown.png`; rollout `rollout_topdown_path.png`,
  `observed_ratio_curve.png`, `frontier_count_curve.png`,
  `reachable_candidates_curve.png`, `reachable_component_count_curve.png`,
  `step_topdown_000.png` through `step_topdown_009.png`, and
  `rollout_index.html`.
- Stage 4A-3.6 tests passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_reachable_candidate_sampling_test.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_astar_planner_regression_test.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_sim_expert_astar_reachable_test.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_regression_sim_paper_expert_euclidean_test.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_regression_sim_expert_rollout_test.log`
- Stage 4A-3.6 logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_medium_expert_step_astar_reachable.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_medium_rollout_astar_reachable_empty_pred.log`
- Stage 4A-4 multi-episode EmptyPredictionLayer A* rollout dataset is
  implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_batch.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_rollout_dataset.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_rollout_dataset_batch.py`
- Stage 4A-4 also updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
- Stage 4A-4 dataset output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar`
- Stage 4A-4 batch setup:
  scene_variant `medium_three_rooms`, scene_seeds `0,1,2`, start_variants
  `start_room_a,start_corridor,start_room_b`, intended episodes `9`,
  max_steps `10`, prediction_mode `empty`, path_cost_mode `astar`,
  candidate_sampling_mode `reachable_frontier`, motion_mode `planar`.
- Stage 4A-4 dataset result:
  ok episodes `9`, failed episodes `0`, total transitions `90`,
  steps min/mean/max `10 / 10 / 10`, done_reason counts `max_steps=9`,
  observed_ratio_end min/mean/max
  `0.08587037037037037 / 0.11455118312757204 / 0.16534722222222223`,
  total_delta_observed_ratio min/mean/max
  `0.08587037037037037 / 0.11455118312757204 / 0.16534722222222223`.
- Stage 4A-4 expert statistics:
  average reachable candidates `64.0`, average reachable component count
  `570.3444444444444`, average best_score `163.2387554327081`, average
  gain_exp `49.15555555555556`, average gain_sc `0.0`, average path_cost
  `0.45623051832594874`, no_valid_candidate episodes `0`.
- Stage 4A-4 generated global outputs:
  `manifest.jsonl`, `dataset_summary.json`, `dataset_summary.md`,
  `rollout_dataset_index.html`, `aggregate_observed_ratio_curve.png`,
  `aggregate_observed_ratio_end_bar.png`, `aggregate_steps_completed_bar.png`,
  `aggregate_steps_hist.png`, `aggregate_done_reasons.png`,
  `aggregate_reachable_candidates_curve.png`, and
  `aggregate_no_valid_candidate_stats.png`.
- Stage 4A-4 tests passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_batch_empty_pred_astar.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_dataset_summary.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_dataset_test.log`
- Stage 4A-4 validation:
  observed_ratio non-decreasing, EmptyPredictionLayer `gain_sc=0`,
  prediction did not write observed_map, no target/ground-truth fields, no
  RL/optimizer/BC/IL training, no UNKNOWN traversability shortcut, and no
  Euclidean fallback.
- Stage 4A-5 Isaac single-frame map_predict alignment smoke is implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_sscnet_preprocess.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_prediction_layer.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_map_predict_single.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_isaac_prediction_alignment.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_map_predict_single.py`
- Stage 4A-5 position convention check:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a5_position_convention_check.log`
  confirmed NYU `position` is `(480,640)` int32 flat high-res indices in
  `[0,240*144*240)`, with 0 used for invalid/out-of-volume pixels.
- Stage 4A-5 uses `Project2Dto3D` convention:
  scatter flat `(x_right,y_up,z_forward)` high-res indices into `(240,144,240)`,
  then model output axes are `(z_forward,y_up,x_right)`.
- Stage 4A-5 input:
  dataset `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar`,
  episode `medium_three_rooms_seed0_start_room_a_empty_astar`, step `0`,
  `depth_000.npy`, `pose_000.json`, `camera_info.json`, and
  `observed_state_step000.npy`.
- Stage 4A-5 output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke`
- Stage 4A-5 generated:
  `sscnet_input_debug.npz`, `sscnet_depth_input.npy`,
  `sscnet_position.npy`, `valid_position_mask.npy`,
  `local_prediction.npz`, `global_prediction_layer.npz`,
  `prediction_alignment_summary.json`, `isaac_depth_input.png`,
  `local_prediction_slices.png`, `global_prediction_topdown.png`,
  `observed_vs_prediction_topdown.png`, and
  `prediction_not_measured_topdown.png`.
- Stage 4A-5 run result:
  depth input `(480,640)`, position `(480,640)`,
  valid position pixels `166888`, logits `(1,12,60,36,60)`,
  local prediction `(60,36,60)`, global prediction `(120,120,30)`,
  global valid prediction voxels `56602`, predicted occupied voxels `15664`,
  predicted_unmeasured voxels `39400`, inference time `0.1617s`.
- Stage 4A-5 local prediction stats:
  confidence min/mean/max `0.160744 / 0.722959 / 0.997107`,
  occupied_prob min/mean/max `0.002893 / 0.348689 / 0.997254`.
- Stage 4A-5 tests passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a5_isaac_map_predict_single.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a5_isaac_map_predict_single_test.log`
- Stage 4A-5 validation:
  py_compile passed, observed_state hash unchanged, SimPredictionLayer API ok,
  no target/ground-truth fields in prediction artifacts, no RL/optimizer/BC/IL
  training, no expert/rollout prediction use, no prediction writeback, and no
  prediction use for traversability/collision/A*.
- Stage 4A-5.1 one-step SC-aware expert scoring is implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_with_prediction.py`
- Stage 4A-5.1 updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py`
- Stage 4A-5.1 runner supports `--prediction_mode empty|sim_npz`,
  `--prediction_npz`, `--tau`, and `--episode_summary`.
- Stage 4A-5.1 input:
  observed_state
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar/observed_state_step000.npy`,
  pose
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar/pose_000.json`,
  prediction
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke/global_prediction_layer.npz`,
  tau `0.1`.
- Stage 4A-5.1 empty baseline output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/empty_baseline`
- Stage 4A-5.1 empty baseline best:
  id `11`, score `331.3448560321166`, gain_exp `55.0`, gain_sc `0.0`,
  gain_hybrid `55.0`, path_cost `0.1659902032541859`, grid `[13,13,11]`,
  world `[-4.65,-4.65,1.15]`.
- Stage 4A-5.1 SC prediction output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/sc_prediction`
- Stage 4A-5.1 SC prediction best:
  id `11`, score `662.6897120642332`, gain_exp `55.0`, gain_sc `55.0`,
  gain_hybrid `110.0`, gain_occ `13.0`, gain_conf `19.406008422374725`,
  path_cost `0.1659902032541859`, grid `[13,13,11]`,
  world `[-4.65,-4.65,1.15]`.
- Stage 4A-5.1 comparison:
  best candidate changed `false`, score delta `331.3448560321166`,
  gain_hybrid delta `55.0`, top-N overlap `16/16`, candidates with
  gain_sc > 0 `64/64`, max/mean gain_sc `174.0 / 71.59375`,
  total predicted_unmeasured visible count `4582`.
- Stage 4A-5.1 outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/comparison_summary.json`
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/comparison_summary.md`
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/empty_vs_prediction_best_candidate.png`
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/gain_comparison_bar.png`
- Stage 4A-5.1 visualizations:
  empty and prediction `expert_topdown.png`, `expert_score_bar.png`,
  `traversability_topdown.png`, plus prediction
  `prediction_overlay_topdown.png` and
  `predicted_unmeasured_visible_topdown.png`.
- Stage 4A-5.1 logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_empty_baseline.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_sc_prediction.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_with_prediction_test.log`
- Stage 4A-5.1 validation:
  py_compile passed, `test_sim_expert_with_prediction.py` passed,
  prediction layer shape matches observed_state, observed_state hash unchanged,
  empty mode gain_sc is zero, prediction mode gain_sc is nonzero,
  gain_hybrid equals gain_exp + gain_sc, gain_occ/gain_conf are finite, no
  target/ground-truth leakage, no RL/optimizer/BC/IL training, no rollout, no
  prediction traversability/collision/A*/ray-blocking/writeback.
- Stage 4A-6 short multi-step SC-aware rollout is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_map_predictor.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/compare_sc_pred_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_sc_aware_rollout.py`
- Stage 4A-6 updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
- Stage 4A-6 run:
  episode `medium_three_rooms_seed0_start_room_a_sc_pred_dynamic_000`,
  scene `medium_three_rooms`, seed `0`, start `start_room_a`,
  max_steps `5`, prediction_mode `sim_dynamic`, path_cost_mode `astar`,
  candidate_sampling_mode `reachable_frontier`, motion_mode `planar`.
- Stage 4A-6 result:
  steps_completed `5`, done_reason `max_steps`, observed_ratio
  `0.0 -> 0.05899768518518519`, final counts unknown/free/occupied
  `406513 / 21226 / 4261`, final pose
  `[-4.25, -4.150000095367432, 1.2000000476837158]`.
- Stage 4A-6 SC-aware expert stats:
  average gain_exp `49.6`, gain_sc `49.4`, gain_hybrid `99.0`,
  gain_occ `8.8`, gain_conf `16.96283725500107`, average best_score
  `441.9845465468916`, candidates_with_gain_sc_positive min/mean/max
  `63 / 63.6 / 64`, no_valid_candidate_steps `[]`.
- Stage 4A-6 map_predict performance:
  model_loaded_once `true`, average preprocess_time `0.05369079960000818s`,
  average inference_time `0.020522295199771178s`, average alignment_time
  `0.03251961960013432s`, average map_predict total time
  `0.14326694260016665s`, average expert_time `1.026360238399866s`,
  total_wall_time `19.86559214800036s`, GPU memory peak `794354176` bytes
  on RTX 5080.
- Stage 4A-6 comparison to Stage 4A-4 matching empty baseline:
  compared_steps `5`, empty final observed_ratio `0.06896296296296296`,
  SC final observed_ratio `0.05899768518518519`, SC-empty delta
  `-0.009965277777777774`, changed selected actions `5`, mean score delta
  `233.79287700349096`, mean gain_exp delta `-5.199999999999996`, mean
  SC gain_sc `49.4`. This is an integration-correctness success, not a
  performance improvement claim.
- Stage 4A-6 outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke`
  comparison:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke/comparison_to_empty_baseline`
- Stage 4A-6 logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a6_sc_pred_dynamic_rollout.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a6_sc_vs_empty_comparison.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a6_sc_aware_rollout_test.log`
- Stage 4A-6 validation:
  py_compile passed, `test_sim_sc_aware_rollout.py` passed, observed_ratio
  non-decreasing, gain_sc nonzero, gain_hybrid identity, observed_state hash
  unchanged by prediction at every step, no prediction writeback, no prediction
  traversability/collision/A*/ray-blocking, checkpoint not modified, no
  target/ground-truth leakage, and no RL/optimizer/BC/IL/SSCNet training.
- Stage 4A-6 closeout verification:
  required episode and comparison files exist with nonzero size; log scan found
  no current Traceback/Error/CUDA-unavailable blocker. The manifest preserves
  the first failed attempt (`KeyError: 'observed_state_source'`) and the final
  ok rerun. The issue was fixed by adding observed/depth/pose/camera source
  fields to per-step prediction summaries.
- Stage 4A-6.1 SC-aware rollout analysis and ablation is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/analyze_sc_rollout_behavior.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sc_pred_ablation_sweep.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_sc_pred_ablation.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sc_pred_ablation.py`
- Stage 4A-6.1 updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
- Stage 4A-6.1 scoring controls:
  `sc_gain_weight`, `sc_gain_cap`, and
  `score_gain_mode=hybrid_raw|hybrid_weighted`; default remains
  Stage 4A-6-compatible `hybrid_raw`, weight `1.0`, cap `None`.
- Stage 4A-6.1 existing SC-vs-empty analysis:
  output
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/existing_sc_vs_empty`,
  compared steps `5`, empty final observed_ratio
  `0.06896296296296296`, original SC final observed_ratio
  `0.05899768518518519`, delta `-0.009965277777777774`, first SC lag step
  `1`, changed actions `5/5`, mean path_cost empty/SC
  `0.36998367643136965 / 0.2768163156997422`, mean gain_exp empty/SC
  `54.8 / 49.6`, mean SC gain_sc `49.4`, dense gain_sc candidate steps
  `0..4`.
- Stage 4A-6.1 ablations:
  output `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_ablation`,
  completed configs `dynamic_w025_tau01`, `dynamic_w05_tau01`,
  `dynamic_w1_tau03`, `dynamic_w1_tau01_cap50`, and
  `static_step0_weight_1p0_tau_0p1`; failed configs `[]`.
- Stage 4A-6.1 ablation result:
  all five configs completed 5 steps, all ended at observed_ratio
  `0.05899768518518519`, all remained below the empty baseline by
  `-0.009965277777777774`, and all selected the same `5/5` actions as the
  original SC rollout.
- Stage 4A-6.1 performance:
  dynamic wall time per ablation `27.75568118299998s` to `30.416639261s`,
  dynamic average map_predict inference `0.02070385000006354s` to
  `0.03013971940004012s`, dynamic GPU peak `794296320` bytes; static step0
  wall time `22.448690754999916s`, average inference `0.0s`, GPU peak `None`.
- Stage 4A-6.1 summary and qualitative outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_ablation/summary`
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/qualitative_inspection`
- Stage 4A-6.1 tests:
  py_compile passed and `test_sc_pred_ablation.py` passed; at least two
  ablations completed, observed_ratio is non-decreasing, weighted gain formula
  is correct, prediction remains read-only and information-gain-only,
  checkpoint was not modified, and no RL/optimizer/BC/IL/SSCNet training ran.
- Stage 4A-6.1 conclusion:
  the underperformance appears to be a scoring/calibration/alignment issue, not
  a rollout plumbing failure. Prediction gain is dense, raw hybrid can double
  count unknown regions, and the selected actions stay local/low-path-cost.
  Next inspect map_predict preprocessing, global alignment, confidence
  calibration, and NYU-to-Isaac domain shift before longer SC-aware rollouts.
- Stage 4A-6.2 map_predict preprocessing / alignment / calibration diagnostics
  are complete.
- Stage 4A-6.2 created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_isaac_sscnet_preprocess.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_prediction_global_alignment.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/evaluate_prediction_against_future_observed.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_alignment_variant_sweep.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_map_predict_diagnostics.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_map_predict_diagnostics.py`
- Stage 4A-6.2 diagnostics root:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics`
- Stage 4A-6.2 preprocessing comparison:
  Isaac mean depth `2.532350206375122m`, NYU mean depth
  `2.8481276988983155m`, Isaac valid position ratio `0.565763671875`, NYU
  position nonzero proxy ratio `0.74495458984375`; the valid position ratio
  difference is suspicious.
- Stage 4A-6.2 global alignment sanity:
  mean valid in-front ratio `0.9977230302200312`, mean inside global bounds
  ratio `0.8669629629629629`, mean valid inside expected local volume ratio
  `0.9922601247618591`; direct sanity did not flag a gross behind-camera yaw
  issue, but local volume clips `10800` voxels below floor before bounds.
- Stage 4A-6.2 future observed evaluation:
  future measured maps were used only for post-hoc delayed sensor validation,
  not planning or expert scoring. At tau `0.1`, mean predicted_unmeasured
  `35118.2`, later measured fraction `0.059004217437215026`, occupied
  precision `0.25632042463242544`, free precision `0.9323112341592195`,
  occupied Brier `0.2786559495144023`, ECE-like calibration
  `0.3405436085907938`. Tau `0.1` is too dense, and the tau sweep did not
  reduce density meaningfully without discarding most delayed sensor coverage.
- Stage 4A-6.2 alignment variant sweep:
  best diagnostic variant `xz_swap_variant`, default variant rank `7`,
  Brier improvement vs default `0.0735458940774611`, likely alignment bug
  `true`.
- Stage 4A-6.2 candidate-score decomposition:
  gain_exp/gain_sc correlation `0.9647202023737985`, final_score vs inverse
  path_cost correlation `0.9713818732156227`, gain_sc duplicates gain_exp
  `true`, path_cost dominance `true`, and all five Stage 4A-6.1 ablations
  matched original SC actions `5/5`.
- Stage 4A-6.2 recommendation:
  primary suspected issue is alignment convention; Stage 4A-6.3 should fix or
  reconcile the local prediction to global projection convention and rerun
  Stage 4A-5/5.1/6 smoke. Secondary issue is dense/unselective confidence
  calibration.
- Stage 4A-6.2 tests:
  py_compile passed and `test_map_predict_diagnostics.py` passed; observed
  state hashes unchanged, checkpoint not modified, future observations marked
  evaluation-only, prediction did not affect observed_map, traversability,
  collision, A*, candidate reachability, or ray blocking, and no RL/optimizer
  /BC/IL/SSCNet training ran.
- Stage 4A-6.3 SSCNet alignment convention fix/reconciliation is complete.
- Stage 4A-6.3 created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/document_sscnet_axis_convention.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/fix_prediction_alignment_convention.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_alignment_convention_fix.py`
- Stage 4A-6.3 modified:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_sscnet_preprocess.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_map_predictor.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_map_predict_single.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_isaac_prediction_alignment.py`
- Stage 4A-6.3 axis audit:
  `Project2Dto3D` scatters flat indices, views `(W,H,D)`, then permutes to
  `(D,H,W)`. The raw Python dataloader branch uses
  `np.ravel_multi_index((x,y,z),(240,144,240))`, but the deployed C++/ROS
  projection path uses `z*(240*144)+y*240+x`. The repackaged npz branch loads
  precomputed `position`; `target_lr.T` reverses stored axes before loss
  flattening. Therefore the Stage 4A-5 output-axis assumption
  `(z_forward,y_up,x_right)` followed the raw Python branch, while the
  code-consistent Isaac convention is `code_consistent_v1` with input position
  flatten `(z_forward,y_up,x_right)` and output axes
  `(x_right,y_up,z_forward)`.
- Stage 4A-6.3 outputs:
  axis audit `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a63_alignment_fix/axis_convention_audit`;
  convention eval `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a63_alignment_fix/convention_eval`;
  fixed single-frame `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke_alignment_fixed`;
  fixed one-step expert `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_alignment_fixed_smoke`;
  fixed rollout `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_alignment_fixed_smoke`.
- Stage 4A-6.3 convention eval result:
  `current_default_v0` occupied Brier `0.2786559495144023`,
  `code_consistent_v1` occupied Brier `0.20511005543694122`, Brier improvement
  `0.0735458940774611`, ECE-like `0.3405436085907937 -> 0.22427722861569463`;
  best diagnostic convention `xz_swap_diagnostic`; recommended fixed
  convention `code_consistent_v1`.
- Stage 4A-6.3 fixed single-frame smoke:
  global_valid_prediction_count `56602`, global_predicted_occupied_count
  `16792`, predicted_unmeasured_count `39400`, observed_state unchanged.
- Stage 4A-6.3 fixed one-step expert:
  best candidate id `11`, best score `662.6897120642332`,
  gain_exp/gain_sc/gain_hybrid `55.0 / 55.0 / 110.0`, observed_state
  unchanged.
- Stage 4A-6.3 fixed 5-step rollout:
  steps_completed `5`, done_reason `max_steps`, observed_ratio
  `0.0 -> 0.05899768518518519`, empty baseline at 5 steps
  `0.06896296296296296`, original SC at 5 steps `0.05899768518518519`,
  changed actions vs empty `5`, changed actions vs original SC `0`.
- Stage 4A-6.3 tests:
  py_compile passed and `test_alignment_convention_fix.py` passed; observed
  state hashes unchanged, checkpoint not modified, future observations marked
  evaluation-only, prediction stayed read-only and information-gain-only, and
  no RL/optimizer/BC/IL/SSCNet training ran.
- Stage 4A-6.4 calibrated / confidence-gated `I_sc` is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/calibrate_prediction_gain.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sc_gain_gating_ablation.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_sc_gain_gating.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sc_gain_gating.py`
- Stage 4A-6.4 updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
- Stage 4A-6.4 scoring additions:
  `sc_gain_formula` supports `raw_count`, `occupied_only`,
  `occupied_margin`, `confidence_weighted`, `entropy_weighted`,
  `calibrated_occupied`, and `novelty_discounted`. Runtime scoring now logs
  raw `gain_sc` separately from `effective_gain_sc` and `weighted_gain_sc`.
  `score_gain_mode=hybrid_weighted` uses
  `gain_exp + sc_gain_weight * min(effective_gain_sc, cap)`.
- Stage 4A-6.4 calibration:
  output `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/calibration`,
  samples `11175`, occupied_prob weighted bin correlation
  `0.8699543518514645`, confidence weighted bin correlation
  `0.893222674245022`, recommended occupied/confidence thresholds
  `0.9 / 0.9`, and `calibrated_occupied_usable=true`. Future observed maps
  were used only for post-hoc reliability-table estimation, not runtime
  planning or expert scoring.
- Stage 4A-6.4 one-step gating:
  output `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/one_step`.
  The best candidate stayed id `11` for empty, raw SC, occupied-only,
  occupied-margin, confidence-weighted, and calibrated-occupied cases. The
  selective formulas reduced effective SC gain but did not change the one-step
  action.
- Stage 4A-6.4 5-step gating ablation:
  output `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/ablation`.
  Completed configs `occupied_only_occ07`, `occupied_only_occ08`,
  `occupied_margin_occ06_w05`, and `confidence_weighted_conf05_cap30`; failed
  configs `[]`.
- Stage 4A-6.4 ablation result:
  empty baseline final observed_ratio at 5 steps
  `0.06896296296296296`; fixed raw SC final observed_ratio
  `0.05899768518518519`; all four gated configs also ended at
  `0.05899768518518519`, changed actions vs fixed raw SC `0/5`, and stayed
  below empty by `-0.009965277777777774`.
- Stage 4A-6.4 selectivity:
  mean raw `gain_sc` for gated rollouts remained `49.4`, while mean
  `effective_gain_sc` dropped to `4.2` for occupied_only_occ07, `3.2` for
  occupied_only_occ08, `1.7860426306724548` for occupied_margin_occ06_w05,
  and `36.19095666408539` for confidence_weighted_conf05_cap30. This confirms
  the scoring is gated, but candidate ranking is still dominated by the same
  measured-frontier/path-cost structure.
- Stage 4A-6.4 summary:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/summary`.
  Recommendation: no completed gating config changed raw SC behavior enough;
  prediction gain remains insufficiently selective for this scene.
- Stage 4A-6.4 logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a64_calibrate_prediction_gain.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a64_gain_gating_ablation.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a64_gain_gating_summary.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a64_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a64_gain_gating_test.log`
- Stage 4A-6.4 tests:
  py_compile passed and `test_sc_gain_gating.py` passed; calibration outputs,
  one-step outputs, ablation manifest, summary outputs, raw/effective gain
  logging, synthetic formulas, and weighted score formulas were validated.
  Prediction remained read-only and information-gain-only; no prediction
  writeback, traversability, collision, A*, ray blocking, future-observation
  planning/scoring, target/ground-truth leakage, RL/PPO/BC/IL/optimizer step,
  SSCNet retraining, or checkpoint modification occurred.

Important boundaries:

- Prediction output is standalone map_predict output.
- Do not write prediction into observed_map.
- Stage 2B expert scoring must not use `target_lr` or `target_hr`.
- Measured S is approximated only from sensor-derived `tsdf_lr` and/or `position`.
- Prediction set P is `PredictionLayer.confidence >= tau` and not measured.
- Default raycast is non-blocking with respect to SC prediction.
- Do not run RL, PPO, imitation learning training, Unreal, AirSim, robot execution, or retraining.
- Stage 3A only created Dataset/DataLoader, stats, policy skeleton, and
  forward-only BC dry-run.
- No optimizer step or model save has been performed.
- Stage 4A-1 only validates simulator depth observation and observed_map update.
- Stage 4A-1-viz only visualizes existing Stage 4A-1 outputs and does not
  modify observed_state.
- Stage 4A-1-scene-viz only visualizes the scripted Isaac scene and does not
  modify observed_state or observed_map.
- Stage 4A-2 only validates one simulator observed_map expert decision.
- Stage 4A-3 validates a deterministic multi-step simulator rollout using
  measured-only Isaac depth updates and EmptyPredictionLayer expert decisions.
- Stage 4A-3 uses planar teleport camera motion only; it is not physical robot
  path execution and does not include A* or a full SC-Explorer RRT tree planner.
- Stage 4A-3.2 only builds and validates a more complex scripted scene with
  fixed smoke views; it does not run a multi-step rollout on the medium scene.
- Stage 4A-3.5 uses A* only for observed-free path-cost scoring; motion still
  teleports and there is no physical path execution or full SC-Explorer RRT
  tree planner.
- Stage 4A-3.6 fixes candidate sampling to draw A* candidates from the current
  reachable observed-free component before scaling rollouts.
- Stage 4A-6.1 validates analysis/ablation tooling and shows that current
  SC-aware settings still underperform the empty baseline at 5 steps.
- Stage 4A-6.2 validates offline map_predict diagnostics and points to a
  projection convention issue plus dense confidence calibration.
- Stage 4A-6.4 validates calibrated/confidence-gated `I_sc`; it makes the SC
  term sparse in score space, but still does not change the 5-step selected
  actions or beat the empty baseline. The next step should inspect why gated
  prediction gain is still rank-insensitive, likely via candidate-level
  score/rank decomposition, spatial qualitative review of selected vs rejected
  candidates, and possibly a more frontier-local SC term. Still do not jump to
  RL/IL or scale rollouts.
- Stage 4A-6.5a candidate rank sensitivity diagnosis is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/analyze_candidate_rank_sensitivity_small.py`
  analyzed existing steps `0..4` only for empty baseline, fixed raw SC, and
  the four completed gated configs. Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65a_rank_sensitivity`.
  Result: gated selected candidate ids and positions are identical across all
  steps; top-1 is stable vs fixed raw SC while top-5/top-16 differ below the
  winner. Final score is best explained by inverse path cost
  (`pearson=0.8919154707376216`), not effective SC gain
  (`pearson=0.03806071813182923`). Selected candidates have mean low-path-cost
  rank `1.0333333333333334`, while mean gain-exp rank is `14.4`.
  Recommended next small task is offline counterfactual score analysis if
  path-cost dominance remains the focus. Still not RL/IL.
- Stage 4A-6.5b offline counterfactual score analysis is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_score_counterfactuals_small.py`
  loaded the Stage 4A-6.5a `candidate_rank_table.csv` and summary files only.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65b_counterfactual_scores`.
  Result: 94 score-formula variants executed over 6 configs, steps `0..4`,
  and 480 candidate rows. Removing path cost changed selection in 30/30
  `exp_only_no_cost` groups; over-cost formulas (`exp_over_cost`,
  `raw_hybrid_over_cost`, `effective_hybrid_over_cost`) did not change top-1.
  `sc_only` changed 10/20 executable groups. SC-specific lambda changes were
  possible with min lambda `0.1` and median `0.5`; `decoupled_sc_lambda0p5`
  is the only plausible one-step smoke candidate from this offline pass.
  Counterfactual formulas remain offline-only and do not claim observed_ratio
  improvement. No rollout, map_predict rerun, expert runtime edit, training,
  checkpoint edit, observed_state edit, prediction writeback, future planning
  observation, target, or ground-truth scoring was performed.
- Stage 4A-6.5d decoupled one-step spatial visualization is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_decoupled_one_step_case.py`
  loaded only the saved Stage 4A-6.5c one-step case:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65c_decoupled_one_step_smoke`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65d_decoupled_spatial_viz`.
  Generated `observed_baseline_decoupled_topdown.png`,
  `prediction_overlay_topdown.png`, `candidate_score_components_topdown.png`,
  `baseline_vs_decoupled_local_zoom.png`, `stage4a65d_spatial_summary.json`,
  and `stage4a65d_spatial_summary.md`.
  Result: baseline top-1 `grid:15,16,11` at `[-4.45,-4.35,1.15]`;
  decoupled top-1 `grid:14,18,11` at `[-4.55,-4.15,1.15]`;
  displacement `[-1,2,0]` cells, `0.22360679774997816 m`.
  The decoupled choice has higher `gain_exp` and `effective_gain_sc`, but also
  higher path cost, and is an adjacent local shift rather than a spatially
  meaningful new exploration branch. It is plausible for one-step formula
  comparison, but not enough to justify rollout.
  Validation passed: `py_compile`, 4 PNGs, summary JSON/MD, observed_state
  read-only and hash unchanged, and no rollout-like files were created. No
  Isaac startup, rollout, map_predict rerun, SSCNet training, RL/PPO/BC/IL,
  checkpoint modification, observed_state modification, prediction writeback,
  future-observation planning, target, or ground-truth scoring was performed.
- Stage 4A-6.5e offline candidate generation / path-level utility diagnosis is
  complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_candidate_generation_path_utility.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65e_path_candidate_diagnosis`.
  Result: 32 candidate sets were analyzed over steps `0..4`; full-64 runtime
  candidates have median distance `2.321601489576769 m`, but saved top-N
  candidates have median distance `0.9196431793625341 m`. Selected candidates
  match the minimum path-cost candidate in `0.9375` of analyzed sets, and
  path-cost/inverse-cost is the strongest final-score component in `0.96875`
  of sets. High-gain candidates are spatially different from selected actions
  (`selected_to_max_gain_exp` median `1.7464247558356059 m`,
  `selected_to_max_effective_gain_sc` median `1.0630145543593017 m`).
  The diagnostic 2-step proxy is not a true counterfactual tree, and its fixed
  next-step estimate preserved the same top candidate in all computed cases.
  Recommended next small task: original SC-Explorer RRT/tree utility source
  code inspection. Still no rollout or RL.
- Stage 4A-6.5f original SC-Explorer RRT/tree utility source-code inspection
  is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/inspect_original_tree_utility.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65f_original_tree_utility_inspection`.
  Result: tracked local `ssc_exploration` source clearly configures an
  external RRT*/tree planner stack through `RRTStar`,
  `RRTStarEvaluatorAdapter`, `SegmentTime`, `GlobalNormalizedGain`, and
  `SubsequentBest`, and integrates SC-specific gain through
  `SSCExplorationEvaluator` and `SSCVoxbloxOccupancyMap`.
  The concrete node/tree data structures, exact accumulated utility formula,
  and best-node/best-branch/first-path-node selection logic are not present in
  this repo; they live in external `mav_active_3d_planning` /
  `active_3d_planning_*` dependencies referenced by `.rosinstall` and
  `package.xml`.
  Recommended next small task: inspect/fetch the external active_3d_planning
  source manually before implementing any tree-utility prototype. Still no
  rollout, no RL, no map_predict rerun, no training.
- Stage 4A-6.5g external active_3d_planning source inspection is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/inspect_external_active_3d_planning.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65g_external_active3d_inspection`.
  External source found:
  `/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning`
  at commit `11634e8325480ce5da36a78b23b917347c973613`.
  Found packages: `active_3d_planning_core`, `active_3d_planning_ros`,
  `active_3d_planning_voxblox`, `active_3d_planning_mav`,
  `active_3d_planning_app_reconstruction`, and `mav_active_3d_planning`.
  Key source conclusion: `TrajectorySegment` stores `trajectory`, local
  `gain`, `cost`, `value`, `parent`, `children`, and `info`;
  `SegmentTime` computes segment execution time from trajectory timestamps
  with optional parent-cost accumulation disabled by default;
  `GlobalNormalizedGain` sets value to the best accumulated
  root-to-subtree `gain / cost`; `SubsequentBest` selects the immediate child
  whose subtree contains the highest-value segment; `OnlinePlanner` then
  executes that child segment. This confirms the original planner uses
  tree/branch utility and first-step-of-best-subtree selection rather than our
  current one-step local `gain / path_cost` collapse.
  Recommended next small task: offline minimal tree-utility prototype over
  saved candidates, reproducing `GlobalNormalizedGain` plus `SubsequentBest`
  without Isaac, rollout, map_predict rerun, RL/IL, or planner implementation.
- Stage 4A-6.5h offline minimal tree-utility prototype is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_tree_utility_prototype.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65h_offline_tree_utility_prototype`.
  The prototype defines `OfflineSegment`, reproduces
  `GlobalNormalizedGain` as subtree maximum accumulated root-to-descendant
  `gain / cost`, and reproduces `SubsequentBest` as selecting the root
  immediate child whose subtree contains the highest-value segment.
  Synthetic tests passed: low-cost trap selected `A` via descendant `A1`,
  subtree-does-not-help selected `B`, and zero-cost safety warned without
  crashing.
  Real-data diagnosis loaded 480 candidate rows over 6 configs and steps
  `0..4`; `one_step_star` matched current one-step top-1 in `30/30` default
  trees and `30/30` runtime-like trees. Recorded chains were built for fixed
  raw SC and empty baseline, but they are actual selected paths only, not
  counterfactual trees. Shallow pseudo-trees built 24 default-mode trees and
  changed `SubsequentBest` vs local one-step in `0/24`; proxy children are not
  true successors.
  Key conclusion: the formula works and overcomes the synthetic low-cost trap,
  but saved candidates do not contain real child-conditioned tree expansion.
  The main missing source-faithful piece remains offline RRT-like successor
  generation over saved observed maps, still without Isaac.
  Validation passed: `py_compile`, `test_offline_tree_utility_prototype.py`,
  all required outputs, no rollout-like outputs, external source git status
  clean. No Isaac startup, rollout, map_predict rerun, SSCNet inference,
  training/RL/BC/IL, checkpoint modification, observed_state modification,
  prediction writeback, target/ground-truth scoring, or external source
  build/modification occurred.
- Stage 4A-6.5i offline mini-RRT tree builder on saved observed map is
  complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_mini_rrt_tree.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65i_offline_mini_rrt_tree`.
  The builder creates real `MiniRRTSegment` parent/children relations on the
  saved fixed-SC `observed_state_step001.npy`, uses measured-only
  traversability and edge checks, evaluates local measured `gain_exp` with
  observed-state raycasts, records SegmentTime-like `segment_length_m / v_max`
  cost, approximates continuous yaw with 8 yaw samples, and then applies
  `GlobalNormalizedGain` plus `SubsequentBest`.
  Result: `255` accepted non-root nodes (`256` total), `100` rejected samples
  (`edge_non_traversable_or_unknown=41`, `target_same_as_nearest=59`), valid
  tree utility values, and observed_state SHA-256 unchanged
  `afacb32647bfa1b2ece34b75f19cb34cd15ec742b43e02b682af0fcd5f2bc59e`.
  `SubsequentBest` selected child `n0140`, grid `[14,13,11]`, world
  `[-4.55,-4.65,1.15]`, value `286.21642816261226`, accumulated gain/cost
  `32.0 / 0.11180350549906025`; the best descendant was the same node.
  The selected child differs from one-step baseline `grid:15,16,11` and
  decoupled `grid:14,18,11`, but it is only `0.11180350549906025 m` from root
  and also the root-local best, so this run did not reveal a nonlocal branch.
  Validation passed: `py_compile` and `test_offline_mini_rrt_tree.py`; required
  JSON/MD/CSV outputs and four PNG visualizations exist. No Isaac startup,
  rollout, online expert loop, map_predict rerun, SSCNet inference/training,
  RL/PPO/BC/IL, checkpoint modification, observed_state modification,
  prediction writeback, target/ground-truth scoring, or external source
  modification/build occurred.
  Recommended next small task: inspect gain/raycast or sampling strategy,
  still offline and still no rollout/RL.
- Stage 4A-6.5j offline mini-RRT gain/raycast/sampling diagnosis is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_mini_rrt_gain_sampling.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_mini_rrt_gain_sampling_diagnosis.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65j_gain_raycast_sampling_diagnosis`.
  Result: selected child `n0140` gain is reproducible (`32.0` logged and
  recomputed), visible unknown overlap with root/parent recorded yaw is `0/32`,
  and the gain is real new measured-map raycast information rather than
  root/parent duplicate unknown. The selected child still wins immediate
  root-child scoring because `32.0 / 0.11180350549906025 = 286.21642816261226`;
  segment lengths are very short overall (median `0.141421356237309m`, p75
  `0.20000000000000018m`), and `target_same_as_nearest=59` plus
  `edge_non_traversable_or_unknown=41` rejections indicate sampling/steering
  discretization pressure.
  Diagnostic filters: immediate root-child min segment length `0.15m` or min
  root distance `0.25m` moves selection off `n0140`; root/parent novelty alone
  does not move the root-child winner. Source inspection found evidence for
  `min_path_length`, `crop_min_length`, `max_density_range`, root rewiring,
  optional parent visible clearing, and continuous yaw; no mandatory
  root-visible overlap filter or near-root gain discount was proven.
  Validation passed: `py_compile` and
  `test_mini_rrt_gain_sampling_diagnosis.py`; required 19 outputs exist,
  observed_state hash unchanged, external source git status unchanged, and no
  rollout-like outputs were created.
  Recommended next small task: offline mini-RRT minimum-edge-length variant,
  no Isaac and no rollout/RL.
- Stage 4A-6.5k offline mini-RRT minimum-edge-length / crop-min-length
  variant is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_mini_rrt_min_edge_variants.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_mini_rrt_min_edge_variants.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65k_min_edge_length_variant`.
  The existing mini-RRT script now has optional source-like controls with
  unchanged defaults:
  `--min_edge_length_m`, `--min_root_child_length_m`,
  `--min_root_distance_m`, `--crop_min_length_m`,
  `--short_edge_policy allow|reject|crop`, `--density_radius_m`,
  `--max_nodes_per_density_radius`, and `--variant_name`.
  Result: all 9 variants completed. `baseline_allow` reproduced selected
  child `n0140` with value `286.21642816261226`, distance
  `0.11180350549906025m`, and observed_state hash unchanged.
  Variants moving off `n0140`: `reject_min_edge_0p15`,
  `reject_min_edge_0p25`, `reject_root_child_0p25`,
  `reject_root_distance_0p25`, `crop_min_length_0p15`,
  `crop_min_length_0p25`, `density_limited`, and `combined_source_like`.
  Nonlocal-branch variants by the Stage 4A-6.5k definition:
  `reject_root_child_0p25`, `reject_root_distance_0p25`, and
  `crop_min_length_0p25`.
  Best recommended variant for the next smoke is `crop_min_length_0p25`:
  selected child `n0001` at grid `[18,12,11]`, distance
  `0.5123476174067144m`, best descendant `n0249` at grid `[39,19,11]`,
  distance `2.6688013442592453m`, accumulated gain/cost
  `645.0 / 4.565369444959812`, accepted nodes `255`, rejected samples `916`,
  median segment length `0.2999999999999998m`, and correlation
  `0.6978966206169462`.
  Density limiting reduced tiny clustering but was too restrictive at this
  setting: `density_limited` and `combined_source_like` accepted `86` nodes
  and selected a local child at `0.287228140627606m`.
  Short-edge dominance is reduced for crop/reject/density variants but not
  eliminated globally; root-child filters still leave high inverse-length
  correlation because they only constrain root-level edges.
  Validation passed: `py_compile`,
  `test_mini_rrt_min_edge_variants.py`, and the original
  `test_offline_mini_rrt_tree.py` regression. No rollout-like outputs were
  created, observed_state SHA-256 stayed unchanged, external source git status
  stayed clean, and prediction was not used.
  Recommended next small task: no-prediction online one-step tree smoke using
  the source-like crop/min-length settings, still no rollout and no RL.

- Stage 4A-6.5l source-protected no-prediction one-step tree smoke is
  complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_source_protected_one_step_tree_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_source_protected_one_step_tree_smoke.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65l_source_protected_one_step_tree_smoke`.
  Profile:
  `source_like_crop_min_length_0p25`, `--short_edge_policy crop`,
  `--crop_min_length_m 0.25`, `--num_nodes 256`, `--sample_mode mixed`,
  `--gain_mode exp`, `--path_cost_mode segment_time`, `--num_yaw_samples 8`.
  Result: the one-step tree smoke ran on saved `observed_state_step001.npy`
  and exactly reproduced Stage 4A-6.5k `crop_min_length_0p25`: selected child
  `n0001` at grid `[18,12,11]`, world `[-4.15,-4.75,1.15]`, distance
  `0.5123476174067144m`; best descendant `n0249` at grid `[39,19,11]`,
  world `[-2.05,-4.05,1.15]`, distance `2.6688013442592453m`; accumulated
  gain/cost `645.0 / 4.565369444959812`; value `141.28100864040323`;
  accepted nodes `255`; rejected samples `916`.
  Protection checklist: crop/min-path-length active at `0.25m`; density
  limiting implemented but inactive; continuous yaw approximated with 8 fixed
  samples and active; root rewiring/reinsert present only as a checklist/hook
  and inactive; optional parent visible clearing and root-visible filtering /
  near-root discount intentionally inactive because source evidence was
  optional or missing.
  Comparison: the selected child moved off old short-edge winner `n0140`,
  differs from one-step baseline grid `[15,16,11]`, differs from decoupled
  grid `[14,18,11]`, and satisfies the Stage 4A-6.5k nonlocal definition.
  Validation passed: `py_compile`,
  `test_source_protected_one_step_tree_smoke.py`, required JSON/MD/CSV/PNG
  outputs exist, observed_state SHA-256 stayed unchanged, checkpoint SHA-256
  stayed unchanged, external active_3d_planning git status stayed clean, and
  no rollout-like or map_predict artifacts were created.
  Recommended next small task: no-prediction Isaac one-step capture + tree
  decision smoke, still no rollout.

- Stage 4A-6.5m no-prediction Isaac one-step capture + source-protected tree
  decision smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_one_step_tree_capture_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_one_step_tree_capture_smoke.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65m_isaac_one_step_tree_capture_smoke`.
  The runner started Isaac headless once, rebuilt deterministic
  `medium_three_rooms` with seed `0`, captured exactly one RGB/depth frame at
  fixed episode pose `pose_001.json`, and fused the measured depth into a new
  output-only observed map:
  `observed_state_isaac_capture_step001.npy`.
  Result: captured depth shape `(120,160)`, depth min/max
  `1.0084033012390137 / 4.979179859161377`, prior
  `observed_state_step000.npy` hash unchanged, and the new observed map hash
  exactly matched saved `observed_state_step001.npy`
  `afacb32647bfa1b2ece34b75f19cb34cd15ec742b43e02b682af0fcd5f2bc59e`.
  The source-protected tree used the Stage 4A-6.5l profile:
  `source_like_crop_min_length_0p25`, `short_edge_policy=crop`,
  `crop_min_length_m=0.25`, `num_nodes=256`, `sample_mode=mixed`,
  `gain_mode=exp`, `path_cost_mode=segment_time`, and `num_yaw_samples=8`.
  Result: exact match with Stage 4A-6.5l saved-map tree decision. Selected
  child `n0001`, grid `[18,12,11]`, world `[-4.15,-4.75,1.15]`, distance
  `0.5123476174067144m`; best descendant `n0249`, grid `[39,19,11]`, world
  `[-2.05,-4.05,1.15]`, distance `2.6688013442592453m`; accumulated
  gain/cost `645.0 / 4.565369444959812`; value `141.28100864040323`;
  accepted nodes `255`; rejected samples `916`.
  Generated required capture/report/tree outputs and plots in the output dir.
  Validation passed: `py_compile`,
  `test_isaac_one_step_tree_capture_smoke.py`, required outputs, active
  crop/min-length and 8-yaw protections, inactive density limiting, selected
  child moved off `n0140`, nonlocal branch found, prior observed map
  unchanged, checkpoint unchanged, external active_3d_planning clean, no
  prohibited rollout outputs, no map_predict artifacts, and no prediction use.
  Recommended next small task: no-prediction two-frame tree smoke. Still not
  rollout.
  No selected action execution, online multi-step loop, rollout, map_predict
  rerun, SSCNet inference/training, RL/PPO/BC/IL, checkpoint modification,
  existing observed_state modification, prediction writeback, prediction
  collision/traversability use, target/ground-truth scoring, external source
  modification/build, or coverage-improvement claim occurred.

- Stage 4A-6.5n no-prediction two-frame source-protected tree smoke is
  complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_two_frame_tree_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_two_frame_tree_smoke.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65n_two_frame_tree_smoke`.
  The runner started Isaac headless once, rebuilt deterministic
  `medium_three_rooms` seed `0`, captured exactly two RGB/depth frames, and
  executed exactly one selected-child move between frames.
  Frame 1 used position `[-4.65,-4.65,1.2]`, yaw
  `0.38710316317995463`; its measured-only update produced
  `observed_state_frame001.npy` with hash
  `afacb32647bfa1b2ece34b75f19cb34cd15ec742b43e02b682af0fcd5f2bc59e` and
  exactly reproduced Stage 4A-6.5m / 6.5l: selected child `n0001`, grid
  `[18,12,11]`; best descendant `n0249`, grid `[39,19,11]`; accumulated
  gain/cost `645.0 / 4.565369444959812`; value `141.28100864040323`.
  The single executed move used selected child world x/y `[-4.15,-4.75]`,
  fixed camera height `1.2m`, and selected child yaw `2.15879915042112`.
  Frame 2 captured at pose `[-4.15,-4.75,1.2]`, added `6251` measured
  observed voxels, and produced `observed_state_frame002.npy` with hash
  `aeb1b990f783d2548c3f738fc5f4ba4ee922b8bda53fb32cf9b7818938c674a1`.
  Frame 2 tree decision succeeded and remained nonlocal: selected child
  `n0001`, grid `[17,16,11]`, world `[-4.25,-4.35,1.15]`, distance
  `0.502493918652551m`; best descendant `n0112`, grid `[8,27,11]`, world
  `[-5.15,-3.25,1.15]`, distance `1.4874475373705685m`; accumulated
  gain/cost `323.0 / 2.315392939101747`; value `139.50115962835548`.
  Validation passed: `py_compile`, `test_isaac_two_frame_tree_smoke.py`,
  exactly two frames, exactly one action execution, active crop/min-length
  `0.25m`, prediction disabled, no map_predict artifacts, no rollout
  manifest/plots, prior observed_state unchanged, frame1 observed_state not
  modified during frame2 update, checkpoint unchanged, external source
  unchanged, no third frame, and no leakage flags.
  Recommended next small task: map_predict + source-protected tree one-step
  smoke. Still not rollout.
  No open-ended multi-step loop, rollout, map_predict rerun during this stage,
  SSCNet inference/training, RL/PPO/BC/IL, checkpoint modification, existing
  observed_state modification, prediction writeback, prediction collision /
  traversability use, target/ground-truth scoring, external source
  modification/build, or coverage-improvement claim occurred.
- Stage 4A-6.5o map_predict + source-protected tree one-step smoke is
  complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_tree_one_step_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_map_predict_tree_one_step_smoke.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65o_map_predict_tree_one_step_smoke`.
  The runner started Isaac headless once, rebuilt deterministic
  `medium_three_rooms` seed `0`, captured exactly one RGB/depth frame at
  `[-4.65,-4.65,1.2]`, yaw `0.38710316317995463`, and fused measured depth
  into a new output-only `observed_state_frame001.npy`.
  Measured-only update added `3727` observed voxels and produced hash
  `afacb32647bfa1b2ece34b75f19cb34cd15ec742b43e02b682af0fcd5f2bc59e`;
  the prior `observed_state_step000.npy` was unchanged.
  map_predict ran once with checkpoint
  `/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar`,
  `alignment_convention=code_consistent_v1`, `tau=0.1`, and saved read-only
  prediction layer
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65o_map_predict_tree_one_step_smoke/map_predict/global_prediction_layer.npz`.
  Prediction shape aligned to observed_state `(120,120,30)`;
  `prediction_valid_count=57382`, `predicted_unmeasured_count=37537`, and
  `predicted_occupied_count=16779`.
  Source-protected measured-only tree and SC tree both used the Stage 4A-6.5l
  profile: `short_edge_policy=crop`, `crop_min_length_m=0.25`,
  `num_nodes=256`, `max_extension_m=0.5`, `sample_mode=mixed`,
  `path_cost_mode=segment_time`, `v_max=1.0`, `robot_radius_m=0.2`,
  `voxel_size=0.1`, `raycast_stride=2`, `num_yaw_samples=8`,
  `max_ray_length_m=4.8`, and `seed=0`.
  SC tree ran with `prediction_mode=sim_dynamic`, `gain_mode=hybrid`,
  `sc_gain_formula=raw_count`, and `gain_hybrid = gain_exp + gain_sc`.
  Measured-only selected child remained `n0001`, grid `[18,12,11]`;
  SC-tree selected child also remained `n0001`, grid `[18,12,11]`.
  The selected child therefore did not differ from measured-only in this
  one-step smoke, so there was no spatially meaningful selected-child change.
  SC-tree best descendant remained `n0249`, grid `[39,19,11]`;
  accumulated gain/cost became `1258.0 / 4.565369444959812`, value
  `275.55491379794235`.
  SC gain was nonzero: `255/255` non-root tree nodes had `gain_sc > 0`;
  `gain_sc` min/mean/max was `15.0 / 48.88235294117647 / 62.0`.
  `gain_occ` min/mean/max was `0.0 / 7.101960784313725 / 35.0`;
  `gain_conf` min/mean/max was
  `2.957338571548462 / 17.04069787941727 / 25.224848687648773`.
  Validation passed: `py_compile`, `test_map_predict_tree_one_step_smoke.py`,
  existing mini-RRT regression, Stage 4A-6.5l source-protected regression,
  one frame only, no selected action execution, no two-frame loop, no rollout,
  checkpoint unchanged, external source unchanged, prior observed_state
  unchanged, new observed_state unchanged after map_predict/tree, no dense
  `class_prob`, and no leakage flags.
  Prediction stayed read-only and information-gain-only: it was not written to
  observed_state, did not affect traversability/collision, did not block
  rays, and used no target/ground-truth scoring.
  Recommended next small task: map_predict + source-protected tree two-frame
  smoke. Still not rollout.
  No rollout, online open-ended loop, selected action execution, two-frame run,
  SSCNet training, RL/PPO/BC/IL, checkpoint modification, existing
  observed_state modification, prediction writeback, prediction collision /
  traversability/ray-blocking use, target/ground-truth scoring, external
  source modification/build, or coverage-improvement claim occurred.
- Stage 4A-6.5p map_predict + source-protected tree two-frame smoke is
  complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_tree_two_frame_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_map_predict_tree_two_frame_smoke.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65p_map_predict_tree_two_frame_smoke`.
  Isaac started headless once, captured exactly two frames, executed exactly
  one selected-child move, and stopped after the second tree decision. The
  SSCNet checkpoint was loaded once through `IsaacMapPredictor`, and
  map_predict ran exactly twice with `alignment_convention=code_consistent_v1`
  and `tau=0.1`.
  Frame 1 measured update added `3727` observed voxels and map_predict
  produced shape-aligned prediction `(120,120,30)` with
  `prediction_valid_count=57382`, `predicted_unmeasured_count=37537`, and
  `predicted_occupied_count=16779`. Frame 1 SC tree matched the current
  Stage 4A-6.5o reference artifact: selected child `n0001`, grid `[18,12,11]`;
  best descendant `n0249`, grid `[39,19,11]`; accumulated
  gain_exp/gain_sc/gain_hybrid/cost `645.0 / 638.0 / 1283.0 /
  4.565369444959812`; value `281.02873501649196`. It did not change the
  measured-only selected child.
  The single executed action moved to SC selected child `n0001`, pose
  `[-4.15,-4.75,1.2]`, yaw `2.15879915042112`.
  Frame 2 measured update added `6251` observed voxels and map_predict
  produced `prediction_valid_count=37258`,
  `predicted_unmeasured_count=26620`, and `predicted_occupied_count=6638`.
  Frame 2 measured tree matched the no-prediction Stage 4A-6.5n frame-2
  decision: selected child `n0001`, grid `[17,16,11]`; best descendant
  `n0112`, grid `[8,27,11]`; accumulated gain/cost
  `323.0 / 2.315392939101747`; value `139.50115962835548`.
  Frame 2 SC tree changed both selected child and best descendant:
  selected child `n0127`, grid `[11,15,11]`; best descendant `n0162`, grid
  `[14,15,11]`; accumulated gain_exp/gain_sc/gain_hybrid/cost
  `76.0 / 75.0 / 151.0 / 0.5872281406276059`; value
  `257.14026551693735`.
  SC gain was nonzero in both frames: frame 1 `255/255` nodes and frame 2
  `248/255` nodes had `gain_sc > 0`. gain_exp/gain_sc correlation was
  `0.9427480283392026` in frame 1 and `0.02431093087475427` in frame 2.
  Validation passed: `py_compile` and
  `test_map_predict_tree_two_frame_smoke.py`; required output/checklist/plot
  files exist, gain_hybrid identity holds, no third frame or rollout artifacts
  were produced, checkpoint and external source were unchanged, existing
  observed_state files were unchanged, and prediction stayed read-only and
  information-gain-only with no traversability/collision/ray-blocking use.
  Recommended next small task: controlled gated SC tree one-step smoke or a
  repeated two-frame smoke. Still not rollout.

- Stage 4A-6.5q SC-tree branch-change diagnosis and gated replay is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_sc_tree_branch_change.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sc_tree_branch_change_diagnosis.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65q_sc_tree_branch_change_diagnosis`.
  This stage was offline only: no Isaac startup, no map_predict rerun, no
  SSCNet inference, no selected action execution, no rollout, no training/RL,
  no checkpoint change, no observed_state modification, no prediction
  writeback, and no prediction traversability/collision/ray blocking.
  Diagnosis result: frame2 SC changed from measured selected child `n0001`
  grid `[17,16,11]` to SC selected child `n0127` grid `[11,15,11]`; SC best
  descendant was `n0162` grid `[14,15,11]`. The selected-child spatial delta
  was `0.6082762530298217m`, and the best-descendant delta was
  `1.3416407864998743m`.
  Cause: on the saved SC tree, branch `n0127 -> n0162` had lower exp-only
  value than `n0001 -> n0112` (`129.42159059130623` vs
  `134.7503461425601`), but higher SC-only value per cost
  (`127.71867492563113` vs `121.79358209039086`) and much lower accumulated
  cost (`0.5872281406276059` vs `2.315392939101747`). Raw hybrid value barely
  favored SC (`257.14026551693735` vs `256.543928232951`, margin
  `0.5963372839863723`).
  Rank result: frame1 gain_exp/gain_sc correlation was
  `0.9427480283392026` and did not change the selected child; frame2
  correlation was `0.02431093087475427`, with `248/255` nodes having
  positive gain_sc.
  Gated replay result: raw_count, weight `1.0`, cap `25`, cap `50`, and
  confidence-weighted gain kept SC child `n0127`; weights `0.0`, `0.25`,
  `0.5`, and occupied-only returned to measured child `n0001`. Minimum SC
  weight changing the selected child was approximately
  `0.899353934095411`. Calibrated occupied replay was skipped because saved
  nodes do not contain per-node visible voxel probability samples.
  Recommended next small task: gated SC tree one-step smoke. Still not
  rollout.

- Stage 4A-6.5r gated SC tree one-step smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_gated_sc_tree_one_step_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_gated_sc_tree_one_step_smoke.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65r_gated_sc_tree_one_step_smoke`.
  This stage was saved-frame offline only. It used the Stage 4A-6.5p Frame 2
  saved `observed_state_frame002.npy`, saved
  `frame002_prediction/global_prediction_layer.npz`, saved pose/camera info,
  and the same source-protected mini-RRT profile/root convention as Stage
  4A-6.5p. It did not start Isaac, capture RGB/depth, rerun map_predict, run
  SSCNet inference, execute an action, run two-frame, run rollout, train,
  modify checkpoint, modify observed_state, write prediction into observed map,
  or use prediction for traversability/collision/ray blocking.
  `offline_mini_rrt_tree.py` now supports optional gated SC runtime scoring
  through explicit `sc_gain_formula` while preserving default measured-only
  `gain_mode=exp` behavior. Nodes record raw `gain_sc`, `gain_occ`,
  `gain_conf`, `effective_gain_sc`, and `gain_hybrid_effective`; tree utility
  uses `gain_exp + effective_gain_sc` only for explicitly requested SC/hybrid
  formulas.
  Formula results on Frame 2:
  measured-only selected `n0001`, grid `[17,16,11]`, best `n0112`, value
  `139.50115962835548`; raw_count reproduced Stage 4A-6.5p with selected
  `n0127`, grid `[11,15,11]`, best `n0162`, value `257.14026551693735`;
  weight `0.5` returned to measured `n0001`; weight `1.0`, cap `25`, cap
  `50`, confidence-weighted, and confidence-weighted cap `25` preserved
  `n0127`; occupied-only returned to measured `n0001`.
  Accumulated effective SC on the raw-count winning branch was `75.0`; cap
  `25` used `50.0`; confidence-weighted used `31.506256222724915`; all on
  accumulated cost `0.5872281406276059` for best descendant `n0162`.
  Validation passed: `py_compile`,
  `test_gated_sc_tree_one_step_smoke.py`, `test_offline_mini_rrt_tree.py`,
  Stage 4A-6.5l source-protected regression, Stage 4A-6.5n no-prediction
  two-frame regression, Stage 4A-6.5o map-predict one-step regression, Stage
  4A-6.5p two-frame output regression, and Stage 4A-6.5q branch-change
  regression. Observed_state,
  prediction NPZ, checkpoint, and external source hashes/status stayed
  unchanged; no rollout-like or prohibited output artifacts were created.
  Next small task should choose a conservative gated formula for a later
  staged test or repeat the saved/two-frame smoke as needed. Still not direct
  rollout.

- Stage 4A-6.5s confidence-weighted / cap25 gated SC tree two-frame smoke is
  complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_gated_sc_tree_two_frame_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_gated_sc_tree_two_frame_smoke.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65s_gated_sc_tree_two_frame_smoke`.
  The runner used `confidence_weighted` as the executed primary SC gain
  formula and `cap25` as a same-frame shadow formula. Isaac started headless
  once through the Stage 4A-6.5p two-frame runner, captured exactly two
  frames, ran exactly two map_predict calls with one loaded `IsaacMapPredictor`,
  executed exactly one `confidence_weighted` selected-child move, and did not
  capture a third frame or execute a second action.
  Frame 1 measured-only and `confidence_weighted` both selected `n0001`, grid
  `[18,12,11]`, best descendant `n0249`, grid `[39,19,11]`.
  `confidence_weighted` winning-path accumulated effective SC gain was
  `240.77071565389633`; raw gain_sc was `633.0`. Frame 1 `cap25` shadow
  selected `n0196`, grid `[15,11,11]`, best `n0196`, and was not executed.
  The single executed move was the `confidence_weighted` selected child
  `n0001`, pose `[-4.15,-4.75,1.2]`, yaw `2.15879915042112`.
  Frame 2 measured-only selected `n0001`, grid `[17,16,11]`, best `n0112`,
  matching Stage 4A-6.5n. Frame 2 `confidence_weighted` changed the selected
  child to `n0127`, grid `[11,15,11]`, best descendant `n0162`, grid
  `[14,15,11]`, matching the Stage 4A-6.5p/6.5r SC branch. Frame 2 `cap25`
  shadow also selected `n0127` and best `n0162`.
  Frame 2 winning-branch effective SC gain was `31.506256222724915` for
  `confidence_weighted` and `50.0` for `cap25`; raw gain_sc was `75.0` and
  accumulated cost was `0.5872281406276059`.
  Prediction stats: frame 1 `prediction_valid_count=57382`,
  `predicted_unmeasured_count=37537`, `predicted_occupied_count=16779`;
  frame 2 `prediction_valid_count=37258`,
  `predicted_unmeasured_count=26620`, `predicted_occupied_count=6638`.
  Observed ratios were `0.05104398148148148` after frame 1 and
  `0.06551388888888889` after frame 2.
  Validation passed: `py_compile`, Stage 4A-6.5r one-step regression, Stage
  4A-6.5p two-frame regression, and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65s_gated_sc_tree_two_frame_smoke_test.log`.
  Safety passed: no rollout, no open-ended loop, no third frame, no second
  action, no RL/PPO/BC/IL/optimizer/training, no checkpoint modification, no
  existing observed_state modification, no prediction writeback, no prediction
  traversability/collision/ray-blocking use, no target/ground-truth scoring,
  no external source modification/build, and no coverage-improvement claim.
  The result is enough for repeated gated two-frame smoke or a short gated SC
  tree smoke if staged next, but still not enough for direct rollout.

- Stage 4A-6.5t alternate-tree-seed gated SC tree two-frame smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_gated_sc_tree_seed_repeat_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_gated_sc_tree_seed_repeat_smoke.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke`.
  This repeated Stage 4A-6.5s with scene `medium_three_rooms`, scene seed `0`,
  start pose unchanged, primary executed formula `confidence_weighted`, cap25
  as shadow only, and changed only the mini-RRT/tree sampling seed to `1`.
  The recorded profile name is
  `source_like_crop_min_length_0p25_seed1`.
  Result: one Isaac headless startup, exactly two captured frames, exactly two
  map_predict calls, exactly one `confidence_weighted` selected-child move, no
  third frame, and no second action.
  Frame 1 measured-only, `confidence_weighted`, and `cap25` all selected
  `n0001`, grid `[18,12,11]`, best `n0245`, grid `[33,14,11]`.
  Frame 1 `confidence_weighted` winning-path accumulated effective SC gain was
  `148.99908256530762`; cap25 effective SC gain was `175.0`; raw gain_sc was
  `368.0`; cost was `2.4879349937152315`.
  The single executed move was still the `confidence_weighted` selected child
  `n0001`, pose `[-4.15,-4.75,1.2]`, yaw `2.15879915042112`.
  Frame 2 measured-only selected `n0057`, grid `[12,16,11]`, best `n0118`,
  grid `[12,19,11]`. Frame 2 `confidence_weighted` also selected `n0057` and
  best `n0118`, so seed `1` returned to the measured-only selected branch.
  It did not reproduce the exact Stage 4A-6.5s `n0127 -> n0162` id branch,
  although the selected/best worlds were spatially close to that reference
  branch (`0.14142135623730964m` and `0.4472135954999583m` deltas).
  Frame 2 `cap25` shadow matched `confidence_weighted` (`n0057 -> n0118`).
  Frame 2 winning-branch effective SC gain was `35.51751762628555` for
  `confidence_weighted` and `50.0` for `cap25`; raw gain_sc was `82.0` and
  accumulated cost was `0.620156278894175`.
  Prediction stats matched Stage 4A-6.5s counts: frame 1
  `prediction_valid_count=57382`, `predicted_unmeasured_count=37537`,
  `predicted_occupied_count=16779`; frame 2
  `prediction_valid_count=37258`, `predicted_unmeasured_count=26620`,
  `predicted_occupied_count=6638`.
  Validation passed: `py_compile`, Stage 4A-6.5s gated two-frame regression,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke_test.log`.
  Safety passed: no rollout, no open-ended loop, no third frame, no second
  action, no RL/PPO/BC/IL/optimizer/training, no checkpoint modification, no
  existing observed_state modification, no prediction writeback, no prediction
  traversability/collision/ray-blocking use, no target/ground-truth scoring,
  no external source modification/build, and no coverage-improvement claim.
  Conclusion: this is not enough to proceed to a 3-frame gated smoke or
  rollout. Recommended next small task is seed robustness diagnosis before any
  longer smoke.

- Stage 4A-6.5u seed robustness diagnosis is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_gated_tree_seed_robustness.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_gated_tree_seed_robustness_diagnosis.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65u_seed_robustness_diagnosis`.
  This was offline-only and read saved Stage 4A-6.5s seed0 plus Stage 4A-6.5t
  seed1 artifacts. It did not start Isaac, capture new RGB/depth, rerun
  map_predict, run SSCNet inference, execute actions, run rollout, train, or
  modify checkpoints/observed_state.
  Result: seed1 Frame 2 `confidence_weighted` truly returned to measured-only
  in seed1 score/tree space (`n0057 -> n0118`, measured rank under confidence
  scoring `1`), but the selected branch remained spatially close to the seed0
  SC branch (`n0127 -> n0162`): selected-child delta
  `0.14142135623730964m`, best-descendant delta
  `0.4472135954999583m`. Top-K confidence branch clouds overlapped.
  Classification: seed0 confidence is `spatially_same_as_seed0_sc`; seed1
  confidence and seed1 cap25 are both `same_as_measured` and also
  `spatially_same_as_seed0_sc`.
  Rank/margin: seed0 confidence margin was narrow (`2.3578116606364574`,
  normalized `0.012879002637364282`); seed1 confidence margin was healthier
  (`31.643506691543223`, normalized `0.1563440683986311`). Effective SC was
  decisive for seed0 but not across both seeds; cost dominance was not
  supported by the saved value/inverse-cost correlations.
  Recommendation: do multi-seed offline replay / seed robustness sweep before
  any longer smoke. Still do not proceed to rollout.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65u_seed_robustness_diagnosis.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65u_seed_robustness_diagnosis_test.log`.

- Stage 4A-6.5v multi-seed offline replay / seed robustness sweep is
  complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_multi_seed_gated_tree_replay.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_multi_seed_gated_tree_replay.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65v_multi_seed_offline_replay`.
  This was offline-only on the saved Stage 4A-6.5p Frame 2 observed_state and
  prediction NPZ, with the saved source-protected replay context reused to
  match the Stage 4A-6.5p/6.5s root. It ran tree seeds `0..9` for
  `measured_only`, `confidence_weighted`, `cap25`, and `raw_count` using the
  crop-min-length source-like profile.
  Seed 0 replay exactly reproduced the Stage 4A-6.5s confidence branch
  `n0127 -> n0162`; seed 1 replay exactly reproduced the Stage 4A-6.5t
  confidence branch `n0057 -> n0118`.
  Multi-seed result: `confidence_weighted` exact seed0 SC fraction `0.1`,
  spatial seed0 SC basin fraction `0.3`, same-as-measured fraction `0.7`,
  measured-but-seed0-SC-basin fraction `0.1`, and distinct SC branch fraction
  `0.1`. `cap25` spatial seed0 SC basin fraction was `0.5`.
  Confidence/cap25 exact selected-child agreement was `0.8`; confidence vs
  measured agreement was `0.7`; cap25 vs measured agreement was `0.7`.
  `raw_count` matched confidence's broad behavior in this sweep
  (`0.3` spatial seed0 SC basin, `0.7` same-as-measured).
  Confidence normalized margins were not mostly narrow: min/median/max
  `0.012879002637364282 / 0.12317906042561935 /
  0.22865634393797724`; narrow seeds (`<0.02`) were `[0,2,8]`.
  Value/effective-SC correlation was high on average (`0.7175741643849938`),
  while value/inverse-cost correlation did not support cost dominance
  (mean `0.29905939149745453`, max `0.47222953967781184`).
  Conclusion: confidence-weighted branch choice is seed-sensitive and not
  spatially robust enough for 3-frame or another start/scene two-frame smoke.
  Recommended next small task is tree sampling stabilization or SC gain design
  review. Still do not proceed to rollout.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65v_multi_seed_gated_tree_replay.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65v_multi_seed_gated_tree_replay_test.log`.
  Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no rollout/open-ended loop,
  no training/RL, no checkpoint/observed_state/prediction NPZ modification, no
  prediction writeback, no prediction traversability/collision/ray blocking,
  no target/ground-truth scoring, no external source modification/build, and
  no coverage-improvement claim.

- Stage 4A-6.5x source-faithful SC gain design review and visible-voxel
  decomposition is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/review_source_sc_gain_design.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_source_sc_gain_design_review.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65x_sc_gain_design_review`.
  This was offline-only on saved Stage 4A-6.5p Frame 2 observed_state and
  prediction NPZ, plus saved Stage 4A-6.5p/6.5r/6.5v/6.5w tree artifacts and
  local/external source. No visible voxel ids were saved in the tree artifacts,
  so visibility was recomputed diagnostically with the current observed-state
  raycaster; prediction was not used for ray blocking.
  Source evidence: `SSCExplorationEvaluator` in `sc_explorer.yaml` rewards
  predicted occupied and predicted free voxels with weights `1.0 / 1.0`,
  sets `unobserved_weight: 0.0`, uses `ssc_confidence_threshold: 0.05`, and
  does not enable direct `weight_by_confidence`. `use_ssc_information_planning`
  is `false`, so prediction does not block visibility raycasting in this
  source profile. Parent-visible clearing is supported by the base evaluator
  but inactive in `sc_explorer.yaml`.
  Current formula audit: `raw_count` is source-inspired but not exact unless
  restricted to source OCC/FREE threshold semantics; `confidence_weighted`,
  `cap25`, `occupied_only`, `occupied_margin`, `calibrated_occupied`,
  `novelty_discounted`, and branch normalization are diagnostic or
  source-inspired, not exact source-faithful formulas.
  Key decomposition: seed0 measured branch `n0001 -> n0112` had recorded
  `gain_exp=323.0`, raw/source OCC+FREE SC `593.0 / 569.0`, parent/root-cleared
  source SC `315.0`, frontier-local source SC `512.0`, and path cost
  `2.315392939101747`. Seed0 SC branch `n0127 -> n0162` had recorded
  `gain_exp=76.0`, raw/source OCC+FREE SC `136.0 / 135.0`, parent/root-cleared
  source SC `102.0`, frontier-local source SC `114.0`, and path cost
  `0.5872281406276059`. Thus the seed0 SC branch advantage is not dominated by
  predicted-source-unknown voxels; it is mainly a short, low-cost local branch
  with low-novelty/root-overlap prediction visibility.
  Candidate variant proxy: `source_raw_predicted_occupied_free` and
  source-thresholded OCC/FREE select the measured branch in the seed0 proxy;
  parent-visible-cleared and spatial-normalized diagnostics can still keep the
  short SC branch because of low cost; frontier-local selects measured in the
  seed0 proxy.
  Recommendation: next small task should be offline source OCC+FREE plus
  parent-visible-cleared/frontier-local seed replay on saved Frame2 artifacts.
  Runtime smoke is not ready and rollout is not ready.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65x_sc_gain_design_review.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65x_sc_gain_design_review_test.log`.
  Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no rollout/open-ended loop,
  no training/RL, no checkpoint/observed_state/prediction NPZ modification, no
  prediction writeback, no prediction traversability/collision/ray blocking,
  no target/ground-truth scoring, no external source modification/build, and
  no coverage-improvement claim.

- Stage 4A-6.5w source-faithful RRTStar root-rewire / tree-persistence
  stabilization prototype is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_source_faithful_rewire_persistence.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_source_faithful_rewire_persistence.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65w_source_faithful_rewire_persistence`.
  This was offline-only on saved Stage 4A-6.5p Frame1/Frame2 observed_state
  and prediction NPZ artifacts. It inspected external active_3d_planning
  source at commit `11634e8325480ce5da36a78b23b917347c973613`, wrote source
  evidence summaries for `rewireRoot`, branch reinsert/preservation,
  `max_density_range`, `min_path_length`, `crop_min_length`,
  `ContinuousYawPlanningEvaluator`, `GlobalNormalizedGain`, and
  `SubsequentBest`, then compared saved fresh Stage 4A-6.5v replay against
  persistent rewire configs.
  Fresh baseline reproduced Stage 4A-6.5v:
  `confidence_weighted` spatial seed0 SC basin `0.3`, same-as-measured `0.7`,
  confidence/cap25 exact agreement `0.8`.
  Persistent no-density results did not improve the seed0-SC basin:
  `persistent_rewire_256_no_density` confidence spatial basin `0.0`,
  same-as-measured `0.5`, confidence/cap25 agreement `0.8`;
  `persistent_rewire_512_no_density` confidence spatial basin `0.0`,
  same-as-measured `0.7`, confidence/cap25 agreement `0.8`.
  Persistent trees did preserve/reinsert many branches:
  256 no-density mean preserved/pruned/reinserted/new nodes
  `169.35 / 8.5 / 56.5 / 29.15`, preserved-subtree winner fraction `0.875`;
  512 no-density mean preserved/pruned/reinserted/new nodes
  `169.35 / 8.5 / 56.5 / 192.075`, preserved-subtree winner fraction `0.85`.
  Source density config values were found (`max_density_range: 1.0`), but the
  source-like density diagnostic was too restrictive in this mini-RRT/profile:
  no valid preserved roots or winners for the 256/512 source-density configs.
  Recommendation: SC gain design review before another runtime smoke. Still
  not rollout.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65w_source_faithful_rewire_persistence.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65w_source_faithful_rewire_persistence_test.log`.
  Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no rollout/open-ended loop,
  no training/RL, no checkpoint/observed_state/prediction NPZ modification, no
  prediction writeback, no prediction traversability/collision/ray blocking,
  no target/ground-truth scoring, no external source modification/build, and
  no coverage-improvement claim.

- Stage 4A-6.5y offline source OCC+FREE / parent-cleared / frontier-local
  seed replay is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_source_gain_seed_replay.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_source_gain_seed_replay.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65y_source_gain_seed_replay`.
  This was offline-only on the saved Stage 4A-6.5p Frame2 observed_state and
  saved prediction NPZ. It did not start Isaac, capture RGB/depth, rerun
  map_predict, run SSCNet inference, execute actions, run two-frame runtime,
  run rollout, train, modify checkpoints/observed_state/prediction NPZ, write
  prediction into observed_map, use prediction for traversability/collision/ray
  blocking, use target/ground truth, modify/build external source, or claim
  coverage improvement.
  Created inventory and mapping reports:
  `prediction_npz_field_inventory.*` and
  `source_occ_free_mapping_report.*`. The simulator NPZ mapping is recorded as
  `source-faithful-approx`: `global_prediction_valid`, `global_confidence`,
  `global_occupied_prob`, `global_free_prob`, and `global_pred_class` are
  present, but the NPZ is not the exact C++ SSCMap log-odds layer.
  Completed 10 seeds (`0..9`) and 11 formulas:
  `measured_only`, `current_confidence_weighted`, `current_cap25`,
  `current_raw_count`, `source_occ_free`, `source_occ_free_thresholded`,
  `parent_visible_cleared_source_occ_free`,
  `root_visible_cleared_source_occ_free`,
  `frontier_local_source_occ_free`,
  `parent_cleared_frontier_local_source_occ_free`, and
  `branch_normalized_source_occ_free`.
  Decision rows: `110`. Required output files/plots: `43`.
  Seed0 `current_confidence_weighted` reproduced the Stage 4A-6.5s reference
  `n0127 -> n0162`. Seed0 `source_occ_free` and
  `source_occ_free_thresholded` still selected `n0127 -> n0162`, so source
  OCC+FREE alone did not remove the short low-cost local SC branch in full-tree
  replay.
  Key multi-seed fractions:
  `current_confidence_weighted` spatial seed0 SC basin `0.3`,
  same-as-measured `0.7`; `source_occ_free` spatial seed0 SC basin `0.4`,
  same-as-measured `0.6`; `parent_visible_cleared_source_occ_free` spatial
  seed0 SC basin `0.5`, same-as-measured `0.7`;
  `frontier_local_source_occ_free` spatial seed0 SC basin `0.4`,
  same-as-measured `0.6`;
  `parent_cleared_frontier_local_source_occ_free` spatial seed0 SC basin
  `0.5`, same-as-measured `0.7`; `branch_normalized_source_occ_free` spatial
  seed0 SC basin `0.5`, same-as-measured `0.7`.
  Conclusion: source OCC+FREE variants did not robustly eliminate the seed0 SC
  basin. Runtime smoke and rollout are still not ready.
  Recommended next faithful step:
  inspect source OCC/FREE mapping and source-inspired novelty filters offline
  before any runtime smoke.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65y_source_gain_seed_replay.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65y_source_gain_seed_replay_test.log`.

- Stage 4A-6.5z offline decoupled SC utility sweep is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_decoupled_sc_utility_sweep.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_decoupled_sc_utility_sweep.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65z_decoupled_sc_utility_sweep`.
  This was offline-only on saved Stage 4A-6.5p Frame2 observed_state,
  prediction NPZ, pose/camera info, and saved Stage 4A-6.5y raw mini-RRT
  trees. It did not start Isaac, capture RGB/depth, rerun map_predict, run
  SSCNet inference, execute actions, run two-frame runtime, run rollout, train,
  modify checkpoints/observed_state/prediction NPZ, write prediction into
  observed_map, use prediction for traversability/collision/ray blocking, use
  target/ground truth, modify/build external source, or claim coverage
  improvement.
  Formula tested:
  `value = gain_exp / cost + lambda * normalized_sc`, with SC outside the cost
  division. `normalized_sc` is per-seed/per-basis max-normalized over valid
  raw-tree paths and clipped to `[0,1]`.
  Completed 10 seeds (`0..9`), 3 SC bases
  (`source_occ_free`, `parent_visible_cleared_source_occ_free`,
  `frontier_local_source_occ_free`), fixed lambdas
  `0,1,2,4,8,12,16,24,32`, and adaptive lambdas
  `0.25,0.5,1.0,2.0 * (p90(base_exp_value)-p50(base_exp_value))`.
  Decision rows: `390`.
  Seed0 base gap report:
  measured `323.0 / 2.315392939101747 = 139.50115962835548`;
  seed0 SC `76.0 / 0.5872281406276059 = 129.42159059130623`;
  measured-minus-SC gap `10.079569037049254`. The Stage 4A-6.5x context
  source OCC+FREE values remain measured `569.0` and seed0 SC `135.0`; the
  Stage 4A-6.5y saved posthoc row fields are also recorded separately in the
  gap report.
  Key result:
  for all tested fixed lambdas and all three SC bases, spatial seed0 SC basin
  fraction was `0.0` and same-as-measured fraction was `1.0`. The same was true
  for all adaptive lambda variants. Seed0 `source_occ_free` at fixed lambda
  `0` selected measured `n0001 -> n0112`.
  Conclusion:
  decoupling the SC bonus from the cost division eliminated the short low-cost
  seed0 SC basin in this saved-tree offline diagnostic sweep. This is not
  source-faithful and does not establish coverage improvement. Runtime smoke
  and rollout are still not recommended directly from this diagnostic alone.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65z_decoupled_sc_utility_sweep.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65z_decoupled_sc_utility_sweep_test.log`.

- Stage 4A-6.5z.1 offline decoupled SC signal-strength and normalization
  diagnosis is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_decoupled_sc_signal_strength.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_decoupled_sc_signal_strength.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65z1_decoupled_signal_strength_diagnosis`.
  This was offline-only on saved Stage 4A-6.5z tables, saved Stage 4A-6.5y
  raw mini-RRT trees, and saved Stage 4A-6.5p Frame2 observed_state /
  prediction NPZ / pose / camera info. It did not start Isaac, capture
  RGB/depth, rerun map_predict, run SSCNet inference, execute actions, run
  two-frame runtime, run rollout, train, modify checkpoints/observed_state/
  prediction NPZ, write prediction into observed_map, use prediction for
  traversability/collision/ray blocking, use target/ground truth, modify/build
  external source, implement a Pareto gate/runtime planner, or claim coverage
  improvement.
  Generated all required z.1 outputs and plots (`38` checked by validation),
  including near-miss branch tables, required-lambda tables, normalization
  diagnostics, adaptive-lambda gap analysis, measured-vs-nonmeasured SC ranks,
  low-cost follow-up, and debug tree regeneration reports.
  Debug regeneration:
  `recomputed_debug_tree_nodes=true`, `7650` candidate path rows,
  `3003` near-miss rows, `4320` non-measured required-lambda rows, and
  `1443` impossible-under-positive-lambda rows.
  Key finding:
  the Stage 4A-6.5z `lambda_sweep_summary_by_basis_variant` claims all tested
  decoupled variants were measured-only, but corrected per-formula/per-seed
  classification rows do not support that claim. For example,
  `decoupled_source_occ_free_fixed_0` has same-as-measured fraction `0.8` and
  spatial seed0-SC-basin fraction `0.2`; `decoupled_source_occ_free_fixed_32`
  has same-as-measured fraction `0.6` and spatial seed0-SC-basin fraction
  `0.1`. The z.1 outputs preserve both the original summary claim and the
  corrected row-level diagnosis.
  Required-lambda result:
  finite required-lambda p50/p90/max were
  `229.31585862120286 / 627.9926880897762 / 34462.89245592027`;
  only `111` finite required-lambda rows were `<=32`, and `173` were
  `<= adaptive 2x`. The seed0 reference SC branch `n0127 -> n0162` is
  impossible to recover with positive lambda under all three SC bases because
  its normalized SC is lower than the measured-like branch
  (`source_occ_free` `0.17209302325581396` vs `0.6116279069767442`;
  parent-visible-cleared `0.2012987012987013` vs `0.5487012987012987`;
  frontier-local `0.16243654822335024` vs `0.6142131979695431`).
  Normalization result:
  measured winners were already in the top SC quartile for `15/30` seed/basis
  rows, the max normalized SC belonged to a measured branch in `18/30`, and
  normalized-SC IQR was `<0.10` in `14/30`.
  Interpretation:
  current Frame2 map_predict SC signal is not cleanly branch-selective; the
  seed0 short local SC artifact is suppressed by decoupling because it has
  lower normalized SC than measured, not because a moderate lambda can promote
  it. Some non-measured branches can theoretically flip, but most require
  lambda values above `32`.
  Recommended next small task:
  larger offline lambda diagnostic sweep only. Runtime two-frame smoke,
  rollout, Pareto-gate implementation, runtime-planner implementation, and RL
  remain not recommended next.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65z1_decoupled_signal_strength_diagnosis.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65z1_decoupled_signal_strength_diagnosis_test.log`.

- Stage 4A-6.5aa controlled synthetic SC validation scene smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_synthetic_sc_validation_scene.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_synthetic_sc_validation_scene.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65aa_synthetic_sc_validation`.
  Scene variant:
  `synthetic_hidden_room_frontier`.
  This was a diagnostic-only synthetic validation, not rollout. It used exactly
  one fixed synthetic frame, no selected action execution, no two-frame runtime,
  no open-ended loop, no training/RL/PPO/BC/IL, no checkpoint modification, no
  existing observed_state modification, no existing prediction NPZ
  modification, no prediction writeback, no prediction traversability/
  collision/ray blocking, no target/ground-truth planning/scoring, no external
  source modification/build, and no coverage-improvement claim.
  Scene/capture outputs include:
  `scene_metadata.json`, `synthetic_scene_summary.*`, `rgb_000.png`,
  `depth_000.npy`, `pose_000.json`, `camera_info.json`,
  `observed_state_synthetic_frame000.npy`, `observed_state_summary.json`,
  `scene_layout_topdown.png`, and `observed_topdown.png`.
  Observed map stats:
  shape `[120, 120, 30]`, unknown/free/occupied
  `256908 / 158196 / 16896`.
  Oracle prediction outputs include:
  `oracle_global_prediction_layer.npz`, `oracle_prediction_summary.*`,
  `oracle_prediction_topdown.png`, and
  `oracle_prediction_overlay_topdown.png`.
  Oracle prediction stats:
  valid `52184`, predicted occupied `11354`, predicted free `40830`,
  hidden valid `44990`, read-only diagnostic, no writeback, no collision/
  traversability/ray blocking.
  Real map_predict was run once on the same saved frame:
  `map_predict/global_prediction_layer.npz`.
  Map_predict stats:
  valid `59904`, predicted occupied `2530`, predicted free `34392`,
  hidden valid `16280`, read-only, no writeback, no collision/
  traversability/ray blocking.
  Tree decisions completed for seeds `0..4` with `45` decision rows.
  Measured-only selected the measured-frontier direction in `5/5` seeds.
  Oracle source OCC+FREE over-cost selected the hidden-room direction in `5/5`
  seeds. Oracle decoupled minmax selected hidden-room direction at lambda `32`
  in `4/5` seeds, but lambda `8` and `16` stayed measured-frontier in `5/5`
  seeds.
  Map_predict source OCC+FREE over-cost selected hidden-room direction in
  `5/5` seeds. Map_predict decoupled minmax selected hidden-room direction at
  lambda `32` in `3/5` seeds, but lambda `8` and `16` stayed
  measured-frontier in `5/5` seeds. Map/oracle direction agreement fraction
  was `0.95`.
  Low-cost artifact flags:
  `0/45`, fraction `0.0`.
  Summary conclusion:
  Oracle and map_predict both produced useful hidden-room signal in this
  diagnostic, but the stage remains diagnostic-only and does not justify
  runtime smoke or rollout by itself.
  Recommended next small task:
  repeat a tiny controlled map_predict calibration smoke before any runtime
  smoke.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aa_synthetic_sc_validation.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aa_synthetic_sc_validation_test.log`.

- Stage 4A-6.5ab tiny controlled map_predict calibration smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_synthetic_map_predict_calibration_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_synthetic_map_predict_calibration_smoke.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ab_synthetic_calibration_smoke`.
  This was an offline replay over the saved Stage 4A-6.5aa
  `synthetic_hidden_room_frontier` frame and raw mini-RRT trees only. It did
  not start Isaac, capture frames, rerun map_predict, run SSCNet inference,
  execute actions, run two-frame runtime, run rollout/open-ended loops, train,
  modify checkpoints, modify existing observed_state/prediction NPZ, write
  prediction into observed_map, use prediction for traversability/collision/
  ray blocking, use target/ground-truth planning/scoring, modify/build
  external source, or claim coverage improvement.
  Inputs loaded:
  `observed_state_synthetic_frame000.npy`,
  `oracle_global_prediction_layer.npz`,
  `map_predict/global_prediction_layer.npz`, `pose_000.json`,
  `camera_info.json`, `scene_metadata.json`,
  `synthetic_sc_validation_summary.json`,
  `per_seed_mode_decisions.csv`, and
  `branch_direction_classification.csv`.
  Sweep size:
  seeds `0..4`, `96` configs, `485` decision rows.
  Measured-only selected measured-frontier in `5/5` seeds.
  Best non-diagnostic candidate:
  `map_predict|source_occ_free|decoupled_minmax_lambda48|tau0p1|occ0p5|free0p5`.
  It selected hidden-room in `5/5` seeds, matched Oracle direction in `5/5`,
  had low-cost artifact fraction `0.0`, mean selected SC gain `5018.4`, mean
  hidden-region count `4527.0`, and median margin `21.285778495568792`.
  Decoupled lambda result:
  map_predict source OCC+FREE minmax lambda `32` was not stable
  (`3/5`, `2/5`, `2/5` hidden-room at tau `0.1`, `0.4`, `0.8`);
  lambda `48` was stable (`5/5` at tau `0.1`, `0.4`, `0.8`). Oracle was
  stable at lambda `48` (`5/5`) and mostly stable at lambda `32` (`4/5`).
  Over-cost result:
  source OCC+FREE over-cost stayed stable across tau `0.05..0.8` for Oracle
  and map_predict, but remains marked useful-but-risky because SC stays inside
  the cost denominator.
  Threshold result:
  map_predict hidden-room selection stayed `5/5` for source OCC+FREE over-cost
  across confidence thresholds, while hidden-region signal magnitude decreased
  at high tau. The optional map_predict occ/free sweep showed over-cost
  remained stable, but decoupled lambda `32` stayed below the `4/5` robustness
  bar under tested occ/free settings.
  Low-cost artifact result:
  the best candidate had `0.0` low-cost artifact fraction. Across all
  calibration rows there were `3/485` low-cost artifact flags, so the summary
  records that artifacts can appear in weaker diagnostic configs.
  Required outputs and plots were generated, including:
  `calibration_summary_by_config.*`, `threshold_sensitivity_summary.*`,
  `lambda_sensitivity_summary.*`, `oracle_map_predict_agreement.*`,
  `low_cost_artifact_diagnosis.*`, `hidden_region_signal_summary.*`,
  `best_config_candidates.*`,
  `stage4a65ab_synthetic_calibration_summary.*`, and 9 PNG plots.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ab_synthetic_calibration_smoke.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ab_synthetic_calibration_smoke_test.log`.
  Recommended next small task:
  saved-frame one-step formula smoke only, using the robust non-over-cost
  lambda48 source OCC+FREE candidate. Runtime smoke and rollout are still not
  ready.

- Stage 4A-6.5ac saved-frame one-step lambda48 formula smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_saved_frame_lambda48_formula_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_saved_frame_lambda48_formula_smoke.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ac_saved_frame_lambda48_formula_smoke`.
  This was an offline saved-frame replay over the saved Stage 4A-6.5aa
  synthetic frame, Oracle NPZ, map_predict NPZ, and raw mini-RRT trees only.
  It did not start Isaac, capture new RGB/depth, rerun map_predict, run
  SSCNet inference, execute a selected action, run two-frame runtime, run
  rollout/open-ended loops, train, modify checkpoints, modify existing
  observed_state/prediction NPZ, write prediction into observed_map, use
  prediction for traversability/collision/ray blocking, use target/
  ground-truth planning/scoring, modify/build external source, or claim
  coverage improvement.
  Formula:
  `gain_exp / cost + 48 * minmax(source_occ_free)`, with tau `0.1`,
  occ/free thresholds `0.5/0.5`, and seeds `0..4`.
  Results:
  measured_only reproduced measured-frontier in `5/5`;
  Oracle lambda48 selected hidden-room in `5/5`;
  map_predict lambda48 selected hidden-room in `5/5`;
  map_predict/Oracle lambda48 agreement was `1.0`;
  map_predict lambda48 low-cost artifact fraction was `0.0`;
  map_predict lambda48 median margin matched 6.5ab at
  `21.285778495568792`.
  Diagnostic modes also ran:
  map_predict lambda32 selected hidden-room in `3/5`; Oracle/map_predict
  over-cost selected hidden-room in `5/5` but remains diagnostic only.
  Required outputs and plots were generated, including loaded-input manifest,
  formula definition, per-seed decisions/value components, branch
  classification, lambda48 reproduction summary, Oracle-vs-map summary,
  low-cost diagnosis, comparison to 6.5ab, safety/hash reports, final summary,
  recommendation, and 7 PNG plots.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ac_saved_frame_lambda48_formula_smoke.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ac_saved_frame_lambda48_formula_smoke_test.log`.
  Recommended next small task:
  saved-frame formula smoke on one real `medium_three_rooms` frame only.
  Runtime smoke and rollout are still not ready.

- Stage 4A-6.5ad saved-frame lambda48 formula smoke on one real
  `medium_three_rooms` Frame2 is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_real_frame_lambda48_formula_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_real_frame_lambda48_formula_smoke.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ad_real_frame_lambda48_formula_smoke`.
  This was offline-only on saved Stage 4A-6.5p Frame2 observed_state,
  prediction NPZ, pose/camera info, saved Stage 4A-6.5y raw trees, and
  Stage 4A-6.5z/z.1 comparison outputs. It did not start Isaac, capture new
  RGB/depth, rerun map_predict, run SSCNet inference, execute selected
  actions, run two-frame runtime, run rollout/open-ended loops, train, modify
  checkpoints, modify existing observed_state/prediction NPZ, write prediction
  into observed_map, use prediction for traversability/collision/ray blocking,
  use target/ground-truth planning/scoring, modify/build external source, or
  claim coverage improvement.
  Formula:
  `gain_exp / cost + 48 * minmax(source_occ_free)`, with tau `0.1`,
  occ/free thresholds `0.5/0.5`, and seeds `0..9`.
  Results:
  completed `60` decision rows over `10` seeds and `6` modes; measured_only
  reproduced the Stage 4A-6.5p Frame2 measured reference for seed0
  (`n0001 -> n0112`). map_predict lambda48 selected same-as-measured in
  `6/10` seeds and distinct non-measured branches in `4/10`; healthy
  non-measured fraction `0.4`, prior low-cost SC basin fraction `0.0`,
  low-cost artifact fraction `0.0`, median margin `21.914349073186955`.
  Lambda32 matched lambda48 at the branch-class level in this saved-frame
  replay (`6/10` same-as-measured, `4/10` distinct, prior basin `0.0`).
  The over-cost diagnostics reproduced the old risk shape: primary
  spatial-prior classification in `3/10` rows and spatial prior basin flag
  fraction `0.5`, including seed0 returning to `n0127`.
  Required outputs and plots were generated, including loaded-input manifest,
  formula/reference definitions, per-seed decisions/value components, branch
  classification, lambda48 behavior summary, low-cost diagnosis,
  comparison-to-6.5z/z.1, safety/hash reports, final summary, recommendation,
  and 8 PNG plots.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ad_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ad_real_frame_lambda48_formula_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ad_real_frame_lambda48_formula_smoke_test.log`.
  Recommended next small task:
  saved-frame formula smoke on another real medium frame only. Runtime smoke
  and rollout are still not ready.

- Stage 4A-6.5ae saved-frame lambda48 formula smoke on another real
  `medium_three_rooms` frame is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_real_frame_lambda48_formula_smoke_another.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_real_frame_lambda48_formula_smoke_another.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ae_real_frame1_lambda48_formula_smoke`.
  This was offline-only on saved Stage 4A-6.5p Frame1 observed_state,
  prediction NPZ, pose/camera info, and historical saved-frame comparison
  outputs. It selected the preferred Frame1 inputs directly and did not fall
  back to the Stage 4A-6.5ad Frame2 frame. It did not start Isaac, capture new
  RGB/depth, rerun map_predict, run SSCNet inference, execute selected
  actions, run two-frame runtime, run rollout/open-ended loops, train, modify
  checkpoints, modify existing observed_state/prediction NPZ, write prediction
  into observed_map, use prediction for traversability/collision/ray blocking,
  use target/ground-truth planning/scoring, modify/build external source, or
  claim coverage improvement.
  Formula:
  `gain_exp / cost + 48 * minmax(source_occ_free)`, with tau `0.1`,
  occ/free thresholds `0.5/0.5`, and seeds `0..9`.
  Results:
  completed `60` decision rows over `10` seeds and `6` modes; measured_only
  reproduced the Stage 4A-6.5p Frame1 measured reference for seed0
  (`n0001 -> n0249`). map_predict lambda48 selected same-as-measured in
  `8/10` seeds and distinct non-measured branches in `2/10`; healthy
  non-measured fraction `0.2`, historical prior low-cost SC basin fraction
  `0.0`, low-cost artifact fraction `0.0`. Lambda32 matched lambda48 at the
  branch-class level in this saved-frame replay (`8/10` same-as-measured,
  `2/10` distinct, prior basin `0.0`). The over-cost diagnostic selected
  same-as-measured in `5/10` and distinct non-measured in `5/10`, with prior
  basin fraction `0.0` and low-cost artifact fraction `0.0`.
  Required outputs and plots were generated, including selected-frame report,
  loaded-input manifest, formula/reference definitions, per-seed decisions/
  value components, branch classification, lambda48 behavior summary,
  low-cost diagnosis, comparison-to-6.5z/z.1, safety/hash reports, final
  summary, recommendation, and 8 PNG plots.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ae_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ae_real_frame1_lambda48_formula_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ae_real_frame1_lambda48_formula_smoke_test.log`.
  Runtime smoke and rollout are still not ready.

- Stage 4A-6.5af offline saved-frame-only lambda48 consolidation / design
  review is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/consolidate_lambda48_saved_frame_review.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_lambda48_saved_frame_consolidation.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65af_lambda48_saved_frame_consolidation`.
  This was an offline read-only consolidation over saved Stage 4A-6.5ab,
  6.5ac, 6.5ad, and 6.5ae outputs only. It did not start Isaac, capture new
  RGB/depth, rerun map_predict, run SSCNet inference, execute selected
  actions, run two-frame runtime, run rollout/open-ended loops, train, modify
  checkpoints, modify existing observed_state/prediction NPZ, write prediction
  into observed_map, use prediction for traversability/collision/ray blocking,
  use target/ground-truth planning/scoring, modify/build external source, add
  a runtime planner, or claim coverage improvement.
  Consolidated result:
  synthetic map_predict lambda48 selected hidden-room `5/5`, oracle agreement
  `1.0`, and low-cost artifact fraction `0.0`. Across the two real saved
  medium frames, map_predict lambda48 was same-as-measured in `14/20`
  seed-frame rows, healthy non-measured in `6/20`, prior-basin `0/20`, and
  low-cost artifact `0/20`; real median margin was
  `25.005253421860232`.
  Lambda32 matched lambda48 at real branch-class level (`20/20`) but exact
  best-descendant selection matched only `13/20`; synthetic evidence still
  favors lambda48 because lambda32 was only `3/5` hidden-room.
  Over-cost remains diagnostic-only: Frame2 reproduced the old prior-basin
  risk shape with prior basin fraction `0.5`, while Frame1 was more aggressive
  without low-cost artifacts.
  Required CSV/JSON/MD outputs and 8 PNG plots were generated, including
  loaded-input manifest, missing-fields report, unified config table,
  cross-frame lambda48 summary, real-frame aggregate, lambda32-vs-lambda48
  comparison, over-cost comparison, low-cost artifact summary, readiness
  matrix, design review findings, final summary, and recommended next step.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65af_lambda48_saved_frame_consolidation.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65af_lambda48_saved_frame_consolidation_test.log`.
  Recommended next faithful small task:
  offline saved-frame-only multi-frame lambda48 replay over all available
  saved real `medium_three_rooms` frames. Runtime smoke and rollout are still
  not ready.

- Stage 4A-6.5ag offline saved-frame-only multi-frame lambda48 replay is
  complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_multi_frame_lambda48_replay.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_multi_frame_lambda48_replay.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ag_multi_frame_lambda48_replay`.
  It discovered `20` candidate frame rows, `17` valid saved real-frame
  candidates, selected `7` unique real medium frames after deduplication, and
  recorded `10` duplicate rows plus skipped incomplete/synthetic candidates.
  Replay used seeds `0..9`, `6` modes, and `420` total decision rows, with
  `70` map_predict lambda48 seed-frame rows.
  Lambda48 aggregate:
  same-as-measured `33/70`, distinct non-measured `35/70`, local jitter
  `2/70`, healthy non-measured `35/70`, historical prior basin `0/70`, and
  low-cost artifact `0/70`; median margin `18.93872168517339`.
  Lambda32-vs-lambda48:
  branch-class agreement `62/70`, selected-child agreement `61/70`, and
  best-descendant agreement `41/70`; synthetic calibration still favors
  lambda48.
  Over-cost remains diagnostic-only: source_occ_free_over_cost had prior
  basin fraction `24/70` despite low-cost artifact `0/70`.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ag_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ag_multi_frame_lambda48_replay.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ag_multi_frame_lambda48_replay_test.log`.
  Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no two-frame runtime, no
  rollout/open-ended loop, no training/RL/PPO/BC/IL, no checkpoint or existing
  observed_state/prediction NPZ modification, no prediction writeback, no
  prediction traversability/collision/ray blocking, no target/ground-truth or
  future-observed planning/scoring, no external source modification/build, and
  no coverage-improvement claim.
  Recommended next faithful small task:
  multi-scene/start saved-frame replay if available, or staged one-frame
  runtime-smoke design review only; still no rollout.

- Stage 4A-6.5ah hardware-aware multi-scene/start saved-frame discovery or
  staged runtime-smoke design review is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65ah_multiscene_or_design_review.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ah_multiscene_or_design_review.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ah_multiscene_or_runtime_design_review`.
  It read existing saved artifacts only and did not start Isaac, capture new
  RGB/depth, rerun map_predict, run SSCNet inference, execute selected
  actions, run two-frame runtime, run rollout/open-ended loops, train, modify
  checkpoints, modify existing observed_state/prediction NPZ, write
  prediction into observed_map, use prediction for traversability/collision/
  ray blocking, use target/ground-truth or future-observed planning/scoring,
  modify/build external source, implement Pareto dominance gate, implement a
  runtime planner, or claim coverage improvement.
  Discovery result:
  `217` candidate rows, `7` already in Stage 4A-6.5ag, `55` duplicates or
  same-frame prediction variants, `127` missing prediction, `26` missing
  pose/camera, `2` synthetic/forbidden, and `0` new complete frames.
  Because no additional complete saved real frames were available beyond
  Stage 4A-6.5ag's seven frame identities, Stage 4A-6.5ah wrote a staged
  one-frame runtime-smoke design review only.
  Hardware report:
  `os_cpu_count=32`, requested `--max_workers 32`,
  `actual_max_workers=32`, `parallel_backend=ProcessPoolExecutor`,
  hash task count `62`, process workers used, and BLAS/OMP inner thread
  variables recorded as `1` to avoid oversubscription.
  Design review next choices:
  if runtime is desired next, use
  `Stage 4A-6.5ai staged one-frame lambda48 runtime smoke, no action
  execution`; otherwise collect additional saved frames in a controlled
  capture-only stage, still no rollout.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ah_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ah_multiscene_or_design_review.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ah_multiscene_or_design_review_test.log`.

- Stage 4A-6.5ai staged one-frame lambda48 runtime smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65ai_one_frame_lambda48_runtime_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ai_one_frame_lambda48_runtime_smoke.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ai_one_frame_lambda48_runtime_smoke`.
  Runtime setup:
  one Isaac startup, deterministic `medium_three_rooms` scene seed `0`,
  canonical start pose `[-4.65, -4.65, 1.2]` yaw `0.38710316317995463`,
  exactly one RGB/depth capture, exactly one measured-only observed_state
  update, exactly one `map_predict` call with `code_consistent_v1`, and no
  selected action execution, second frame, two-frame runtime, or rollout.
  Observed map:
  shape `(120,120,30)`, observed_ratio `0.0425462962962963`.
  Prediction stats:
  prediction shape aligned to observed_state, valid predictions `57382`,
  predicted unmeasured OCC+FREE `40328` (`9494` occupied, `30834` free).
  Formula:
  `gain_exp / cost + 48 * minmax(source_occ_free)` with tau `0.1` and
  occ/free thresholds `0.5/0.5`; over-cost was not executed as runtime
  primary.
  Result:
  measured-only shadow selected `n0013 -> n0159`;
  lambda48 primary selected `n0001 -> n0228`;
  lambda32 shadow selected `n0001 -> n0228`.
  Lambda48 classification:
  `distinct_nonmeasured_branch`, healthy non-measured `true`,
  low-cost artifact `false`, historical prior basin `false`.
  Hardware report:
  `os_cpu_count=32`, requested/actual `--max_workers 32/32`,
  `parallel_backend=single_process_runtime_stage_no_process_pool`,
  thread env recorded as OMP `1`, OPENBLAS `32`, MKL `1`, NUMEXPR `1`,
  VECLIB `1`, GPU `NVIDIA GeForce RTX 5080`, total wall time
  `29.64396972299801s`.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ai_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ai_one_frame_lambda48_runtime_smoke.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ai_one_frame_lambda48_runtime_smoke_test.log`.
  Safety passed: no action execution, no second frame, no rollout/open-ended
  loop, no training/RL/PPO/BC/IL, checkpoint unchanged, no existing
  observed_state or prediction NPZ modified, prediction stayed read-only and
  was not written/fused into observed_state, prediction was not used for
  traversability/collision/ray blocking/candidate sampling/edge validity, no
  target/ground-truth/future-observed planning/scoring, no external source
  modification/build, no Pareto gate/runtime planner implementation, no
  over-cost runtime promotion, and no coverage-improvement claim.
  Current next small task:
  `Stage 4A-6.5aj staged two-frame one-action lambda48 runtime smoke design
  review only`, or controlled capture-only additional saved-frame collection.
  Still no rollout.

- Stage 4A-6.5aj staged two-frame one-action lambda48 runtime smoke design
  review is complete:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/design_stage4a65aj_two_frame_one_action_runtime_smoke.py`.
  Test:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65aj_two_frame_one_action_design_review.py`.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65aj_two_frame_one_action_runtime_design_review`.
  This was design review only: no Isaac startup, no RGB/depth capture, no
  map_predict call, no SSCNet inference, no selected action execution, no
  two-frame runtime, no rollout/open-ended loop, no training/RL/PPO/BC/IL, no
  checkpoint or existing observed_state/prediction NPZ modification, no
  prediction writeback/fusion, no external source build, no over-cost runtime
  primary, and no coverage-improvement claim.
  It loaded Stage 4A-6.5ag / 6.5ah / 6.5ai context and reviewed the clean
  Stage 4A-6.5ai one-frame runtime result:
  lambda48 selected `n0001 -> n0228`, branch
  `distinct_nonmeasured_branch`, low-cost artifact `false`, historical prior
  basin `false`, and rollout_ready `false`.
  Future Stage 4A-6.5ak design:
  exactly two frames, exactly two measured-only observed_state updates,
  exactly two map_predict calls, exactly one Frame 1 lambda48-selected action
  if pre-action gates pass, no second action, no third frame, and no rollout.
  Future primary formula remains:
  `gain_exp / cost + 48 * minmax(source_occ_free)`, with SC outside the cost
  denominator; over-cost remains prohibited for runtime primary.
  Required future safety gates:
  no low-cost artifact, no historical prior basin unless action is blocked,
  prediction read-only, no prediction traversability/collision/ray
  blocking/candidate sampling/edge validity, no target/ground-truth/
  future-observed planning or scoring, finite in-bounds selected action, and
  action_execution_count `0` before Frame 1 action.
  Hardware plan:
  future command includes `--max_workers 32`; process-pool helper workers
  keep BLAS/OMP inner threads at `1`.
  Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aj_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aj_two_frame_one_action_design_review.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aj_two_frame_one_action_design_review_test.log`.
  Current next small task:
  `Stage 4A-6.5ak staged two-frame one-action lambda48 runtime smoke execution`
  only if explicitly requested by user.
  Still not next:
  rollout, online open-ended loop, RL/PPO/BC/IL, prediction writeback/fusion,
  prediction traversability/collision/ray blocking, target/ground-truth or
  future-observed scoring, checkpoint changes, external source build, runtime
  planner implementation, over-cost runtime promotion, or coverage-improvement
  claim.

Hardware utilization policy for future offline replay/analysis stages:
use maximum available CPU parallelism by default. On this workstation,
CPU-bound offline stages should request `--max_workers 32` and use
`actual_max_workers=min(32, os.cpu_count() or 1)`. Process-pool workers
should set BLAS/OMP inner threads to `1` to avoid oversubscription; single
process numeric runs may set BLAS/OMP/torch threads to `32`. Every stage
should log requested/actual workers, `os.cpu_count()`, parallel backend,
task count, wall time, worker/process/thread mode, and thread environment
variables.

Stage 4A-6.5ak update / current next:

- Stage 4A-6.5ak staged two-frame one-action lambda48 runtime smoke execution
  is complete and validated:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke.py`.
- Runtime setup:
  one Isaac startup, `medium_three_rooms` seed `0`, canonical start pose
  `[-4.65, -4.65, 1.2]`, yaw `0.38710316317995463`, exactly `2`
  RGB/depth frames, exactly `2` measured-only observed_state updates,
  exactly `2` map_predict calls, exactly `1` lambda48-selected action,
  no second action, no third frame, and no rollout/open-ended loop.
- Frame 1:
  measured-only shadow selected `n0013 -> n0159`; lambda48 primary selected
  `n0001 -> n0228`; lambda32 shadow selected `n0001 -> n0228`.
  Lambda48 classification was `distinct_nonmeasured_branch`, healthy
  non-measured `true`, low-cost artifact `false`, historical prior basin
  `false`; all pre-action safety gates passed.
- Executed action:
  pose `[-4.15, -4.55, 1.2]`, yaw `1.7681918866447788`, planar fixed-height
  teleport to the lambda48 selected child.
- Frame 2:
  measured-only shadow selected `n0014 -> n0108`; lambda48 diagnostic selected
  `n0002 -> n0158`; lambda32 shadow selected `n0002 -> n0158`.
  Lambda48 classification was `distinct_nonmeasured_branch`, healthy
  non-measured `true`, low-cost artifact `false`, historical prior basin
  `false`.
- Hardware report:
  `os_cpu_count=32`, requested/actual max_workers `32/32`,
  OMP/OPENBLAS/MKL/NUMEXPR/VECLIB threads `1/1/1/1/1`, GPU
  `NVIDIA GeForce RTX 5080`, total wall time `57.02001969000048s`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ak_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ak_two_frame_one_action_lambda48_runtime_smoke.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ak_two_frame_one_action_lambda48_runtime_smoke_test.log`.
- Safety passed:
  checkpoint unchanged, no existing observed_state modified, generated
  prediction NPZs unchanged after creation, prediction stayed read-only and
  information-gain-only, no prediction writeback/fusion, no prediction
  traversability/collision/ray blocking/candidate sampling/edge validity, no
  target/ground-truth/future-observed planning/scoring, no training/RL/PPO/
  BC/IL, no external source modification/build, no over-cost runtime primary,
  and no coverage-improvement claim.
- Current next small task:
  `Stage 4A-6.5al post-action/two-frame diagnosis and repeat-safety review
  only`. Do not recommend rollout directly.

Stage 4A-6.6c-usd-defaultprim-fix result:

- Output: `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_defaultprim_fix`.
- Previous dependency closure remains complete: 67 initial URLs, 278 package files, remote refs 0, omniverse refs 0, unresolved local deps 0.
- Previous Isaac blocker was the localized USD missing `defaultPrim` / concrete spawn target.
- Chosen fix: `defaultPrim=World` on `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd`; wrapper used `False`.
- scene_factory points to `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd`; larger_complex_scene_v1 remains disabled.
- Isaac validation attempted: `True`; attempt_count `1`; result `succeeded`.
- RGB/depth summary: `{'validation_rgb_count': 20, 'validation_depth_count': 20, 'inspection_rgb_count': 36, 'inspection_depth_count': 36}`.
- observed_state_final: `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_final.npy`.
- visual HTML: `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_defaultprim_fix/visual_inspection_index.html`.
- MP4/flythrough: `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_defaultprim_fix/usd_scene_flythrough.mp4`.
- Gates remain closed: `human_visual_inspection_done=false`, `formal_expert_sampling_ready=false`, `full_expert_dataset_ready=false`, `stage4a66d_executed=false`, `stage4a67_executed=false`.
- No rollout, selected action, expert sampling, map_predict, SSCNet inference, prediction NPZ, checkpoint change, or RL/GDPO/PPO/BC/IL was run.
- Next: `Stage 4A-6.6d USD scene audit + human visual review`.

Stage 4A-6.6c-camera-pose-fix result:

- Previous USD defaultPrim/dependency validation was successful: defaultPrim `/World`, dependency closure complete, Isaac load/RGB/depth/observed_state succeeded.
- User-reported issue: previous camera/validation/inspection/start poses were outside the house/interior.
- Corrected output dir: `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_camera_pose_fix`.
- Corrected visual HTML: `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_camera_pose_fix/visual_inspection_index.html`.
- Corrected MP4/flythrough: `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_camera_pose_fix/usd_scene_flythrough.mp4`.
- Corrected observed_state_final.npy: `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_camera_pose_fix/observed_state_final.npy`.
- Interior validation poses: `20`.
- Interior inspection poses: `36`.
- Interior start proposals: `10`.
- Rejected exterior/suspect previous poses: `59`.
- RGB/depth validation passed: `True`.
- observed_state validation passed: `True`.
- Gates: `human_visual_inspection_done=false`, `user_needs_to_review_visuals=true`, `formal_expert_sampling_ready=false`, `full_expert_dataset_ready=false`, `stage4a66d_executed=false`, `stage4a67_executed=false`.
- Next: user should review corrected HTML/MP4. If accepted, proceed to Stage 4A-6.6d USD scene audit + human visual review. If rejected, manually adjust camera/start poses or revise USD.
- No rollout, expert sampling, map_predict, SSCNet inference, selected action, prediction NPZ, checkpoint change, or RL/GDPO/PPO/BC/IL was run.
