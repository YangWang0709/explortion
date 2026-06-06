Project TODO
Updated: 2026-06-06

Current:

- Current next:
  Stage 4A-7.6 manual topdown review packet is complete. User should manually
  review the 30 rows in `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a76_stage4a72_manual_topdown_review_packet/stage4a72_topdown_review_index.html` and return
  exported review JSON or edited CSV/JSON. Future Stage 4A-7.7 may import review
  results and decide promotion policy. Do not train, checkpoint, promote labels,
  run rollout/runtime, or use RL/GDPO/PPO unless a future explicit approval changes
  policy.

- Stage 4A-7.2 second bounded short rollout data-expansion design is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a72_second_bounded_short_rollout_data_expansion_design`.
- Design result:
  design-only, not run. Proposed bounds are up to `10` starts, max `3` decision steps per start, max `40` captures, same primary expert policy `stage4a613_uncertainty_bonus_executed_primary` from `uncertainty_bonus_composite_beta8`.
- Safety boundary:
  close guard required, lambda48 shadow/baseline only, no long rollout, no
  training, no checkpoint creation, no RL/GDPO/PPO, no prediction/uncertainty
  writeback.
- Historical next note at that time:
  offline small-variation start proposal and QA are complete. If continuing,
  run only the reviewed bounded Stage 4A-7.2 runtime envelope with close guard,
  no long rollout, no checkpoint/training/RL, and primary expert policy
  `stage4a613_uncertainty_bonus_executed_primary`.

- Stage 4A-7.1b LOOSO tiny evaluation QA review is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a71b_looso_tiny_eval_qa_review`.
- QA result:
  `looso_tiny_eval_qa_passed=True`.
  Fold-wise no-checkpoint plumbing works, but metrics are weak/high-variance:
  mean eval top1 `0.13333333333333333`, zero-top1 folds
  `6`, eval loss mean/stdev
  `3.9079074382781984`/`0.400544809512944`.
- Interpretation:
  this is still not policy quality, generalization, full BC training,
  checkpoint readiness, rollout readiness, or RL readiness evidence.
- Historical selected next at that time:
  `Option B bounded data expansion design review before deeper training`.
  Reason: training mechanics work, but the 30-sample dataset and high variance
  make data expansion a better next move than more optimizer depth.

- Stage 4A-7.1b no-checkpoint LOOSO tiny evaluation execution is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a71b_looso_tiny_eval_no_checkpoint`.
- Execution:
  `10` folds, one tiny epoch per fold, batch size
  `8`, forward/backward/optimizer counts
  `40/40/40`.
- Validation:
  post-run validation `all_passed=True`;
  checkpoint created `False`; source hashes unchanged
  `True`; labels recomputed from lambda48
  `False`.
- Aggregate smoke metrics:
  mean eval loss `3.9079074382781984`, mean eval
  top1 `0.13333333333333333`, mean eval top3
  `0.3666666666666666`, mean eval MRR
  `0.2794476287343797`.
- Interpretation:
  still a bounded no-checkpoint tiny evaluation, not full BC training and not
  a generalization or policy-quality claim. Next conservative step is QA review
  of fold variance and failure modes.

- Stage 4A-7.1b no-checkpoint leave-one-start-out tiny evaluation design is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a71b_looso_tiny_eval_design_no_checkpoint`.
- Design result:
  `10` LOOSO folds over the 10 start variants, with train
  samples `27` and eval samples `3` per fold.
- Boundary:
  design-only at this point; primary label remains `expert_action_index_primary`
  as `stage4a613_uncertainty_bonus_executed_primary`; lambda48 remains shadow/baseline only;
  checkpoint policy is no checkpoint/model/state_dict save.
- Historical next action at that time:
  execute the bounded no-checkpoint LOOSO tiny evaluation within this design,
  then QA-review fold variance. This remains not full BC training.

- Stage 4A-7.1 bounded tiny dry-run QA review is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a71_tiny_dryrun_qa_review`.
- Review result:
  `tiny_dryrun_qa_passed=True`. The review accepts the run only as a bounded no-checkpoint plumbing smoke:
  dataset loading, primary-label loading, candidate masking, masked CE,
  backward, and optimizer step ran with finite losses.
- Non-claims:
  no policy-quality claim, no generalization claim, no full BC training claim,
  no checkpoint-readiness claim, and no rollout/RL readiness claim.
- Hard constraints carried forward:
  `expert_action_index_primary` remains the only primary BC label; do not
  recompute labels from lambda48; lambda48 remains shadow/baseline only; no
  checkpoint by default; no full BC training; no long rollout; no RL/GDPO/PPO.
- Conservative next step selected without asking:
  `Stage 4A-7.1b no-checkpoint leave-one-start-out tiny evaluation design`.

- Stage 4A-7.1 bounded tiny dry-run execution with no checkpoint is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a71_bounded_tiny_dryrun_no_checkpoint`.
- Completed task:
  ran the approved bounded tiny dry-run using `expert_action_index_primary`
  from Stage 4A-7.0 as `stage4a613_uncertainty_bonus_executed_primary`. Labels were not
  recomputed from lambda48; lambda48 remained shadow/baseline only.
- Execution counts:
  epochs `1`, batch size `8`,
  train samples `21`, forward/backward/optimizer
  counts `3/3/3`.
- Validation:
  post-run validation `all_passed=True`;
  all batch losses finite `True`; source hashes
  unchanged `True`; checkpoint created
  `False`; no checkpoint-like files were found.
- Metrics snapshot:
  final train/val/test losses `4.102459907531738` /
  `3.34206223487854` /
  `3.8099067211151123`.
- User direction:
  User requested no further questions; proceed according to Codex conservative project method and append reports of actions taken.
- Historical recommended next task:
  perform a bounded QA review of this tiny run and continue conservatively
  within no-checkpoint/no-full-training/no-rollout/no-RL boundaries unless a
  later explicit instruction changes those hard constraints.

- Stage 4A-7.1 primary-label lineage audit and design QA review are complete.
  Outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a71_primary_label_lineage_audit` and `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a71_bc_design_qa_review`.
- User hard requirement enforced:
  the primary BC label must be `expert_action_index_primary` loaded from the
  Stage 4A-7.0 dataset with policy `stage4a613_uncertainty_bonus_executed_primary`. It traces to Stage 4A-6.13
  executed primary actions selected by `uncertainty_bonus_composite_beta8`. Lambda48 is shadow/baseline
  only and must not be used to recompute primary labels.
- Validation:
  lineage audit passed, text validation scanned Stage 4A-7.1 artifacts with
  `violation_count=0`, design QA review remains passed, and the primary-vs-
  lambda48 shadow arrays differ on `4` samples, confirming they are not treated
  as an aliased primary source.
- Future execution gate updated:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a71_tiny_dryrun_execution_approval_packet`.
  To proceed, use the exact approval phrase now recorded there:
  `Approve bounded Stage 4A-7.1 tiny dry-run execution with no checkpoint, using expert_action_index_primary from Stage 4A-7.0 as stage4a613_uncertainty_bonus_executed_primary only. Do not recompute labels from lambda48; do not run full BC training, long rollout, checkpoint creation, or RL/GDPO/PPO.`
- Negative scope:
  no tiny training execution, no backward pass, no optimizer step, no model
  save, no checkpoint creation, no Isaac startup, no rollout, and no
  RL/GDPO/PPO.

- Stage 4A-7.1 BC dry-run/tiny training design is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a71_bc_dryrun_tiny_training_design`.
- Completed task:
  produced a bounded design-only package for candidate-set BC dry-run/tiny
  training from the validated Stage 4A-7.0 dataset. This was design and
  contract validation only, not tiny training execution.
- Key design files:
  `stage4a71_bc_dryrun_tiny_training_design.md`,
  `dataset_contract.md/json`, `model_contract.md/json`,
  `loss_metrics_plan.md/json`, `leakage_guard.md/json`,
  `execution_boundary.md/json`, `design_validation_report.json`, and
  `no_runtime_no_training_no_checkpoint_report.json`.
- Design contract:
  input `candidate_features_model` shape `[30, 64, 16]`, primary label
  `expert_action_index_primary` shape `[30]`, valid-candidate mask `[30, 64]`,
  model logits `[B, 64]`, and masked 64-way cross entropy.
- Validation:
  design-only validation reports `all_passed=true`; JSON artifacts parse,
  manifest files exist, labels are in range, selected labels are valid
  candidates, normalization feature names match, forbidden-field audit remains
  passed, and negative-scope flags are clean.
- Negative scope:
  no tiny training execution, no backward pass, no optimizer step, no model
  save, no checkpoint creation, no Isaac startup, no capture, no map_predict,
  no SSCNet inference, no action execution, no rollout, no long rollout, and
  no RL/GDPO/PPO.
- Historical recommended next task:
  review the Stage 4A-7.1 design package. If accepted, explicitly approve a
  bounded tiny dry-run execution with no checkpoint by default, or return to
  Option B data expansion. Do not run full BC training, long rollout,
  checkpoint creation, or RL/GDPO/PPO without separate explicit approval.

- Stage 4A-7.0 next-step approval packet is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_next_step_approval_packet`.
- Completed task:
  created a bounded approval/decision packet after the Stage 4A-7.0 QA review,
  preserving the explicit gate instead of starting Stage 4A-7.1 or a second
  rollout without user approval.
- Packet contents:
  `next_step_approval_packet.md`, `next_step_approval_packet.json`,
  `option_a_stage4a71_bc_dryrun_design_gate.md`,
  `option_b_second_short_rollout_variation_gate.md`,
  `exact_approval_phrases.md`, and
  `no_runtime_no_training_no_rollout_report.json`.
- Recommended default:
  Option A, Stage 4A-7.1 BC dry-run/tiny training design, because it is the
  lower-runtime-risk next step after QA. This recommendation is not an
  execution approval.
- Current gate:
  waiting for explicit user choice. Do not run Stage 4A-7.1, second rollout,
  long rollout, full BC training, checkpoint creation, or RL/GDPO/PPO until
  the user approves a bounded option.

- Stage 4A-7.0 BC dataset QA review is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_qa_review`.
- Completed task:
  reviewed the existing Stage 4A-7.0 BC dataset design/preparation package
  against current artifacts, NPZ/schema/count evidence, visual QA files,
  forbidden-field safety, negative-scope reports, hash safety, and a fresh
  validation rerun. This was review/audit only, not Stage 4A-7.1 and not
  training or rollout.
- Validation evidence:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_qa_review/stage4a70_validation_rerun.json`
  reports return code `0` and `all_passed=true`.
- Review result:
  `qa_review_passed=true`; all required artifacts were present and non-empty,
  `17` PNG QA assets had readable headers, forbidden-field audit passed,
  forward-only smoke stayed no-backward/no-optimizer/no-checkpoint, source USD,
  fixed USD, checkpoint, and prior datasets were unchanged, and negative-scope
  reports stayed clean.
- Historical recommended next task:
  choose explicitly between Stage 4A-7.1 BC dry-run/tiny training design and a
  second bounded short rollout with small variations. Do not jump directly to
  long rollout, full BC training, or RL/GDPO/PPO.

- Stage 4A-7.0 BC dataset design/preparation is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_design_preparation`.
- Completed task:
  built an offline BC-ready candidate-set dataset schema/converter/audit
  package from existing validated expert artifacts. This was not BC training,
  not IL training, not rollout, and not policy checkpointing.
- Negative scope:
  no real Isaac startup, no capture, no map_predict, no SSCNet inference, no
  action execution, no rollout, no long rollout, no BC training, no optimizer
  step, no model checkpoint, and no RL/GDPO/PPO.
- Primary label policy:
  `stage4a613_uncertainty_bonus_executed_primary`, preserving Stage 4A-6.13
  executed `uncertainty_bonus_composite_beta8` actions as the default BC
  candidate-set labels.
- Dataset counts:
  primary samples `30`, starts `10`, sequence steps `[0, 1, 2]`, padded
  candidate rows `1920`, `D_raw=91`, `D_model=16`, valid primary labels `30`.
  Quality masks: `strict_keep=30`, `moderate_keep=30`,
  `analysis_only=30`.
- Split policy:
  leave-one-start-out plus split-by-start-variant (`train=21`, `val=6`,
  `test=3`) with `10` leave-one-start-out folds.
- Key outputs:
  primary dataset
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_design_preparation/bc_dataset_primary_short_rollout.npz`;
  shadow dataset
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_design_preparation/bc_dataset_shadow_multilabel.npz`;
  one-action reference dataset
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_design_preparation/bc_dataset_one_action_reference.npz`;
  feature stats
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_design_preparation/normalization_stats.npz`;
  visual HTML
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a70_bc_dataset_design_preparation/bc_dataset_index.html`.
- Audits:
  forbidden field audit passed; no target/ground-truth/future-observed feature,
  label, score, or filter use. Prediction and uncertainty remained
  candidate-feature records only, with no observed_state writeback.
- Forward-only smoke:
  ran with CE loss `4.157931327819824`; optimizer step `false`, backward
  `false`, model saved `false`, checkpoint created `false`.
- Hash safety:
  source USD, fixed USD, checkpoint, Stage 4A-6.13 dataset/manifest, and prior
  6.7/6.8/6.11/6.12 datasets unchanged.
- Historical recommended next task:
  review the Stage 4A-7.0 dataset QA package. If accepted, explicitly approve
  Stage 4A-7.1 BC dry-run/tiny training design, or choose a second bounded
  short rollout with small variations for data expansion. Do not jump directly
  to long rollout, full BC training, or RL/GDPO/PPO.

- Stage 4A-6.13a Isaac close timeout lifecycle hardening is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a613a_isaac_close_guard_hardening`.
- Completed task:
  implemented a reusable Isaac lifecycle guard, process-level close supervisor,
  fake child tests, required reports, process/GPU snapshots, orphan scan
  reporting, and a future finalization-sentinel hook in the Stage 4A-6.13
  runner. This was engineering stability hardening only, not a rollout and not
  training.
- Counts for this hardening stage:
  `isaac_startup_count_this_stage=0`, `capture_count_this_stage=0`,
  `map_predict_calls_this_stage=0`,
  `sscnet_inference_calls_this_stage=0`,
  `action_execution_count_this_stage=0`,
  `rollout_executed_this_stage=false`,
  `long_rollout_executed_this_stage=false`,
  `training_executed_this_stage=false`, and
  `bc_il_rl_gdpo_ppo_executed_this_stage=false`.
- Fake child test results:
  clean exit passed with `close_status=clean_exit`; hang after safe
  finalization was terminated by the supervisor and classified as
  `success_with_close_hang=true`; hang before finalization was terminated and
  classified as `failed_before_finalization`.
- Safety:
  supervisor termination is restricted to its own child process group, orphan
  scanning is report-only by default, and no unrelated process kill list was
  produced. Source USD, fixed USD, checkpoint, Stage 4A-6.13 dataset, and Stage
  4A-6.13 manifest hashes were unchanged.
- Historical recommended next task:
  review the 6.13 visual/audit package. If clean, choose either BC dataset
  design/preparation or a second explicitly approved short rollout with small
  variations. Do not jump directly to long rollout; any future long rollout
  must use the close guard and include expert data quality visualization/audit
  outputs.

- Stage 4A-6.13 uncertainty-bonus bounded short rollout pilot is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a613_uncertainty_bonus_short_rollout_pilot`.
- Completed task:
  explicitly approved 10-start, max-3-action-per-start short rollout using
  `uncertainty_bonus_composite_beta8` as the primary expert, with measured-only,
  lambda48, and confidence-gated shadows. This was not a long rollout, not a
  full expert dataset, and not training/BC/IL/RL/GDPO/PPO.
- Counts:
  `start_count=10`, `max_decision_steps_per_start=3`,
  `decision_frame_count=30`, `terminal_frame_count=10`,
  `capture_count=40`, `map_predict_calls=30`,
  `dense_uncertainty_artifacts=30`, and `executed_action_count=30`.
  Isaac startup count was `1`; `simulation_app.close()` hung after all files
  were finalized and the process was terminated without a second Isaac startup.
- Result summary:
  observed ratio mean start/end `0.2444367224367224 ->
  0.262136036036036`; total newly observed voxels `412571`, mean `41257.1`
  per start. Done reasons: `max_steps_reached: 10`.
  `no_valid_candidate_count=0`, `stuck_revisit_count=0`,
  `candidate_all_local_count=0`, low-cost artifact `0`, historical-prior
  basin `0`, and formula-dominated-by-uncertainty `0`.
- Decision summary:
  action changed count is `17` vs measured-only, `3` vs lambda48, `6` vs
  confidence-gated, and `0` vs Stage 4A-6.12 step-0 decision. Branch counts vs
  measured are `same_as_measured=7`, `local_jitter=17`, and
  `distinct_nonmeasured_branch=6`.
- Uncertainty summary:
  selected confidence mean/min `0.8360315690272979 / 0.6457599577356558`,
  entropy mean/max `0.23500558781373931 / 0.4559122721354167`, and margin
  mean/min `0.7639459535016347 / 0.46364005667264346`.
- Quality and gates:
  expert data quality, prediction safety, uncertainty safety, rollout safety,
  runtime quality, and dataset integrity all passed with no warnings or
  blockers. No long rollout, no full expert dataset, no BC/IL/RL/GDPO/PPO, no
  training, no replay buffer, no policy checkpoint, no prediction writeback, no
  uncertainty writeback, and no source/fixed USD/checkpoint/prior dataset
  modification. Prediction/uncertainty remained scoring-only and were not used
  for traversability, collision, ray blocking, candidate validity, or edge
  validity.
- Key artifacts:
  HTML
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a613_uncertainty_bonus_short_rollout_pilot/short_rollout_uncertainty_bonus_index.html`,
  MP4
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a613_uncertainty_bonus_short_rollout_pilot/short_rollout_flythrough.mp4`,
  and dataset
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a613_uncertainty_bonus_short_rollout_pilot/short_rollout_dataset_uncertainty_bonus.npz`.
- Historical recommended next task:
  review the 6.13 visual/audit package. If clean, choose either BC dataset
  design/preparation or a second explicitly approved short rollout with small
  variations. Do not jump directly to long rollout; any future long rollout
  must include expert data quality visualization/audit outputs.

- Stage 4A-6.12 uncertainty-as-exploration-bonus decision pilot is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a612_uncertainty_exploration_bonus_pilot`.
- Completed task:
  offline uncertainty bonus beta sweep over existing Stage 4A-6.10a dense
  uncertainty artifacts and Stage 4A-6.11 measured-valid candidate features.
  Formulas tested: uncertainty bonus `fraction`, `entropy`, `low_margin`, and
  `composite` with beta values `{2,4,8,16,32}`.
- Counts:
  `start_count=10`, `candidate_rows_loaded=469`,
  `selected_action_records=10`, `action_execution_count_this_stage=0`,
  `isaac_startup_count_this_stage=0`, `capture_count_this_stage=0`,
  `map_predict_calls_this_stage=0`, `sscnet_inference_calls_this_stage=0`,
  `second_action_count=0`, `third_frame_count=0`, `rollout_executed=false`,
  and `training_executed=false`.
- Recommendation:
  `recommended_uncertainty_bonus_formula=uncertainty_bonus_composite_beta8`,
  `recommended_beta=8`, and `uncertainty_bonus_runtime_ready=true`.
- Decision summary:
  action changed count is `6` vs measured-only, `1` vs lambda48, and `1` vs
  Stage 4A-6.11 confidence-gated primary. Candidate-all-local count is `5`,
  same as lambda48. Selected confidence/entropy/margin mean values are
  `0.8623173562170562`, `0.20959808035924596`, and
  `0.8020552913679388`.
- Risk summary:
  quality and risk audits passed with no warnings or blockers. Source USD,
  fixed USD, checkpoint, source observed_state, and prior outputs were not
  modified. `source_occ_free` stayed separate from uncertainty fields.
- Key artifacts:
  HTML
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a612_uncertainty_exploration_bonus_pilot/uncertainty_bonus_index.html`,
  decision dataset
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a612_uncertainty_exploration_bonus_pilot/expert_decision_dataset_uncertainty_bonus.npz`,
  and future sketch
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a612_uncertainty_exploration_bonus_pilot/future_short_rollout_with_uncertainty_bonus_sketch.md`.
- Historical recommended next task:
  review the 6.12 decision-only visual/audit package. If accepted, design an
  explicitly approved short rollout using `uncertainty_bonus_composite_beta8`.
  Do not jump directly to long rollout.

- Stage 4A-6.11 uncertainty-aware lambda one-action pilot is complete.
  Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a611_uncertainty_aware_lambda_one_action_pilot`.
- Completed task:
  conservative uncertainty-aware primary formula
  `confidence_gated_lambda48_v1` was applied over measured-valid
  reachable-frontier candidates using real dense confidence/entropy/margin
  features from Stage 4A-6.10a. Shadows saved: measured-only, lambda48
  baseline, confidence-margin gated, uncertainty bonus beta8, uncertainty
  penalty beta8, and entropy penalty beta8.
- Counts:
  `start_count=10`, `frame_count=10`, `capture_count=10`,
  `map_predict_calls=10`, `dense_uncertainty_artifacts=10`,
  `executed_action_count=10`, `second_action_count=0`,
  `third_frame_count=0`, and `long_rollout_executed=false`.
  No new Isaac startup was needed; the stage reused validated 6.8 captures and
  6.10a dense artifacts.
- Decision summary:
  primary vs measured-only produced `same_as_measured=2`, `local_jitter=6`,
  `distinct_nonmeasured_branch=2`, `no_valid_candidate=0`.
  Action changed counts are `6` vs measured-only, `0` vs the 6.11 lambda48
  baseline shadow, and `2` vs Stage 4A-6.8 / Stage 4A-6.9 Frame1.
- Uncertainty summary:
  candidate rows `469`; candidate confidence/entropy/margin means
  `0.8604786937920237`, `0.1888305449472253`, and
  `0.8094706315352114`. Selected primary confidence/entropy/margin means
  `0.8796395396528542`, `0.19162816140892908`, and
  `0.8282547902721005`.
- Safety reminder:
  no rollout, no second action, no third frame, no long rollout, no
  BC/IL/RL/GDPO/PPO, no training, no prediction writeback, no uncertainty
  writeback, no source/fixed USD/checkpoint/prior dataset modification, and
  no prediction/uncertainty use for traversability, collision, ray blocking,
  or candidate validity.
- Historical recommended next task:
  review the 6.11 uncertainty-aware visual package and choose BC dataset
  design/preparation or an explicitly approved short rollout. Do not jump
  directly to long rollout; any future rollout must include expert data
  quality visualization and audit outputs.

- Stage 4A-6.10a dense prediction uncertainty artifact regeneration is
  complete. Dense artifact output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a610a_dense_prediction_uncertainty_artifacts`.
  Dense audit rerun output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a610a_uncertainty_audit_rerun_dense`.
- Completed task:
  map_predict dense artifact-saving contract updated; dense compact
  confidence, entropy_norm, margin, occupied_prob, free_prob, valid mask, and
  predicted-unmeasured mask fields generated from existing 6.8/6.9 depth,
  pose, camera, and observed_state files only.
- Counts:
  `logical_frame_count=30`, `physical_map_predict_regeneration_calls=30`,
  dense artifacts `30`, candidate-visible uncertainty rows `480`.
  Candidate confidence/entropy/margin means are `0.8564193916817506`,
  `0.2142625541271021`, and `0.7945013785113891`.
- Dense readiness:
  `candidate_level_uncertainty_ready=true` and
  `uncertainty_aware_expert_pilot_ready=true`. Stage 4A-6.11 is still
  `not executed`.
- Dense relationship summaries:
  source_occ_free vs uncertainty Pearson is `0.037934232555910705` for Stage
  6.8/6.9 frame1 and `-0.109843280729622` for Stage 6.9 frame2. Frame2 branch
  mean uncertainty is `0.2738123838789761` for
  `distinct_nonmeasured_branch`, `0.14163060652624285` for `local_jitter`, and
  `0.07121249474585056` for `same_as_measured`.
- Safety reminder:
  Stage 4A-6.10a did not start Isaac, capture, execute actions, rollout, run
  long rollout, train, run BC/IL/RL/GDPO/PPO, write prediction to observed
  state, or modify source/fixed USD, checkpoint, prior 6.8/6.9 datasets, old
  6.10 outputs, or source observed_state.
- Historical recommended next task:
  Stage 4A-6.11 uncertainty-aware lambda pilot design, bounded one-action only,
  not rollout. Do not jump to long rollout.

- Stage 4A-6.10 prediction uncertainty offline audit is complete in
  `summary_only_limited` mode. Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a610_prediction_uncertainty_offline_audit`.
- Loaded Stage 4A-6.8 and 6.9 outputs, plus optional Stage 4A-6.4 calibration
  and Stage 4A-6.2 diagnostics context. The audit analyzed `30` frames and
  `480` top-candidate rows, but produced `0` candidate-level uncertainty rows
  because dense probability/confidence/logit artifacts are absent.
- Key metrics:
  dense prediction/probability artifacts `0`; confidence, entropy, margin,
  low-confidence fractions, and high-entropy fractions are
  `not_available_summary_only`. Candidate `source_occ_free` mean/range is
  `48.32083333333333 / 13..85`; selected lambda48 `source_occ_free`
  mean/range is `53.6 / 36..78`; prediction density mean/range is
  `0.025579121979121978 / 0.021895323895323896..0.02678078078078078`.
  Frame2 minus Frame1 predicted-unmeasured count delta mean/range is
  `1467.9 / -9142..13754`.
- Readiness:
  `uncertainty_feature_extraction_complete=true`,
  `candidate_level_uncertainty_ready=false`, and
  `uncertainty_aware_expert_pilot_ready=false`. Main blocker:
  missing dense probability/confidence/entropy/margin fields and missing
  candidate-visible voxel probability lists.
- Historical recommended next task:
  update future map_predict artifact saving so dense confidence/probability,
  entropy, margin, and candidate-visible probability references are persisted,
  then rerun Stage 4A-6.10. Do not run Stage 4A-6.11 yet and do not jump to
  long rollout.
- Safety reminder:
  Stage 4A-6.10 did not start Isaac, capture, rerun map_predict, run SSCNet
  inference, execute actions, rollout, train, run BC/IL/RL/GDPO/PPO, write
  prediction to observed_state, or modify source/fixed USD, checkpoint, source
  observed_state, or 6.8/6.9 datasets.

- Stage 4A-6.9 bounded two-frame lambda48 pilot is complete. Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot`.
- Validated counts:
  `start_count=10`, `frame_count=20`, `capture_count=20`,
  `executed_action_count=10`, `map_predict_calls=20`,
  `predictor_loaded_once=true`, `sscnet_inference_called=true`,
  `exactly_one_action_per_start=true`, `second_action_count=0`,
  `third_frame_count=0`, `continuous_rollout_executed=false`, and
  `long_rollout_executed=false`.
  Isaac startup count is `1`; `simulation_app.close()` hung after all 20
  frame captures completed, so the process was terminated and a recovery run
  reused the captured files without a second Isaac startup.
- Lambda48 formula stayed:
  `gain_exp / cost + 48 * minmax(source_occ_free)`, `lambda_sc=48`,
  per-frame/per-start valid candidate/yaw min-max normalization, reachable
  frontier candidate selection, top_n `16`.
- Frame1 comparison:
  lambda48 vs measured shadow produced `same_as_measured=4`,
  `local_jitter=4`, `distinct_nonmeasured_branch=2`,
  `no_valid_candidate=0`, `low_cost_artifact=0`, and
  `historical_prior_basin=0`. Frame1 reproduced Stage 4A-6.8 for all 10
  starts with mean action delta `0.0m` and mean yaw delta `0.0rad`.
- Frame2 diagnostic comparison:
  lambda48 vs measured shadow produced `same_as_measured=1`,
  `local_jitter=7`, `distinct_nonmeasured_branch=2`,
  `no_valid_candidate=0`, `low_cost_artifact=0`, and
  `historical_prior_basin=0`. No second action was executed.
- Observed delta and audits:
  total Frame1-to-Frame2 newly observed voxels `132834`, mean `13283.4`, min
  `3894`, max `22129`. Frame2 candidate health passed with min candidate
  count `17`, mean candidate count `58.8`, and no valid-candidate failures.
  Prediction safety, dataset integrity, safety audit, expert data quality
  audit, and two-frame stability audit all passed. The only warning class is
  `candidate_all_local`.
- Key artifacts:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot/expert_dataset_two_frame.npz`,
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot/expert_two_frame_index.html`,
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot/expert_two_frame_flythrough.mp4`,
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot/expert_data_quality_audit.md`,
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot/two_frame_stability_audit.md`,
  and comparisons to Stage 4A-6.7 and Stage 4A-6.8.
- Historical recommended next task:
  review the 6.9 two-frame quality visual package. If clean, choose either BC
  dataset design/preparation or an explicitly approved short rollout. Do not
  jump directly to long rollout unless explicitly approved; any future long
  rollout must include expert data quality visualization and audit outputs.

- Stage 4A-6.8 map_predict/lambda48 expert pilot is complete. Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot`.
- Validated counts:
  `sample_count=10`, `capture_count=10`, `map_predict_calls=10`,
  `predictor_loaded_once=true`, `sscnet_inference_called=true`,
  `exactly_one_headless_capture_per_start=true`, and
  `exactly_one_action_per_start=true`.
  Isaac startup count is `1`; `simulation_app.close()` hung after all 10
  captures completed, so the process was terminated and a recovery run reused
  the captured files without a second Isaac startup.
- Lambda48 formula:
  `gain_exp / cost + 48 * minmax(source_occ_free)`, `lambda_sc=48`,
  per-start valid candidate/yaw min-max normalization, reachable frontier
  candidate selection, top_n `16`.
- Comparison result:
  lambda48 vs measured shadow produced `same_as_measured=4`,
  `local_jitter=4`, `distinct_nonmeasured_branch=2`,
  `no_valid_candidate=0`, `low_cost_artifact=0`, and
  `historical_prior_basin=0`. Stage 4A-6.8 vs Stage 4A-6.7 action changed
  count is `4`, mean action distance is `0.3074937611088073m`, and mean yaw
  delta is `0.6706520898196431rad`.
- Safety and quality audits passed:
  prediction writeback `false`, prediction traversability/collision/
  ray-blocking/candidate-validity use `false`, target/ground-truth/future
  observed scoring `false`, checkpoint unchanged, source USD unchanged, fixed
  USD unchanged, source observed_state unchanged. Expert data quality audit
  passed with warning class `candidate_all_local`.
- Key artifacts:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot/expert_dataset.npz`,
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot/expert_pilot_index.html`,
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot/expert_action_flythrough.mp4`,
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot/expert_data_quality_audit.md`,
  and `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot/stage4a68_vs_stage4a67_comparison.md`.
- Historical recommended next task:
  review the 6.7 vs 6.8 comparison and expert data quality visual package.
  If clean, choose either a bounded two-frame pilot or BC dataset design
  preparation. Do not jump directly to long rollout unless the user explicitly
  approves; when long rollout eventually starts, it must include expert data
  quality visualization and audit outputs.

- Stage 4A-6.6c-usd-download-official-isaac-deps is complete as a blocked
  run. Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_download_official_isaac_deps`.
  It created
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/download_stage4a66c_official_isaac_deps.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66c_official_isaac_deps.py`.
- Exact official dependency result: all 67 initial requested Isaac URLs are
  present/valid, and the localized package contains 278 files (67 USD,
  23 MDL, 187 PNG, 1 DDS; 511,952,260 bytes) under
  `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment/dependencies`.
  The localized USD is
  `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized/home_like_scene_v1.usd`.
  Post-patch scan reports remote official refs `0`, `omniverse://` refs `0`,
  unsupported external deps `0`, and unresolved local deps `0`.
- Source and original staged USD hashes remain unchanged:
  `11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b`.
  No procedural fallback, cuboid fallback, random asset, unknown-license
  substitute, or old `larger_complex_scene_v1` restoration was used.
- Isaac retry was allowed by dependency closure and attempted exactly once.
  It failed because the localized USD has no `defaultPrim`, producing an
  unresolved `<defaultPrim>` reference for `/World/HomeLikeSceneV1`. No
  validation RGB/depth, `observed_state_final.npy`, visual HTML, MP4, or
  flythrough was produced or fabricated.
- Validation:
  `python test_stage4a66c_official_isaac_deps.py --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_download_official_isaac_deps --source_usd /home/ubuntu22/sc_explorer_ws/building_scene.usd --staged_usd /home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment/home_like_scene_v1.usd --dependency_request /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_home_like_scene_validation/dependency_package_request.md --previous_output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_home_like_scene_validation --localized_root /home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized --allow_blocked_if_download_or_unresolved_dependencies --expect_no_rollout --expect_no_formal_expert_sampling --expect_no_map_predict --expect_no_rl_gdpo`
  reports `all_passed: true`.
- Current required next task: fix the localized USD load target by setting a
  concrete `defaultPrim` (likely `World`) or by spawning a concrete root prim
  path, then run a new explicitly authorized validation attempt. Keep
  Stage 4A-6.6d and Stage 4A-6.7 blocked until successful RGB/depth capture,
  measured `observed_state_final.npy`, and human visual review exist.

- Stage 4A-6.6c-usd-dependency-fix-env-corrected is complete as a blocked
  dependency re-search using the user-corrected conda environment
  `env_isaaclab` (task/report filenames retain `env_isaacsim`). Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_env_isaacsim_dependency_fix`.
- Result: `env_isaaclab` exists and imports `isaacsim`; `omni` namespace
  imports; direct `pxr` import fails unless `omni.usd.libs` is added through
  `PYTHONPATH`/`LD_LIBRARY_PATH`. Searching `env_isaaclab`, IsaacSim install
  roots, and Omniverse/cache roots found 0 trusted exact local matches for the
  67 required remote Isaac USD assets. No dependencies were copied, no USD was
  patched, no localized validation package was created, and no Isaac retry was
  executed.
- Validation:
  `python test_stage4a66c_usd_env_isaacsim_dependency_fix.py --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_env_isaacsim_dependency_fix --source_usd /home/ubuntu22/sc_explorer_ws/building_scene.usd --staged_usd /home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment/home_like_scene_v1.usd --previous_output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_home_like_scene_validation --expect_env_isaaclab --allow_blocked_if_unresolved_dependencies --expect_no_rollout --expect_no_formal_expert_sampling --expect_no_map_predict --expect_no_rl_gdpo`
  reports `all_passed: true`.
- Current required next task: provide a self-contained dependency package
  preserving `Assets/Isaac/4.5/Isaac/...` under
  `assets/home_like_scene_v1/current_environment/dependencies`, including
  transitive USD/material/texture dependencies, or explicitly allow downloading
  the exact 67 missing URLs. Do not enter Stage 4A-6.6d or Stage 4A-6.7, and
  do not run Isaac again until the dependencies are complete.

- Stage 4A-6.6c-usd-dependency-fix is complete as an analysis/blocker pass.
  It generated:
  `dependency_localization_input_summary.*`,
  `usd_dependency_expanded_report.*`,
  `missing_dependency_table.*`,
  `remote_dependency_table.*`,
  `local_dependency_candidates.*`,
  `dependency_localization_summary.*`, and
  `dependency_package_request.*` in
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_home_like_scene_validation`.
- Result: 99 remote reference occurrences, 67 unique missing remote assets,
  0 local candidates, 0 copied dependencies, 0 staged USD patches, and no
  Isaac retry. Source/staged SHA256 remains
  `11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b`.
- Current required next task: provide the dependency package requested in
  `dependency_package_request.md`, preserving the `Assets/Isaac/4.5/Isaac/...`
  tree with transitive USD/material/texture dependencies under
  `assets/home_like_scene_v1/current_environment/dependencies`, or provide a
  lighter fully local USD. Do not rerun Isaac until dependencies are complete.
- Stage 4A-6.6c-usd-import used the user-provided current project candidate
  `/home/ubuntu22/sc_explorer_ws/building_scene.usd`.
- Staged current environment:
  `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment/home_like_scene_v1.usd`.
  Source/staged sha256:
  `11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b`.
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py`,
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/import_stage4a66c_usd_home_like_scene.py`,
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66c_usd_home_like_scene.py`.
- Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_home_like_scene_validation`.
- Offline USD reports were generated successfully: 758 prims, 99 references,
  67 unresolved/missing dependencies, Z-up, `metersPerUnit=1.0`, 1 mesh prim,
  2 materials, 0 texture references, and home/interior semantic-name guesses
  including sofa/table/bed/bathroom/cabinet/chair/door/hallway/room/floor/wall.
- Isaac headless validation was attempted exactly once and is blocked:
  `LLVM ERROR: out of memory` while loading/resolving the staged USD external
  dependencies. No RGB/depth capture, no measured `observed_state_final.npy`,
  and no MP4 flythrough were produced.
- Blocker/validation evidence:
  `isaac_headless_blocker_report.*`, `scene_load_validation.*`,
  `fixed_capture_validation.*`,
  `visual_inspection_capture_validation.*`,
  `observed_state_validation_summary.*`,
  `stage4a66c_usd_home_like_scene_summary.*`, topdown PNGs, start/pose
  manifests, blocked HTML index, and manual review gate files.
- Validation:
  `python test_stage4a66c_usd_home_like_scene.py --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_home_like_scene_validation --source_usd /home/ubuntu22/sc_explorer_ws/building_scene.usd --staged_usd /home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment/home_like_scene_v1.usd --expect_old_larger_scene_disabled --expect_no_rollout --expect_no_formal_expert_sampling --expect_no_map_predict --expect_no_rl_gdpo --allow_blocked_isaac_load`
  reports `all_passed: true` for the blocked evidence. This is not a
  successful scene/capture validation.
- Current required next task:
  get/provide a self-contained local dependency package for
  `building_scene.usd`, or a lighter fully local USD, then rerun
  Stage 4A-6.6c-usd-import and its single-Isaac validation. Only after
  successful load/capture/observed-state generation can Stage 4A-6.6d audit
  and human visual confirmation proceed.
- Keep gates closed:
  `human_visual_inspection_done=false`,
  `formal_expert_sampling_ready=false`,
  `full_expert_dataset_ready=false`, and `stage4a67_executed=false`.
- Do not proceed to Stage 4A-6.7, rollout, expert sampling, map_predict,
  SSCNet inference, or RL/GDPO/PPO/BC/IL.

- Stage 4A-6.6c-build-v2 generated `home_like_scene_v1` output package has
  been deleted at user request.
- Deleted output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation`.
- Deleted temporary top-down renderer script:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/render_stage4a66c_home_like_sim_topdown.py`.
- Created/updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/build_stage4a66c_home_like_scene_v1.py`,
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py`,
  `.project_context/CURRENT_STATE.md`,
  `.project_context/TODO.md`, and `.project_context/CODEX_LOG.md`.
- Cleanup/blocking rule:
  `larger_complex_scene_v1 rejected` and disabled. The old Stage 4A-6.6 /
  6.6a / 6.6b larger-scene outputs are absent. Stage 4A-6.7 is blocked from
  using the old larger scene or any old larger-scene output bundle.
- Asset rule:
  procedural composite furniture is not the main solution. The active
  furniture set must use licensed local/downloaded mesh assets if rebuilt.
  The previous downloaded/converted Kenney asset files and their manifests
  were inside the deleted output directory and are no longer available.
- Safety/negative scope:
  no expert sampling, no expert dataset, no rollout, no selected action
  execution, no map_predict, no SSCNet inference, no prediction NPZ, no replay
  buffer, no checkpoint creation/modification, no observed_state modification
  by prediction, no RL/GDPO/PPO/BC/IL, and Stage 4A-6.7 was not executed.
- Required next task:
  wait for user direction. A new replacement scene package must be built
  before Stage 4A-6.6d audit + human visual confirmation can proceed. Keep
  `human_visual_inspection_done=false` and
  `formal_expert_sampling_ready=false`.

- Stage 4A-6.6b `larger_complex_scene_v1` GUI / visual inspection setup is
  complete and validated.
- Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection`
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/inspect_stage4a66b_gui_visual_environment.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66b_gui_visual_environment.py`.
- GUI status:
  DISPLAY `:1` existed and `xdpyinfo` responded; a bounded GUI-mode IsaacSim
  probe was attempted once, but user-visible GUI inspection was not confirmed
  (`gui_attempt_status: failed`, DRI3 presentation warnings). The visual
  fallback package is therefore the review path.
- Visual package:
  Isaac headless fallback rendering succeeded with 24 inspection RGB/depth
  views, topdown/labeled maps, warning-region maps, closeups, an HTML index,
  and MP4 flythrough.
  HTML:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection/visual_inspection_index.html`
  Flythrough:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection/larger_complex_scene_v1_flythrough.mp4`
  Checklist:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection/human_visual_review_checklist.md`
- Validation:
  `python sim_explorer/test_stage4a66b_gui_visual_environment.py --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection --stage4a66_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66_larger_complex_scene_v1_validation --stage4a66a_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66a_scene_complexity_audit --expect_no_expert_sampling --expect_no_rollout --expect_no_map_predict --expect_no_rl_gdpo --min_inspection_views 24`
  reported `all_passed: true`.
- Manual review gate:
  `human_visual_inspection_done=false` and
  `user_needs_to_review_visuals=true`. Visual approval is required before
  Stage 4A-6.7.
- Safety/negative scope:
  no expert sampling, no expert dataset, no rollout, no selected action
  execution, no map_predict, no SSCNet inference, no prediction NPZ, no replay
  buffer, no checkpoint creation/modification, no observed_state modification,
  no source build, no RL/GDPO/PPO/BC/IL, and Stage 4A-6.7 was not executed.
- Next small task:
  user reviews the HTML/images/MP4/checklist. If approved, proceed to
  Stage 4A-6.7 bounded formal expert sampling pilot, measured-only first. If
  rejected, run Stage 4A-6.6c scene visual revision / scene editing. Full
  expert dataset collection remains blocked; long-term GDPO remains future
  direction only.

- Stage 4A-6.6a `larger_complex_scene_v1` offline scene complexity audit is
  complete and validated.
- Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66a_scene_complexity_audit`
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/audit_stage4a66a_scene_complexity.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66a_scene_complexity_audit.py`.
- Result:
  `scene_complexity_audit_passed: true`,
  `scene_ready_for_formal_expert_sampling_pilot: true`,
  `formal_expert_sampling_ready_full_dataset: false`, hard blockers none, and
  all audit categories passed.
- Main warnings:
  close topology among a few starts, direct fixed-view gaps for
  `corridor_east_spur`/`room_j`, intentionally low fixed-view observed_ratio,
  multiple measured-only observed-free components, higher spur-room obstacle
  density, and measured-only Stage 4A-6.7 should precede any lambda48 read-only
  map_predict pilot.
- Safety/negative scope in 6.6a:
  no Isaac startup, no capture, no action, no rollout, no formal expert
  sampling, no expert dataset, no map_predict, no SSCNet inference, no
  prediction NPZ, no replay buffer, no policy checkpoint, and no
  RL/GDPO/PPO/BC/IL.
- Validation:
  `sim_explorer/test_stage4a66a_scene_complexity_audit.py` reported
  `all_passed: true`; Stage 4A-6.6 input hashes, including
  `observed_state_final.npy`, were unchanged.
- Next small task:
  Stage 4A-6.7 bounded formal expert sampling pilot design/execution,
  measured-only first, small/qualified start subset or all qualified starts,
  not full dataset. Do not scale rollout, do not collect a full expert
  dataset, and do not start RL/GDPO/PPO/BC/IL. Long-term GDPO remains future
  direction only.

- Stage 4A-6.6 `larger_complex_scene_v1` construction and fixed-view
  validation is complete and validated.
- Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66_larger_complex_scene_v1_validation`
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/build_stage4a66_larger_complex_scene_v1.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66_larger_complex_scene_v1.py`.
  Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py` with
  `build_larger_complex_scene_v1`.
- Scene summary:
  deterministic `scene_seed=0`, x/y bounds `[-12, 12]`, z bounds `[0, 3]`,
  voxel size `0.1`, observed_state shape `(240, 240, 30)`, 10 rooms,
  7 corridors, 21 openings, 69 walls, 52 obstacles, 9 starts, 14 fixed
  validation poses, and topology cycle rank 5.
- Validation:
  exactly one successful Isaac headless startup in the clean validation run,
  14/14 nonblank RGB captures, 14/14 finite-positive depth captures,
  measured-only final observed_ratio `0.09458275462962963`, invalid labels `0`,
  and `sim_explorer/test_stage4a66_larger_complex_scene_v1.py` reported
  `all_passed: true`.
- Safety/negative scope:
  no rollout, no open-ended loop, no selected action execution, no formal
  expert sampling, no expert dataset, no map_predict call, no SSCNet inference,
  no prediction NPZ, no prediction writeback/fusion, no target/ground-truth/
  future-observed planning/scoring, no replay buffer, no policy checkpoint, and
  no RL/GDPO/PPO/BC/IL.
- Next small task:
  Stage 4A-6.6a scene complexity audit using the 6.6 audit input bundle. Do
  not start rollout, formal expert sampling, open-ended loops, map_predict
  runtime smoke, or RL/GDPO/PPO/BC/IL.

- Stage 4A-6.5av `start_room_b` tree_seed `0` bounded two-frame one-action
  lambda48 runtime smoke is complete and validated.
- Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65av_start_room_b_bounded_smoke`
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65av_start_room_b_bounded_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65av_start_room_b_bounded_smoke.py`.
- Logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65av_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65av_start_room_b_bounded_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65av_start_room_b_bounded_smoke_test.log`.
- Runtime:
  `medium_three_rooms`, scene seed `0`, start variant `start_room_b`, pose
  `[2.75, -2.55, 1.2]`, yaw `2.7052603405912112`, repeat variant
  `start_room_b_tree_seed0`, tree_seed `0`, formula
  `gain_exp / cost + 48 * minmax(source_occ_free)`.
- Sequence:
  exactly one Isaac startup, exactly two frames, exactly two map_predict calls,
  exactly one selected action execution, no second action, no third frame, no
  rollout, no formal expert sampling, and no training/RL/GDPO/PPO/BC/IL.
- Frame 1:
  measured/lambda48/lambda32 all selected `n0001 -> n0053`,
  classification `same_as_measured`, low-cost artifact `false`, historical
  prior basin `false`; action executed to
  `[2.25, -2.4499999999999997, 1.2]`, yaw `0.588002603547567`.
- Frame 2:
  measured-only `n0167 -> n0167`, lambda48/lambda32 `n0002 -> n0200`,
  classification `local_jitter`, low-cost artifact `false`, historical prior
  basin `false`.
- observed_state/map_predict:
  observed_ratio delta `0.04165740740740741`, newly observed `17996`;
  Frame 1 valid/OCC+FREE `60060 / 53080`, Frame 2 `52286 / 33383`,
  density ratio `0.6289186134137151`, no explosion/collapse.
- Outcome:
  `spatially_consistent_healthy_start_room_b`, clean but still not
  rollout-ready and not formal expert sampling-ready.
- Next small task:
  Stage 4A-6.6 `larger_complex_scene_v1` construction and validation, then
  Stage 4A-6.6a scene complexity audit. Do not go directly to rollout, formal
  expert sampling, open-ended loops, third-frame/two-action runtime, or
  RL/GDPO/PPO/BC/IL.

- Stage 4A-6.5at start_corridor seed0/seed1 repeat-comparison diagnosis and
  next-start design is complete and validated.
- Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65at_start_corridor_seed01_review_next_start_design`
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/review_stage4a65at_start_corridor_seed01_next_start_design.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65at_start_corridor_seed01_next_start_design.py`.
- Logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65at_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65at_start_corridor_seed01_review_next_start_design.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65at_start_corridor_seed01_review_next_start_design_test.log`.
- 6.5at was diagnosis/design only: no Isaac, no capture, no map_predict, no
  SSCNet inference, no action, no two-frame runtime execution, no rollout, no
  open-ended loop, no checkpoint change, no existing observed_state or
  prediction NPZ modification, and no RL/GDPO/PPO/BC/IL.
- Reverified aq/as:
  both were `start_corridor` at `[0.0, -4.45, 1.2]`, yaw
  `1.5707963267948966`, tree_seed `0` vs `1`, exactly two frames, exactly two
  map_predict calls, exactly one action, no second action, no third frame, and
  no rollout, with formula `gain_exp / cost + 48 * minmax(source_occ_free)`.
- aq/as comparison:
  aq Frame1/Frame2 lambda48 was `same_as_measured`; as Frame1/Frame2 lambda48
  was `distinct_nonmeasured_branch`. Frame1 selected/best deltas were `0.2m`
  / `1.6881943016134136m`; Frame2 selected/best deltas were
  `0.458257569495584m` / `2.4103941586387903m`. Action pose/yaw deltas were
  `0.20000000000000018m` / `2.7504672066207645rad`.
- observed_state/map_predict:
  aq observed_ratio delta `0.012087962962962964`, newly observed `5222`; as
  observed_ratio delta `0.006354166666666667`, newly observed `2745`;
  as-aq observed_ratio delta `-0.005733796296296297`. map_predict remained
  stable: Frame1 exact match `61152 / 49164`, Frame2 aq `52988 / 43828` vs as
  `47866 / 41937`, both `code_consistent_v1`, no explosion/collapse.
- lambda32/lambda48:
  Frame1 matched selected/best for both seeds. Frame2 aq matched selected
  child only; as lambda48 diverged from lambda32/measured in a healthy
  diagnostic way.
- Outcome:
  `healthy_distinct_seed1_after_conservative_seed0`, clean and seed-sensitive
  but not a safety regression. No low-cost artifact, no historical prior
  basin, no prediction writeback/fusion, no prediction
  traversability/collision/ray blocking, no candidate sampling/edge-validity
  use, no target/ground-truth/future-observed scoring, no over-cost runtime
  promotion, and no coverage-improvement claim.
- start_corridor tree_seed `2` was not executed and is not automatically next.
  Current evidence is still not rollout-ready.
- Next small task:
  Stage 4A-6.5au `start_room_b` tree_seed `0` bounded repeat-safety smoke
  execution only, not rollout. Use pose `[2.75, -2.55, 1.2]`, yaw
  `2.7052603405912112`, from
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_b_empty_astar/scene_metadata.json`.
  Keep exactly two frames, exactly two map_predict calls if action executes,
  exactly one selected action, no second action, no third frame, no rollout,
  formula `gain_exp / cost + 48 * minmax(source_occ_free)`, and
  `--max_workers 32`.
- Long-term GDPO remains future direction only; RL/GDPO/PPO/BC/IL remains
  explicitly not next until bounded repeats and rollout data are ready.

- Stage 4A-6.5as start_corridor tree_seed `1` bounded repeat-safety runtime
  smoke is complete and validated.
- Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65as_start_corridor_tree_seed1_bounded_smoke`
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65as_start_corridor_seed1_bounded_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65as_start_corridor_seed1_bounded_smoke.py`.
- Logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65as_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65as_start_corridor_tree_seed1_bounded_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65as_start_corridor_tree_seed1_bounded_smoke_test.log`.
- Runtime setup:
  `medium_three_rooms`, scene seed `0`, start variant `start_corridor`, pose
  `[0.0, -4.45, 1.2]`, yaw `1.5707963267948966`, pose source
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_corridor_empty_astar/scene_metadata.json`,
  repeat variant `alternate_start_corridor_tree_seed1`, reference Stage
  4A-6.5aq tree_seed `0`, current tree_seed `1`.
- Sequence:
  exactly one Isaac startup in the clean run, exactly two frames, exactly two
  map_predict calls, exactly one selected action execution, no second action,
  no third frame, and no rollout.
- Frame 1:
  measured-only shadow `n0018 -> n0022`, lambda48 primary `n0001 -> n0135`,
  lambda32 shadow `n0001 -> n0135`, classification
  `distinct_nonmeasured_branch`, low-cost artifact `false`, historical prior
  basin `false`; the single action executed to
  `[0.15000000000000036, -3.9499999999999997, 1.2]`, yaw
  `-0.29145679447786677`.
- Frame 2:
  measured-only shadow `n0036 -> n0106`, lambda48 diagnostic
  `n0008 -> n0137`, lambda32 shadow `n0036 -> n0106`, classification
  `distinct_nonmeasured_branch`, low-cost artifact `false`, historical prior
  basin `false`.
- Repeat comparison vs Stage 4A-6.5aq:
  Frame1 selected/best deltas `0.2m` / `1.6881943016134136m`, Frame2
  selected/best deltas `0.458257569495584m` / `2.4103941586387903m`,
  action pose/yaw deltas `0.20000000000000018m` /
  `2.7504672066207645rad`; observed_ratio delta `0.006354166666666667`
  was `-0.005733796296296297` vs aq; map_predict remained stable with
  density ratio `0.8530021967293141`.
- Outcome:
  `spatially_consistent_healthy_repeat`, clean but not rollout-ready and not
  coverage-improvement evidence.
- Prediction stayed read-only/information-gain-only with no writeback/fusion,
  no traversability/collision/ray blocking, no candidate sampling or
  edge-validity use, no target/ground-truth/future-observed scoring, no
  over-cost runtime primary, no checkpoint changes, and no external source
  build.
- Validation:
  test log reported `all_passed: true`.
- Next small task:
  Stage 4A-6.5at start_corridor seed0/seed1 repeat-comparison diagnosis and
  next-start design only, not rollout. Still no rollout, no open-ended loop,
  no RL/GDPO/PPO/BC/IL, no prediction writeback/fusion, and no over-cost
  runtime promotion.
- Long-term GDPO remains future direction only; RL/GDPO/PPO/BC/IL is not next
  until bounded repeats and rollout data are ready.

- Stage 4A-6.5ar alternate-start post-action/two-frame diagnosis and
  repeat-safety review is complete and validated. It was diagnosis/design
  only: no Isaac startup, no RGB/depth capture, no map_predict call, no
  SSCNet inference, no action execution, no rollout, no open-ended loop, no
  checkpoint change, no existing observed_state or prediction NPZ
  modification, and no RL/GDPO/PPO/BC/IL.
- Stage 4A-6.5ar output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ar_alternate_start_post_action_diagnosis`
- Stage 4A-6.5ar created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_stage4a65ar_alternate_start_post_action.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ar_alternate_start_post_action.py`.
- Stage 4A-6.5ar logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ar_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ar_alternate_start_post_action_diagnosis.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ar_alternate_start_post_action_diagnosis_test.log`.
- Reverified Stage 4A-6.5aq sequence:
  exactly two frames, exactly two map_predict calls, exactly one selected
  action, no second action, no third frame, and no rollout.
- start_corridor pose/yaw matched Stage 4A-6.5ap design and metadata:
  `[0.0, -4.45, 1.2]`, yaw `1.5707963267948966`.
- Action pose consistency passed:
  `[-0.04999999999999982, -3.9499999999999997, 1.2]`, yaw
  `-3.0419240010986313`, matched Frame2 pose.
- Observed/map_predict diagnosis:
  observed_ratio `0.03149537037037037 -> 0.043583333333333335`, newly
  observed `5222`, unknown->free `4876`, unknown->occupied `346`, invalid
  labels `0`; Frame1 valid/OCC+FREE `61152 / 49164`, Frame2
  `52988 / 43828`, density ratio `0.8914652998128713`, no
  explosion/collapse, both `code_consistent_v1`.
- Tree/branch diagnosis:
  Frame1 lambda48 matched measured-only (`n0001 -> n0104`); Frame2 lambda48
  remained `same_as_measured`, shared selected child `n0001`, and had best
  descendant `n0127` versus measured `n0126`. lambda32/lambda48 matched on
  Frame1 selected/best and matched Frame2 selected child only.
- No low-cost artifact, no historical prior basin, no prediction
  writeback/fusion, no prediction traversability/collision/ray blocking, no
  prediction candidate sampling/edge-validity use, no target/ground-truth/
  future-observed scoring, no over-cost runtime primary, and no coverage
  improvement claim.
- Stage 4A-6.5ar outcome:
  `clean_same_as_measured`, conservative but safe. Rollout readiness is
  `false`.
- Next small task if proceeding:
  Stage 4A-6.5as start_corridor tree_seed `1` bounded repeat-safety smoke
  execution only, not rollout. It should keep exactly two frames, exactly two
  map_predict calls if action executes, exactly one selected action, no second
  action, no third frame, no rollout, formula
  `gain_exp / cost + 48 * minmax(source_occ_free)`, and `--max_workers 32`.
- Future Stage 4A-6.5as command sketch is present and begins
  `DO NOT RUN IN STAGE 4A-6.5ar.` It was not executed.
- Long-term GDPO remains future direction only; no RL/GDPO/PPO/BC/IL is next
  until bounded repeats and rollout data are ready.

- Stage 4A-6.5aq alternate-start bounded two-frame/one-action lambda48 smoke
  at `start_corridor`, `tree_seed=0`, is complete and validated.
- Stage 4A-6.5aq output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65aq_alternate_start_corridor_bounded_smoke`
- Stage 4A-6.5aq created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65aq_alternate_start_bounded_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65aq_alternate_start_bounded_smoke.py`.
- Stage 4A-6.5aq logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aq_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aq_alternate_start_corridor_bounded_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aq_alternate_start_corridor_bounded_smoke_test.log`.
- Runtime:
  `medium_three_rooms`, scene seed `0`, start variant `start_corridor`, pose
  `[0.0, -4.45, 1.2]`, yaw `1.5707963267948966`, pose source
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_corridor_empty_astar/scene_metadata.json`,
  distance `4.654299087940095m` from canonical start, `tree_seed=0`.
- Sequence:
  exactly one Isaac startup in the clean run, exactly two frames, exactly two
  map_predict calls, exactly one selected action execution, no second action,
  no third frame, and no rollout.
- Frame 1:
  measured-only shadow `n0001 -> n0104`, lambda48 primary
  `n0001 -> n0104`, lambda32 shadow `n0001 -> n0104`,
  `same_as_measured`, low-cost artifact `false`, historical prior basin
  `false`; action executed to
  `[-0.04999999999999982, -3.9499999999999997, 1.2]`, yaw
  `-3.0419240010986313`.
- Frame 2:
  measured-only shadow `n0001 -> n0126`, lambda48 diagnostic
  `n0001 -> n0127`, lambda32 shadow `n0001 -> n0126`,
  `same_as_measured`, low-cost artifact `false`, historical prior basin
  `false`.
- Observed/map_predict:
  observed_ratio `0.03149537037037037 -> 0.043583333333333335`, newly
  observed `5222`; Frame 1 valid/OCC+FREE `61152 / 49164`, Frame 2
  `52988 / 43828`, density ratio `0.8914652998128713`, no
  explosion/collapse, both `code_consistent_v1`.
- Stage 4A-6.5aq outcome:
  `clean_same_as_measured`. It matched Stage 4A-6.5ap design. Canonical-start
  seed0/1/2 comparison is context only because the start pose changed.
- Prediction stayed read-only/information-gain-only, with no writeback/fusion,
  no traversability/collision/ray blocking, no candidate sampling or
  edge-validity use, no target/ground-truth/future-observed scoring, no
  checkpoint changes, no external source build, no over-cost runtime primary,
  and no coverage-improvement claim.
- Hardware:
  `os_cpu_count=32`, requested/actual workers `32/32`,
  OMP/OPENBLAS/MKL/NUMEXPR/VECLIB threads `1/1/1/1/1`, GPU
  `NVIDIA GeForce RTX 5080`, total runtime wall time
  `39.62141864500154s`.
- Validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aq_alternate_start_corridor_bounded_smoke_test.log`
  reported `all_passed: true`.
- Next small task:
  Stage 4A-6.5ar alternate-start post-action/two-frame diagnosis and
  repeat-safety review only, not rollout.
- Still no rollout, online open-ended loop, RL/GDPO/PPO/BC/IL, policy
  checkpoint, replay buffer, prediction writeback/fusion,
  prediction traversability/collision/ray blocking, target/ground-truth/
  future-observed scoring, over-cost runtime promotion, Pareto runtime planner,
  or coverage-improvement claim.
- Long-term GDPO remains future direction only; RL/GDPO/PPO/BC/IL is not next
  until bounded repeats and rollout data are ready.

- Stage 4A-6.5ap seed0/1/2 repeat-comparison review and alternate-start
  bounded-repeat design is complete and validated.
- Stage 4A-6.5ap output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ap_seed012_repeat_review_alternate_start_design`
- Stage 4A-6.5ap created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/review_stage4a65ap_seed012_alternate_start_design.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ap_seed012_alternate_start_design.py`.
- Stage 4A-6.5ap logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ap_seed012_repeat_review_alternate_start_design.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ap_seed012_repeat_review_alternate_start_design_test.log`.
- Stage 4A-6.5ap was offline review/design only:
  no Isaac startup, no RGB/depth capture, no map_predict, no SSCNet
  inference, no selected action execution, no two-frame runtime execution, no
  rollout, no online/open-ended loop, no training/RL/GDPO/PPO/BC/IL, no
  checkpoint changes, and no existing observed_state or prediction NPZ
  modification.
- Stage 4A-6.5ap reverified seed0/1/2 bounded smokes:
  Stage 4A-6.5ak tree_seed `0`, Stage 4A-6.5am tree_seed `1`, and Stage
  4A-6.5ao tree_seed `2` each had exactly two frames, exactly two
  map_predict calls, exactly one selected action, no second action, no third
  frame, and no rollout.
- Frame 1 lambda48:
  seed0 `n0001 -> n0228` (`distinct_nonmeasured_branch`), seed1
  `n0001 -> n0157` (`same_as_measured`), seed2 `n0001 -> n0248`
  (`same_as_measured`).
- Frame 2 lambda48:
  seed0 `n0002 -> n0158`, seed1 `n0001 -> n0214`, seed2
  `n0003 -> n0227`; all are `distinct_nonmeasured_branch`.
- Action/observed/map_predict comparison:
  action deltas were seed0-vs-seed1 `0.20000000000000018m`,
  seed0-vs-seed2 `0.22360679774997896m`, and seed1-vs-seed2
  `0.4123105625617663m`; observed deltas were seed0
  `0.026840277777777775` / `11595`, seed1 `0.015152777777777777` / `6546`,
  and seed2 `0.013023148148148148` / `5626`; Frame 1 map_predict matched
  `57382 / 40328`, while Frame 2 stayed stable at seed0 `47814 / 30133`,
  seed1 `37258 / 27254`, and seed2 `32890 / 24936`.
- lambda32/lambda48 agreement:
  seed0 matched both frames, seed1 matched Frame 1 but diverged Frame 2, and
  seed2 matched both frames.
- No seed/frame had low-cost artifact or historical prior basin. Prediction
  stayed read-only/information-gain-only with no writeback/fusion,
  traversability/collision/ray blocking, candidate sampling, edge-validity
  use, target/ground-truth/future-observed scoring, over-cost runtime
  promotion, or coverage-improvement claim.
- Combined outcome:
  `seed_sensitive_but_clean`. Seed2 is spatially consistent with seed1 and
  closer than seed1 to seed0 on Frame 2 selected child, but current evidence
  is still not rollout-ready.
- Selected future alternate start:
  `start_corridor`, pose `[0.0, -4.45, 1.2]`, yaw
  `1.5707963267948966`, from
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_corridor_empty_astar/scene_metadata.json`.
  Future Stage 4A-6.5aq should use tree_seed `0` first.
- Future Stage 4A-6.5aq command sketch is present and begins
  `DO NOT RUN IN STAGE 4A-6.5ap.` It was not executed.
- Long-term GDPO is future direction only; no RL/GDPO/PPO/BC/IL in 6.5ap.
- Next small task:
  Stage 4A-6.5aq alternate-start bounded two-frame/one-action lambda48 repeat
  smoke at `start_corridor`, tree_seed `0`, still no rollout.

- Stage 4A-6.5ao bounded repeat-safety smoke is complete and validated.
- Stage 4A-6.5ao output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ao_bounded_repeat_safety_smoke_tree_seed2`
- Stage 4A-6.5ao created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65ao_bounded_repeat_safety_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ao_bounded_repeat_safety_smoke.py`.
- Stage 4A-6.5ao logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ao_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ao_bounded_repeat_safety_smoke_tree_seed2.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ao_bounded_repeat_safety_smoke_tree_seed2_test.log`
- Stage 4A-6.5ao sequence:
  exactly one Isaac startup in the clean run, exactly two frames, exactly two
  map_predict calls, exactly one selected action execution, no second action,
  no third frame, and no rollout. A first launch attempt failed at GLX
  initialization before capture; the clean run used the validated headless
  NVIDIA/Vulkan env.
- Stage 4A-6.5ao used the same `medium_three_rooms` scene seed `0` and the
  same start pose as Stage 4A-6.5ak/6.5am, changing only mini-RRT
  `tree_seed` from references `0` and `1` to current `2`.
- Frame 1:
  measured-only shadow `n0001 -> n0248`, lambda48 primary `n0001 -> n0248`,
  lambda32 shadow `n0001 -> n0248`; lambda48 classification
  `same_as_measured`, low-cost artifact `false`, historical prior basin
  `false`; pre-action safety gates passed.
- The single action executed to `[-4.25, -4.35, 1.2]`, yaw
  `2.2142974355881817`.
- Frame 2:
  measured-only shadow `n0126 -> n0186`, lambda48 diagnostic
  `n0003 -> n0227`, lambda32 shadow `n0003 -> n0227`; lambda48
  classification `distinct_nonmeasured_branch`, low-cost artifact `false`,
  historical prior basin `false`.
- Observed_state delta was sane:
  observed_ratio `0.0425462962962963 -> 0.05556944444444444`, delta
  `0.013023148148148148`, newly observed `5626`, unknown->free `5078`,
  unknown->occupied `548`, occupied->free `0`, invalid labels `0`.
- map_predict remained stable:
  Frame 1 valid/OCC+FREE `57382 / 40328`, Frame 2 valid/OCC+FREE
  `32890 / 24936`, density ratio `0.6183296964887919`, no
  explosion/collapse, both `code_consistent_v1`.
- Repeat comparison against seed0/seed1:
  Frame 1 selected deltas `0.223606797749979m` / `0.41231056256176607m`;
  Frame 2 selected deltas `0.5m` / `0.632455532033676m`; Frame 2
  best-descendant deltas `4.036087214122113m` / `1.2083045973594573m`.
- Repeat outcome:
  `spatially_consistent_healthy_repeat`, spatially consistent with seed1 and
  seed-sensitive but clean. Still not rollout-ready.
- Next small task:
  Stage 4A-6.5ap repeat-comparison review / alternate-start design only, not
  rollout.
- Still no rollout, online open-ended loop, RL/PPO/BC/IL/GDPO, prediction
  writeback/fusion, prediction traversability/collision/ray blocking,
  target/ground-truth/future-observed scoring, checkpoint changes, external
  source build, runtime planner implementation, over-cost runtime promotion,
  or coverage-improvement claim.

- Stage 4A-6.5an repeat-comparison review and next bounded-repeat design is
  complete and validated. This was review/design only: no Isaac startup, no
  capture, no map_predict, no SSCNet inference, no action execution, no
  two-frame runtime execution, and no rollout.
- Stage 4A-6.5an output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65an_repeat_comparison_and_next_design`
- Stage 4A-6.5an created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/review_stage4a65an_repeat_comparison.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65an_repeat_comparison.py`.
- Stage 4A-6.5an logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65an_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65an_repeat_comparison_and_next_design.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65an_repeat_comparison_and_next_design_test.log`
- Stage 4A-6.5an reverified both bounded repeats:
  Stage 4A-6.5ak tree_seed `0` and Stage 4A-6.5am tree_seed `1` each had
  exactly two frames, exactly two map_predict calls, exactly one selected
  action, no second action, no third frame, and no rollout.
- Stage 4A-6.5an comparison:
  Frame 1 changed from tree_seed `0` lambda48 `n0001 -> n0228`
  (`distinct_nonmeasured_branch`) to tree_seed `1` lambda48 `n0001 -> n0157`
  (`same_as_measured`), selected delta `0.2m`; Frame 2 changed from
  `n0002 -> n0158` to `n0001 -> n0214`, selected delta
  `1.0816653826391969m`, with both Frame 2 branches still
  `distinct_nonmeasured_branch`.
- Stage 4A-6.5an observed/action comparison:
  action pose delta `0.20000000000000018m`; observed_state delta was
  `0.026840277777777775` / `11595` newly observed for tree_seed `0` and
  `0.015152777777777777` / `6546` newly observed for tree_seed `1`; this is
  plausible from the different one-action pose and not suspicious.
- Stage 4A-6.5an map_predict comparison:
  Frame 1 valid/OCC+FREE matched `57382 / 40328`; Frame 2 was
  `47814 / 30133` for tree_seed `0` and `37258 / 27254` for tree_seed `1`,
  with no density explosion/collapse and both `code_consistent_v1`.
- Stage 4A-6.5an branch health:
  no low-cost artifact and no historical prior basin in either run/frame;
  prediction stayed read-only and information-gain-only with no writeback,
  traversability/collision/ray blocking/candidate-sampling/edge-validity use.
- Repeat outcome:
  `divergent_but_healthy`; this indicates tree_seed sensitivity but not a
  safety regression. Still not rollout-ready.
- Next small task:
  Stage 4A-6.5ao bounded repeat-safety smoke, same scene/start,
  `tree_seed=2`, with exactly two frames if gates pass, exactly two
  map_predict calls if action executes, exactly one selected action, no second
  action, no third frame, no rollout, formula
  `gain_exp / cost + 48 * minmax(source_occ_free)`, and `--max_workers 32`.
- Still no rollout, online open-ended loop, RL/PPO/BC/IL, prediction
  writeback/fusion, prediction traversability/collision/ray blocking,
  target/ground-truth/future-observed scoring, checkpoint changes, external
  source build, runtime planner implementation, over-cost runtime promotion,
  or coverage-improvement claim.
- Stage 4A-6.5am bounded repeat-safety smoke is complete and validated.
- Stage 4A-6.5am used the same `medium_three_rooms` scene seed `0` and the
  same start pose as Stage 4A-6.5ak, changing only mini-RRT `tree_seed` from
  reference `0` to current `1`.
- Stage 4A-6.5am output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65am_bounded_repeat_safety_smoke_tree_seed1`
- Stage 4A-6.5am created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65am_bounded_repeat_safety_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65am_bounded_repeat_safety_smoke.py`.
- Stage 4A-6.5am logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65am_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65am_bounded_repeat_safety_smoke_tree_seed1.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65am_bounded_repeat_safety_smoke_tree_seed1_test.log`
- Stage 4A-6.5am sequence:
  exactly one Isaac startup, exactly two frames, exactly two map_predict calls,
  exactly one selected action execution, no second action, no third frame, and
  no rollout.
- Frame 1:
  measured-only shadow `n0001 -> n0157`, lambda48 primary `n0001 -> n0157`,
  lambda32 shadow `n0001 -> n0157`; lambda48 classification
  `same_as_measured`, low-cost artifact `false`, historical prior basin
  `false`; pre-action safety gates passed.
- The single action executed to `[-4.15, -4.75, 1.2]`, yaw
  `2.1587989303424653`.
- Frame 2:
  measured-only shadow `n0003 -> n0255`, lambda48 diagnostic
  `n0001 -> n0214`, lambda32 shadow `n0003 -> n0179`; lambda48
  classification `distinct_nonmeasured_branch`, low-cost artifact `false`,
  historical prior basin `false`.
- Observed_state delta was sane: observed_ratio
  `0.0425462962962963 -> 0.057699074074074076`, delta
  `0.015152777777777777`, newly observed `6546`, unknown->free `5893`,
  unknown->occupied `653`, occupied->free `0`, invalid labels `0`.
- map_predict remained stable: Frame 1 valid/OCC+FREE `57382 / 40328`,
  Frame 2 valid/OCC+FREE `37258 / 27254`, density ratio
  `0.6758083713548899`, no explosion/collapse, both `code_consistent_v1`.
- Repeat comparison against Stage 4A-6.5ak:
  Frame 1 selected delta `0.2m`, Frame 2 selected delta
  `1.0816653826391969m`, action pose delta `0.20000000000000018m`;
  repeat outcome `divergent_but_healthy`.
- Validation passed:
  required outputs, repeat variant/tree_seed contract, formula,
  hardware report, prediction safety, hash checks, no-rollout checks, repeat
  outputs, and forbidden-output absence.
- Hash/safety audit passed:
  checkpoint unchanged; referenced Stage 4A-6.5ak/6.5al observed_state and
  prediction NPZ files unchanged; generated prediction NPZs unchanged after
  creation; prediction remained read-only and information-gain-only, with no
  writeback/fusion and no traversability/collision/ray blocking/candidate
  sampling/edge-validity use.
- Next small task:
  another bounded repeat review, likely alternate start or `tree_seed=2`
  design, still exactly two frames/one action, not rollout.
- Still no rollout, online open-ended loop, RL/PPO/BC/IL, prediction
  writeback/fusion, prediction traversability/collision/ray blocking,
  target/ground-truth/future-observed scoring, checkpoint changes, external
  source build, runtime planner implementation, over-cost runtime promotion,
  or coverage-improvement claim.

Done:

- Verify full SSCNet training result and best checkpoint.
- Build `offline_infer_npz.py` for best-checkpoint offline inference.
- Build read-only `PredictionLayer`.
- Run single-sample offline inference smoke test.
- Run PredictionLayer smoke test.
- Run first-5-sample offline inference check.
- Stage 2B strict paper-faithful deterministic expert scorer.
- Stage 2B single-sample paper expert run.
- Stage 2B paper expert smoke test.
- Stage 2B batch5 paper expert smoke test.
- Stage 2C paper expert dataset generator.
- Stage 2C paper expert dataset validator.
- Stage 2C batch5 dataset smoke run.
- Stage 2C dataset validation test.
- Stage 3A IL Dataset/DataLoader module.
- Stage 3A feature stats helper.
- Stage 3A CandidateMLPPolicy skeleton.
- Stage 3A behavior cloning script skeleton with dry-run only.
- Stage 3A data-loading and forward-only BC smoke tests.
- Stage 4A-1 Isaac / IsaacLab environment discovery.
- Stage 4A-1 official headless empty scene smoke test.
- Stage 4A-1 official USD camera/depth sensor smoke test.
- Stage 4A-1 minimal Isaac depth scene with 3 fixed poses.
- Stage 4A-1 measured-only depth-to-observed-voxel-map conversion.
- Stage 4A-1 pure Python depth_to_voxel smoke tests.
- Stage 4A-6.5ak staged two-frame one-action lambda48 runtime smoke execution.
- Stage 4A-6.5al post-action/two-frame diagnosis and repeat-safety review.
- Stage 4A-6.5am bounded repeat-safety smoke with same scene/start and
  `tree_seed=1`.
- Stage 4A-6.5ao bounded repeat-safety smoke with same scene/start and
  `tree_seed=2`.
- Stage 4A-1-viz depth PNG visualizations.
- Stage 4A-1-viz observed topdown comparison.
- Stage 4A-1-viz occupied/free 3D scatter visualizations.
- Stage 4A-1-viz z-slice visualizations.
- Stage 4A-1-viz summary JSON and HTML index.
- Stage 4A-1-scene-viz scripted Isaac scene RGB render views.
- Stage 4A-1-scene-viz per-pose depth color visualizations.
- Stage 4A-1-scene-viz overview RGB/depth render.
- Stage 4A-1-scene-viz scene layout topdown image and summary JSON.
- Stage 4A-2 simulator observed-map expert scorer core.
- Stage 4A-2 EmptyPredictionLayer integration.
- Stage 4A-2 frontier and frontier-adjacent FREE voxel detection.
- Stage 4A-2 candidate sampling from observed FREE voxels.
- Stage 4A-2 observed-map raycast visibility with conservative UNKNOWN blocking.
- Stage 4A-2 paper-style gain/cost scoring and top-N expert action output.
- Stage 4A-2 expert decision NPZ/JSON/JSONL output.
- Stage 4A-2 topdown and score-bar visualizations.
- Stage 4A-2 smoke test with observed_state hash/no-modification check.
- Stage 4A-3 rollout utilities and transition serialization.
- Stage 4A-3 measured-only single-frame depth update wrapper.
- Stage 4A-3 Isaac headless multi-step expert rollout runner.
- Stage 4A-3 planar teleport camera motion mode.
- Stage 4A-3 per-step transition `.npz`, episode `transitions.jsonl`,
  final observed map, episode summary, and global manifest output.
- Stage 4A-3 rollout visualizations: topdown path, observed ratio curve,
  frontier count curve, per-step topdown maps, and HTML index.
- Stage 4A-3 rollout smoke test with synthetic transition serialization and
  real episode validation.
- Stage 4A-3 EmptyPredictionLayer rollout run:
  10 steps, observed_ratio `0.0 -> 0.21754166666666666`,
  done_reason `max_steps`.
- Stage 4A-3.2 scene factory for scripted minimal and medium-complexity Isaac
  scenes.
- Stage 4A-3.2 medium-complexity Isaac scene with 3 rooms, 1 corridor,
  3 openings, 13 walls, 13 cuboid obstacles, and 5 fixed camera poses.
- Stage 4A-3.2 fixed-pose RGB/depth capture:
  `depth_000.npy` through `depth_004.npy`, `rgb_000.png` through
  `rgb_004.png`, pose JSON files, `camera_info.json`, and
  `scene_metadata.json`.
- Stage 4A-3.2 depth_to_voxel bounds CLI:
  `--x_min --x_max --y_min --y_max --z_min --z_max`.
- Stage 4A-3.2 measured-only observed map for bounds `x/y=[-6,6]`, `z=[0,3]`:
  shape `(120, 120, 30)`, unknown/free/occupied `339813 / 86064 / 6123`,
  observed_ratio `0.21339583333333334`.
- Stage 4A-3.2 visualizations:
  `scene_overview_rgb.png`, `scene_overview_depth_color.png`,
  `scene_layout_topdown.png`, `camera_rgb_grid.png`, `camera_depth_grid.png`,
  `observed_topdown_compare.png`, `free_occupied_voxels_3d_final.png`, and
  `slices_final.png`.
- Stage 4A-3.2 pure Python metadata test and validation checks.
- Stage 4A-3.2 optional one-step EmptyPredictionLayer expert smoke on the
  medium observed map.
- Stage 4A-3.5 observed-free A* planner:
  `astar_planner.py`, `build_traversability_grid`, `astar_2d`,
  `path_length_m`, and traversability summaries.
- Stage 4A-3.5 A* path-cost integration into one-step simulator expert and
  multi-step rollout expert via `--path_cost_mode astar`, preserving default
  `--path_cost_mode euclidean`.
- Stage 4A-3.5 candidate reachability bookkeeping, best A* path saving,
  traversability visualization, rollout A* segment overlays, and
  reachable-candidates curve.
- Stage 4A-3.5 one-step medium A* expert smoke:
  reachable/unreachable candidates `12 / 52`, best score
  `51.651363679237036`, best path_cost `2.129663036258191`, `gain_sc=0.0`.
- Stage 4A-3.5 medium A* rollout smoke:
  5 completed transitions, observed_ratio `0.0 -> 0.04308796296296296`,
  done_reason `no_valid_candidate`, `gain_sc=0.0`, no Euclidean fallback.
- Stage 4A-3.5 tests:
  A* planner test, simulator A* validator, Euclidean one-step regression, and
  Euclidean rollout regression.
- Stage 4A-3.6 reachability-aware A* candidate generation:
  connected component from current pose, nearest traversable snap, reachable
  frontier candidate mask, `--candidate_sampling_mode`, and reachable
  diagnostics.
- Stage 4A-3.6 one-step medium reachable A* expert smoke:
  reachable/unreachable candidates `64 / 0`, reachable component count
  `1196`, reachable frontier-adjacent count `1196`, `top_n=16`, best score
  `88.24634362636618`, best path_cost `0.7479063413600806`, `gain_sc=0.0`.
- Stage 4A-3.6 medium reachable A* rollout smoke:
  10 completed transitions, observed_ratio `0.0 -> 0.10147453703703704`,
  done_reason `max_steps`, average reachable candidates `64.0`,
  `no_valid_candidate_steps=[]`, `gain_sc=0.0`, no Euclidean fallback.
- Stage 4A-3.6 tests:
  reachable candidate sampling test, A* planner regression, and simulator A*
  reachable-output validator, plus Euclidean one-step and rollout regressions.
- Stage 4A-4 multi-episode EmptyPredictionLayer A* rollout dataset runner.
- Stage 4A-4 dataset summarizer with aggregate plots and HTML index.
- Stage 4A-4 batch dataset validator.
- Stage 4A-4 rollout runner start-variant metadata and explicit selected
  expert fields (`gain_exp`, `gain_sc`, `gain_hybrid`, `path_cost`,
  `final_score`).
- Stage 4A-4 9-episode medium dataset:
  seeds `0,1,2`, starts `start_room_a,start_corridor,start_room_b`,
  ok episodes `9`, failed episodes `0`, total transitions `90`.
- Stage 4A-4 dataset result:
  steps min/mean/max `10 / 10 / 10`, done_reason `max_steps=9`,
  observed_ratio_end min/mean/max
  `0.08587037037037037 / 0.11455118312757204 / 0.16534722222222223`,
  average reachable candidates `64.0`, average gain_sc `0.0`.
- Stage 4A-4 tests:
  py_compile, reachable candidate sampling regression, A* planner regression,
  rollout regression, dataset summary, and batch dataset validation.
- Stage 4A-5 single-frame Isaac map_predict preprocessing and shape-alignment
  smoke test.
- Stage 4A-5 SSCNet position convention check against `projection_layer.py`,
  `dataloaders/dataloader.py`, and real NYU `position` samples.
- Stage 4A-5 Isaac depth preprocessing:
  `depth_000.npy` -> `(480,640)` float32 depth input, `(480,640)` int64
  position map, and valid position mask.
- Stage 4A-5 strict checkpoint inference with
  `/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar`.
- Stage 4A-5 local prediction output:
  logits `(1,12,60,36,60)`, pred_class/confidence/free_prob/occupied_prob
  `(60,36,60)`.
- Stage 4A-5 global read-only prediction layer aligned to observed_state shape
  `(120,120,30)`.
- Stage 4A-5 visualization outputs:
  depth input, local slices, global topdown, observed-vs-prediction, and
  prediction-not-measured topdown.
- Stage 4A-5 smoke test:
  observed_state hash unchanged, SimPredictionLayer API ok, no target or
  ground-truth fields, no training, no expert/rollout prediction use, no
  prediction writeback, and no traversability/collision/A* use.
- Stage 4A-5.1 one-step SC-aware expert scoring with read-only
  SimPredictionLayer.
- Stage 4A-5.1 `run_sim_expert_step.py` CLI support:
  `--prediction_mode empty|sim_npz`, `--prediction_npz`, `--tau`, and
  `--episode_summary`.
- Stage 4A-5.1 paper-style prediction gains in simulator expert:
  `gain_sc`, `gain_hybrid`, `gain_occ`, and `gain_conf`.
- Stage 4A-5.1 prediction visualizations:
  `prediction_overlay_topdown.png` and
  `predicted_unmeasured_visible_topdown.png`.
- Stage 4A-5.1 comparison outputs:
  `comparison_summary.json`, `comparison_summary.md`,
  `empty_vs_prediction_best_candidate.png`, and `gain_comparison_bar.png`.
- Stage 4A-5.1 smoke test:
  empty baseline `gain_sc=0`, prediction mode `64/64` candidates with
  `gain_sc>0`, best `gain_hybrid=110.0`, observed_state hash unchanged, and
  no prediction use for traversability/collision/A*/ray blocking/writeback.
- Stage 4A-6 short multi-step SC-aware rollout with dynamic per-step
  map_predict.
- Stage 4A-6 `IsaacMapPredictor` loads SSCNet checkpoint once and reuses it
  for all rollout steps.
- Stage 4A-6 rollout command:
  medium_three_rooms seed `0`, start `start_room_a`, max_steps `5`,
  prediction_mode `sim_dynamic`, path_cost_mode `astar`,
  candidate_sampling_mode `reachable_frontier`, motion_mode `planar`.
- Stage 4A-6 result:
  steps_completed `5`, done_reason `max_steps`, observed_ratio
  `0.0 -> 0.05899768518518519`, final unknown/free/occupied
  `406513 / 21226 / 4261`.
- Stage 4A-6 SC gain:
  average gain_sc `49.4`, average gain_hybrid `99.0`,
  candidates_with_gain_sc_positive min/mean/max `63 / 63.6 / 64`.
- Stage 4A-6 comparison to matching Stage 4A-4 empty baseline:
  compared_steps `5`, changed selected actions `5`, empty final observed_ratio
  `0.06896296296296296`, SC final observed_ratio `0.05899768518518519`.
- Stage 4A-6 tests:
  py_compile passed and `test_sim_sc_aware_rollout.py` passed; prediction
  stayed read-only and information-gain-only, checkpoint not modified, and no
  RL/optimizer/BC/IL/SSCNet training ran.
- Stage 4A-6.1 SC-aware rollout underperformance analysis / ablation / tuning.
- Stage 4A-6.1 analysis outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/existing_sc_vs_empty`
- Stage 4A-6.1 ablation outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_ablation`
- Stage 4A-6.1 qualitative inspection:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/qualitative_inspection`
- Stage 4A-6.1 result:
  existing SC final observed_ratio `0.05899768518518519`, empty baseline
  final observed_ratio at 5 steps `0.06896296296296296`, delta
  `-0.009965277777777774`, first SC lag step `1`, changed actions `5/5`,
  mean path_cost empty/SC `0.36998367643136965 / 0.2768163156997422`.
- Stage 4A-6.1 ablations completed:
  `dynamic_w025_tau01`, `dynamic_w05_tau01`, `dynamic_w1_tau03`,
  `dynamic_w1_tau01_cap50`, `static_step0_weight_1p0_tau_0p1`; all completed
  5 steps, all ended at
  observed_ratio `0.05899768518518519`, and all selected the same 5/5 actions
  as the original SC rollout.
- Stage 4A-6.1 tests:
  py_compile passed and `test_sc_pred_ablation.py` passed; weighted gain
  formula verified, observed_ratio non-decreasing, checkpoint not modified,
  prediction stayed read-only and information-gain-only, and no
  RL/optimizer/BC/IL/SSCNet training ran.
- Stage 4A-6.2 map_predict preprocessing / alignment / calibration diagnostics.
- Stage 4A-6.2 preprocessing diagnostics:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/preprocess_stats`
- Stage 4A-6.2 global alignment diagnostics:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/global_alignment`
- Stage 4A-6.2 future observed evaluation:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/future_observed_eval`
  using future measured maps only as post-hoc delayed sensor validation.
- Stage 4A-6.2 alignment variant sweep:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/alignment_variant_sweep`
- Stage 4A-6.2 candidate-score decomposition:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/candidate_score_decomposition`
- Stage 4A-6.2 summary:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/summary`
- Stage 4A-6.2 result:
  primary suspected issue is alignment convention. The diagnostic
  `xz_swap_variant` ranked above `current_default` with Brier improvement
  `0.0735458940774611`; default rank was `7`. Secondary issue is dense
  confidence calibration at tau `0.1`, with mean predicted_unmeasured
  `35118.2`, later measured fraction `0.059004217437215026`, occupied Brier
  `0.2786559495144023`, and ECE-like calibration `0.3405436085907938`.
- Stage 4A-6.2 tests:
  py_compile passed and `test_map_predict_diagnostics.py` passed; observed
  state hashes unchanged, checkpoint not modified, future observations marked
  evaluation-only, prediction stayed read-only and information-gain-only, and
  no RL/optimizer/BC/IL/SSCNet training ran.
- Stage 4A-6.3 SSCNet alignment convention fix/reconciliation.
- Stage 4A-6.3 axis audit:
  confirmed `Project2Dto3D` flat -> view `(W,H,D)` -> permute `(D,H,W)`,
  documented the raw Python `(x,y,z)` flatten branch, documented the C++/ROS
  `z*(240*144)+y*240+x` projection flatten path, and selected
  `code_consistent_v1` as the code-consistent Isaac convention.
- Stage 4A-6.3 convention eval:
  tested `current_default_v0`, `xz_swap_diagnostic`, and
  `code_consistent_v1`; `code_consistent_v1` improved occupied Brier from
  `0.2786559495144023` to `0.20511005543694122` and ECE-like from
  `0.3405436085907937` to `0.22427722861569463`; recommendation is
  `code_consistent_v1`.
- Stage 4A-6.3 fixed smokes:
  single-frame map_predict passed, one-step SC-aware expert passed, and
  5-step SC-aware rollout passed with `code_consistent_v1`.
- Stage 4A-6.3 rollout result:
  fixed SC observed_ratio `0.0 -> 0.05899768518518519`, equal to original SC
  and still below empty baseline `0.06896296296296296`; changed actions vs
  original SC `0`.
- Stage 4A-6.3 tests:
  py_compile passed and `test_alignment_convention_fix.py` passed; prediction
  stayed read-only and information-gain-only, observed_state unchanged,
  checkpoint not modified, future observations evaluation-only, and no
  RL/PPO/optimizer/BC/IL/SSCNet training ran.

Next:

- Stage 4A-6.4 should implement calibrated/confidence-gated `I_sc` because
  fixed alignment improves post-hoc diagnostics but the 5-step rollout still
  matches the original SC actions and remains below the empty baseline.
- Use `code_consistent_v1` for future Isaac map_predict runs.
- A 10-step fixed-alignment SC-aware rollout is not the next best step yet;
  first make `gain_sc` selective enough to change the short-horizon action
  sequence.
- If calibrated/confidence-gated `I_sc` still cannot improve the 5-step smoke,
  create a controlled Isaac synthetic SSC validation scene before further SC
  rollout scaling.
- If domain shift dominates after alignment/calibration checks, collect
  Isaac-domain prediction validation data or synthetic supervised data before
  relying on SC-aware rollout.
- If a future lower-weight/cap setting changes selected actions and improves
  observed_ratio, then run a 10-step tuned SC-aware rollout.
- If a future static prediction ablation performs better than dynamic
  prediction, investigate dynamic prediction feedback/noise.
- Stage 4A-7 can add simple prediction fusion across steps only after the
  Stage 4A-6.1/4A-6.2 behavior is understood.
- Keep prediction read-only and information-gain-only when entering rollout.
- Keep prediction separate from observed_map.
- Keep `target_lr` out of expert scoring.
- Keep simulator observed_map measured-only.
- Keep visualization read-only with respect to observed_state.
- Keep scene visualization read-only with respect to observed_map and
  observed_state.

Not next:

- RL, PPO, or policy optimization.
- Behavior cloning or imitation-learning training.
- Jumping directly to RL/IL because the SC-aware 5-step rollout underperformed
  the measured-only baseline in observed_ratio.
- Scaling SC-aware rollouts before explaining/fixing the preprocessing,
  alignment, calibration, and domain-shift issues found in Stage 4A-6.1/6.2.
- Unreal or AirSim.
- Network retraining.
- Full test-set batch inference unless explicitly requested.
- Training on only the 5-sample smoke dataset as a meaningful result.
- Writing PredictionLayer output into observed_map.
- Physical robot path execution or collision-checked control.
- Full SC-Explorer RRT tree planner unless explicitly staged.

Stage 4A-6.5a update / current next:

- Candidate rank sensitivity diagnosis is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65a_rank_sensitivity`.
- If rank diagnosis shows path_cost dominates, next small task is offline
  counterfactual score analysis.
- If candidate fields are insufficient, next small task is improving candidate
  logging only.
- If high-SC candidates are spatially distinct, next small task is spatial
  visualization only.
- Still not RL.

Stage 4A-6.5b update / current next:

- Offline counterfactual score analysis is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65b_counterfactual_scores`.
- If decoupled or normalized formula changes top-1 plausibly, next small task
  is one-step smoke only. Current plausible formula-smoke candidate:
  `decoupled_sc_lambda0p5`.
- If only absurd lambda changes action, next small task is candidate generation
  / spatial review.
- If SC-only picks different far candidates, next small task is spatial
  visualization only.
- Still not RL.

Stage 4A-6.5d update / current next:

- Decoupled one-step spatial visualization is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65d_decoupled_spatial_viz`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_decoupled_one_step_case.py`.
- Result: baseline `grid:15,16,11` and decoupled `grid:14,18,11` are distinct
  candidates but only `0.22360679774997816 m` apart. Decoupled has higher
  `gain_exp` and `effective_gain_sc`, but higher `path_cost`; this is local
  jitter, not enough for rollout.
- Next small task should be candidate generation / path-level or tree-utility
  diagnosis. A one-step formula comparison is still acceptable, but do not
  jump to rollout.
- Still not RL.

Stage 4A-6.5e update / current next:

- Offline candidate generation / path-level utility diagnosis is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65e_path_candidate_diagnosis`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_candidate_generation_path_utility.py`.
- Result: full-64 runtime candidate generation contains wider alternatives
  (median distance `2.321601489576769 m`), but saved top-N candidates and
  selected actions remain local/path-cost dominated. Selected candidates match
  the minimum path-cost candidate in `0.9375` of analyzed sets; path-cost or
  inverse path-cost is the strongest final-score component in `0.96875` of
  sets. High-gain candidates are spatially different from the selected action.
- The 2-step proxy is diagnostic-only, not a true counterfactual tree, and
  fixed next-step utility does not change candidate ranking.
- Next small task should be original SC-Explorer RRT/tree utility source-code
  inspection.
- Still no rollout, no RL, no map_predict rerun, no training.

Stage 4A-6.5f update / current next:

- Original SC-Explorer RRT/tree utility source-code inspection is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65f_original_tree_utility_inspection`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/inspect_original_tree_utility.py`.
- Result: tracked local source configures an external RRT*/tree stack via
  `RRTStar`, `RRTStarEvaluatorAdapter`, `SegmentTime`,
  `GlobalNormalizedGain`, and `SubsequentBest`.
- SC-specific in-repo logic is map/evaluator integration:
  `SSCVoxbloxOccupancyMap`, `SSCExplorationEvaluator`, `SSCServer`, and
  `SSCGrid` network bridge.
- Exact node/tree fields, accumulated branch/path utility formula, and final
  best-node/best-branch/first-path-node selection are not present in the local
  `ssc_exploration` source; they live in external `mav_active_3d_planning` /
  `active_3d_planning_*` dependencies referenced by `.rosinstall` and
  `package.xml`.
- Next small task should be source evidence completion only: inspect/fetch the
  external active_3d_planning source and summarize `RRTStar`,
  `RRTStarEvaluatorAdapter`, `SegmentTime`, `GlobalNormalizedGain`,
  `SubsequentBest`, `ContinuousYawPlanningEvaluator`, and `TrajectorySegment`.
- Still no rollout, no RL, no map_predict rerun, no training, no planner
  implementation.

Stage 4A-6.5g update / current next:

- External active_3d_planning source inspection is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65g_external_active3d_inspection`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/inspect_external_active_3d_planning.py`.
- External source found:
  `/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning`
  at commit `11634e8325480ce5da36a78b23b917347c973613`.
- Result: `TrajectorySegment` stores tree node/edge fields including
  trajectory, local gain/cost/value, parent, children, and info.
  `GlobalNormalizedGain` computes best accumulated subtree `gain / cost`.
  `SubsequentBest` selects the immediate child whose subtree contains the
  highest-value segment, and `OnlinePlanner` executes that child.
- Next small task should be an offline minimal tree-utility prototype over
  saved candidates that reproduces `GlobalNormalizedGain` plus
  `SubsequentBest`; keep it source-faithful and offline.
- Still no rollout, no RL, no map_predict rerun, no training, no planner
  implementation, and no prediction writeback.

Stage 4A-6.5h update / current next:

- Offline minimal tree-utility prototype is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65h_offline_tree_utility_prototype`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_tree_utility_prototype.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_offline_tree_utility_prototype.py`.
- Result: `GlobalNormalizedGain` and `SubsequentBest` are reproduced in a
  minimal offline `OfflineSegment` tree. Synthetic tests passed and show the
  tree utility can overcome a local low-cost trap.
- Real data: 480 candidate rows loaded over 6 configs and steps `0..4`.
  `one_step_star` matched current one-step top-1 in `30/30` default trees.
  Recorded episode chains were built for fixed raw SC and empty baseline, but
  they are not counterfactual trees. Shallow pseudo-trees built 24 default
  trees and changed `SubsequentBest` vs local one-step in `0/24`.
- Interpretation: saved candidates/pseudo-children are insufficient to expose
  source-like branch selection; real child-conditioned tree expansion is still
  missing.
- Next small task should be offline mini-RRT tree builder on saved observed
  map, no Isaac. It should create real parent/children candidate relations
  before any online tree planner smoke or rollout.
- Still no rollout, no RL, no map_predict rerun, no training, no planner
  implementation, and no prediction writeback.

Stage 4A-6.5i update / current next:

- Offline mini-RRT tree builder on saved observed map is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65i_offline_mini_rrt_tree`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_mini_rrt_tree.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_offline_mini_rrt_tree.py`.
- Result: a real parent/children mini-RRT tree was built offline on
  `observed_state_step001.npy` from the fixed SC episode. The tree accepted
  `255` non-root nodes (`256` total), rejected `100` samples, and computed
  valid `GlobalNormalizedGain` / `SubsequentBest` values.
- `SubsequentBest` selected immediate child `n0140` at grid `[14,13,11]`
  with value `286.21642816261226`; its best descendant was itself, only
  `0.11180350549906025 m` from root.
- Interpretation: the mini-RRT created true child-conditioned relations, but
  this run still collapsed to the root-local best child and did not find a
  nonlocal high-gain branch. It differs from the one-step baseline
  `grid:15,16,11` and decoupled `grid:14,18,11`, but remains an even more
  local move.
- Next small task should be inspect gain/raycast or sampling strategy, still
  offline and still no rollout.
- Still no Isaac startup, rollout, online expert loop, map_predict rerun,
  SSCNet inference/training, RL/PPO/BC/IL, checkpoint modification,
  observed_state modification, prediction writeback, target/ground-truth
  scoring, or external source modification/build.

Stage 4A-6.5j update / current next:

- Offline mini-RRT gain/raycast/sampling diagnosis is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65j_gain_raycast_sampling_diagnosis`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_mini_rrt_gain_sampling.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_mini_rrt_gain_sampling_diagnosis.py`.
- Result: selected child `n0140` gain `32.0` is reproducible and is not
  duplicate root/parent visible unknown under recorded yaw (`0/32` overlap).
  The collapse is mainly short-edge/cost normalization plus sampling/steering
  discretization, not a raycast mismatch.
- Segment diagnosis: median segment length `0.141421356237309m`, p75
  `0.20000000000000018m`; local gain/cost is strongly correlated with inverse
  segment length (`0.9282421006554769`).
- Sampling diagnosis: accepted `255`, rejected `100`, with
  `target_same_as_nearest=59` and `edge_non_traversable_or_unknown=41`;
  far targets often shrink to short snapped/voxelized edges.
- Filter/rerank diagnosis: immediate root-child min segment length `0.15m`
  or min root distance `0.25m` moves the winner off `n0140`; root/parent
  novelty does not move the immediate root-child winner.
- Source evidence: external active_3d_planning has `min_path_length`,
  `crop_min_length`, `max_density_range`, root rewiring/reinsert, optional
  parent visible clearing, and continuous yaw; no mandatory root-visible
  overlap filter or near-root gain discount was proven.
- Next small task should be offline mini-RRT minimum-edge-length variant, no
  Isaac. It should test source-like `min_path_length` / `crop_min_length`
  behavior before any online tree smoke.
- Still no rollout, no RL, no map_predict rerun, no SSCNet inference/training,
  no planner runtime change, no prediction writeback, no target/ground-truth
  scoring, and no coverage-improvement claim.

Stage 4A-6.5k update / current next:

- Offline mini-RRT minimum-edge-length / crop-min-length variant is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65k_min_edge_length_variant`.
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_mini_rrt_tree.py`
  with optional min-edge, root-child, root-distance, crop-min-length, density,
  and variant-name parameters while preserving default Stage 4A-6.5i behavior.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_mini_rrt_min_edge_variants.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_mini_rrt_min_edge_variants.py`.
- Result: all 9 variants completed. `baseline_allow` reproduced `n0140`.
  Every non-baseline variant moved off `n0140`.
- Nonlocal branch variants:
  `reject_root_child_0p25`, `reject_root_distance_0p25`, and
  `crop_min_length_0p25`.
- Best next candidate: `crop_min_length_0p25`, with selected child `n0001`
  at grid `[18,12,11]`, selected distance `0.5123476174067144m`, best
  descendant `n0249` at grid `[39,19,11]`, best descendant distance
  `2.6688013442592453m`, accepted nodes `255`, and rejected samples `916`.
- Density limiting worked as an anti-clustering mechanism but was too
  restrictive at radius `0.25m` / max nodes `1`: it accepted `86` nodes and
  selected a local child.
- Next small task should be no-prediction online one-step tree smoke with
  source-like crop/min-length settings, still no rollout.
- Still not next: rollout, RL, PPO, BC/IL training, map_predict tree
  integration, prediction writeback, observed_map prediction fusion, target or
  ground-truth scoring, checkpoint changes, or external source build.

Stage 4A-6.5l update / current next:

- Source-protected no-prediction one-step tree smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65l_source_protected_one_step_tree_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_source_protected_one_step_tree_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_source_protected_one_step_tree_smoke.py`.
- Protection profile:
  `source_like_crop_min_length_0p25`, `short_edge_policy=crop`,
  `crop_min_length_m=0.25`, `num_nodes=256`, `sample_mode=mixed`,
  `gain_mode=exp`, `path_cost_mode=segment_time`, `num_yaw_samples=8`.
- Protection checklist:
  crop/min-path-length active at `0.25m`; density limiting / max density range
  implemented but inactive; continuous yaw approximated with 8 fixed samples
  and active; root rewiring/reinsert present only as inactive hook/checklist;
  optional parent visible clearing and root-visible filtering / near-root
  discount intentionally inactive.
- Result: exactly reproduced Stage 4A-6.5k `crop_min_length_0p25`.
  Selected child `n0001` at grid `[18,12,11]`, distance
  `0.5123476174067144m`; best descendant `n0249` at grid `[39,19,11]`,
  distance `2.6688013442592453m`; accumulated gain/cost
  `645.0 / 4.565369444959812`; value `141.28100864040323`; accepted nodes
  `255`; rejected samples `916`.
- The selected child moved off `n0140`, differs from one-step baseline
  `[15,16,11]`, differs from decoupled `[14,18,11]`, remains measured-only
  `gain_exp`, and uses no prediction/map_predict.
- Validation passed: `py_compile`,
  `test_source_protected_one_step_tree_smoke.py`, observed_state hash
  unchanged, checkpoint hash unchanged, external active_3d_planning git status
  clean, no rollout-like outputs, and no map_predict artifacts.
- Next small task should be no-prediction Isaac one-step capture + tree
  decision smoke, still no rollout.
- Still not next: rollout, RL, PPO, BC/IL training, map_predict tree
  integration, prediction writeback, observed_map prediction fusion, target or
  ground-truth scoring, checkpoint changes, or external source build.

Stage 4A-6.5m update / current next:

- No-prediction Isaac one-step capture + source-protected tree decision smoke
  is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65m_isaac_one_step_tree_capture_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_one_step_tree_capture_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_one_step_tree_capture_smoke.py`.
- Result: one Isaac headless startup, deterministic `medium_three_rooms` seed
  `0`, one fixed-pose RGB/depth capture at `pose_001.json`, and measured-only
  update to a new output observed map.
- Prior `observed_state_step000.npy` hash unchanged; new
  `observed_state_isaac_capture_step001.npy` exactly matched saved
  `observed_state_step001.npy`.
- Source-protected tree decision exactly matched Stage 4A-6.5l:
  selected child `n0001` at grid `[18,12,11]`, distance
  `0.5123476174067144m`; best descendant `n0249` at grid `[39,19,11]`,
  distance `2.6688013442592453m`; accumulated gain/cost
  `645.0 / 4.565369444959812`; value `141.28100864040323`; accepted
  nodes `255`; rejected samples `916`.
- Active protections: `short_edge_policy=crop`, `crop_min_length_m=0.25`,
  and `num_yaw_samples=8`. Density limiting is implemented but inactive.
- Validation passed: `py_compile`,
  `test_isaac_one_step_tree_capture_smoke.py`, no prohibited rollout outputs,
  no map_predict artifacts, no prediction use, prior observed map unchanged,
  checkpoint unchanged, and external active_3d_planning source clean.
- Next small task should be no-prediction two-frame tree smoke.
- Still not next: rollout, RL, PPO, BC/IL training, map_predict tree
  integration, prediction writeback, observed_map prediction fusion, target or
  ground-truth scoring, checkpoint changes, coverage-improvement claims, or
  external source build.

Stage 4A-6.5n update / current next:

- No-prediction two-frame source-protected tree smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65n_two_frame_tree_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_two_frame_tree_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_two_frame_tree_smoke.py`.
- Result: Isaac started headless once, captured exactly two frames, executed
  exactly one selected-child move, and stopped after the second tree decision.
- Frame 1 exactly reproduced Stage 4A-6.5m / 6.5l:
  selected child `n0001` at grid `[18,12,11]`; best descendant `n0249` at
  grid `[39,19,11]`; accumulated gain/cost
  `645.0 / 4.565369444959812`; value `141.28100864040323`.
- Move once: camera moved to selected child x/y `[-4.15,-4.75]`, fixed height
  `1.2m`, yaw `2.15879915042112`.
- Frame 2 measured-only update added `6251` observed voxels and the tree
  decision remained nonlocal: selected child `n0001` at grid `[17,16,11]`;
  best descendant `n0112` at grid `[8,27,11]`; accumulated gain/cost
  `323.0 / 2.315392939101747`; value `139.50115962835548`.
- Validation passed: `py_compile`, `test_isaac_two_frame_tree_smoke.py`, no
  third frame, no rollout outputs, no map_predict artifacts, prediction
  disabled, checkpoint unchanged, prior observed_state unchanged, and external
  source unchanged.
- Next small task should be map_predict + source-protected tree one-step
  smoke, still no rollout.
- Still not next: rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target or ground-truth
  scoring, checkpoint changes, coverage-improvement claims, or external source
  build.

Stage 4A-6.5o update / current next:

- map_predict + source-protected tree one-step smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65o_map_predict_tree_one_step_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_tree_one_step_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_map_predict_tree_one_step_smoke.py`.
- Updated `offline_mini_rrt_tree.py` to record `gain_occ` and `gain_conf` on
  tree segments and CSV rows while preserving default measured-only
  `gain_mode=exp`.
- Result: Isaac started headless once, captured exactly one frame, ran one
  measured-only observed_state update, ran one map_predict call, and ran one
  measured-only baseline tree plus one SC-tree decision.
- measured-only update added `3727` observed voxels; new observed_state hash
  `afacb32647bfa1b2ece34b75f19cb34cd15ec742b43e02b682af0fcd5f2bc59e`;
  prior `observed_state_step000.npy` unchanged.
- map_predict used `alignment_convention=code_consistent_v1`, `tau=0.1`, and
  produced a shape-aligned read-only prediction layer `(120,120,30)` with
  `prediction_valid_count=57382`,
  `predicted_unmeasured_count=37537`, and
  `predicted_occupied_count=16779`.
- Source-protected SC tree used `prediction_mode=sim_dynamic`,
  `gain_mode=hybrid`, `sc_gain_formula=raw_count`, and
  `gain_hybrid = gain_exp + gain_sc`.
- SC gain was nonzero: `255/255` tree nodes had `gain_sc > 0`;
  `gain_sc` min/mean/max `15.0 / 48.88235294117647 / 62.0`.
- Selected child did not change from measured-only in this one-step smoke:
  measured-only selected `n0001` grid `[18,12,11]`; SC selected `n0001` grid
  `[18,12,11]`. Therefore no spatially meaningful selected-child change was
  observed at one step.
- SC best descendant remained `n0249` grid `[39,19,11]`; accumulated
  gain/cost `1258.0 / 4.565369444959812`; value
  `275.55491379794235`.
- Validation passed: `py_compile`, `test_map_predict_tree_one_step_smoke.py`,
  existing `test_offline_mini_rrt_tree.py`, and Stage 4A-6.5l regression.
- Safety passed: one frame only, no selected action execution, no two-frame
  loop, no rollout, checkpoint unchanged, prior observed_state unchanged,
  no dense `class_prob`, prediction stayed read-only and information-gain-only,
  no prediction traversability/collision/ray blocking, no target/ground truth,
  and external source unchanged.
- Next small task should be map_predict + source-protected tree two-frame
  smoke.
- Still not next: rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target or ground-truth
  scoring, checkpoint changes, coverage-improvement claims, or external source
  build.

Stage 4A-6.5p update / current next:

- map_predict + source-protected tree two-frame smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65p_map_predict_tree_two_frame_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_tree_two_frame_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_map_predict_tree_two_frame_smoke.py`.
- Result: Isaac started once, captured exactly two frames, executed exactly
  one SC selected-child move, ran exactly two map_predict calls through one
  loaded `IsaacMapPredictor`, and ran measured-only plus SC/hybrid
  source-protected tree decisions in both frames.
- Frame 1: SC selected child stayed `n0001`, grid `[18,12,11]`; best
  descendant stayed `n0249`, grid `[39,19,11]`; accumulated
  gain_exp/gain_sc/gain_hybrid/cost `645.0 / 638.0 / 1283.0 /
  4.565369444959812`; value `281.02873501649196`; `255/255` nodes had
  `gain_sc > 0`.
- Frame 2: measured-only matched Stage 4A-6.5n frame 2 (`n0001` -> `n0112`);
  SC changed selected child to `n0127`, grid `[11,15,11]`, and best
  descendant to `n0162`, grid `[14,15,11]`; accumulated
  gain_exp/gain_sc/gain_hybrid/cost `76.0 / 75.0 / 151.0 /
  0.5872281406276059`; value `257.14026551693735`; `248/255` nodes had
  `gain_sc > 0`.
- Prediction stayed read-only and information-gain-only. It did not write
  observed_state, did not participate in traversability/collision/ray
  blocking, did not use target/ground-truth scoring, and did not modify the
  checkpoint or external source.
- Validation passed: `py_compile` and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65p_map_predict_tree_two_frame_smoke_test.log`.
- Because SC changed the selected child meaningfully in frame 2, the next
  small task should be controlled gated SC tree one-step smoke or a repeated
  two-frame smoke, not rollout.
- Still not next: rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target or ground-truth
  scoring, checkpoint changes, coverage-improvement claims, or external source
  build.

Stage 4A-6.5q update / current next:

- SC-tree branch-change diagnosis and gated SC tree one-step replay is
  complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65q_sc_tree_branch_change_diagnosis`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_sc_tree_branch_change.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sc_tree_branch_change_diagnosis.py`.
- Result: frame2 SC selected child `n0127` is spatially meaningful versus
  measured `n0001` (`0.6082762530298217m`), but the branch is local and the
  raw-hybrid margin is narrow (`0.5963372839863723` value).
- Cause: lower path cost plus higher SC-only value per cost; exp-only value
  alone would favor the measured branch.
- Gated replay: raw_count, weight `1.0`, cap `25`, cap `50`, and
  confidence-weighted gain preserved `n0127`; weights `0.0`, `0.25`, `0.5`,
  and occupied-only returned to measured `n0001`. Minimum SC weight was
  `0.899353934095411`.
- Validation passed: `py_compile` and
  `test_sc_tree_branch_change_diagnosis.py`.
- Safety passed: no Isaac startup, no map_predict rerun, no SSCNet inference,
  no rollout, no selected action execution, no training/RL, no checkpoint
  modification, no observed_state modification, no prediction writeback, no
  prediction traversability/collision/ray blocking, no target/ground-truth
  scoring, and no external source modification/build.
- Next small task should be gated SC tree one-step smoke, using a conservative
  gated formula such as confidence-weighted or cap `25`.
- Still not next: rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target or ground-truth
  scoring, checkpoint changes, coverage-improvement claims, or external source
  build.

Stage 4A-6.5r update / current next:

- Gated SC tree one-step smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65r_gated_sc_tree_one_step_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_gated_sc_tree_one_step_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_gated_sc_tree_one_step_smoke.py`.
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_mini_rrt_tree.py`
  with explicit optional gated SC runtime formulas while preserving default
  measured-only `gain_mode=exp` and prediction-disabled behavior.
- Result on Stage 4A-6.5p Frame 2 saved artifacts:
  measured-only selected `n0001`, grid `[17,16,11]`, best `n0112`;
  raw_count reproduced Stage 4A-6.5p with selected `n0127`, grid
  `[11,15,11]`, best `n0162`;
  weight `0.5` and occupied-only returned to measured `n0001`;
  weight `1.0`, cap `25`, cap `50`, confidence-weighted, and
  confidence-weighted cap `25` preserved `n0127`.
- Safety passed: no Isaac startup, no new RGB/depth capture, no map_predict
  rerun, no SSCNet inference, no two-frame run, no selected action execution,
  no rollout, no training/RL/PPO/BC/IL, no checkpoint change, no observed_state
  modification, no prediction writeback, no prediction traversability /
  collision / ray blocking, no target/ground-truth scoring, no external source
  modification/build, and no coverage-improvement claim.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_gated_sc_tree_one_step_smoke_test.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_offline_mini_rrt_tree_regression.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_source_protected_one_step_regression.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_isaac_two_frame_tree_regression.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_map_predict_tree_one_step_regression.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_map_predict_tree_two_frame_regression.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_sc_tree_branch_change_regression.log`.
- Next small task should choose a conservative gated formula for a later
  staged smoke or repeat saved/two-frame validation if needed.
- Still not next: direct rollout, online open-ended loop, RL/PPO/BC/IL
  training, prediction writeback, observed_map prediction fusion, target or
  ground-truth scoring, checkpoint changes, coverage-improvement claims, or
  external source build.

Stage 4A-6.5s update / current next:

- Confidence-weighted / cap25 gated SC tree two-frame smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65s_gated_sc_tree_two_frame_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_gated_sc_tree_two_frame_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_gated_sc_tree_two_frame_smoke.py`.
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_tree_two_frame_smoke.py`
  so its default remains `raw_count`, but explicit gated `--sc_gain_formula`
  values are passed through to the mini-RRT tree.
- Primary executed formula: `confidence_weighted`.
- Shadow same-frame formula: `cap25`.
- Result: one Isaac startup, exactly two captured frames, exactly two
  map_predict calls, exactly one `confidence_weighted` selected-child move,
  no third frame, and no second action.
- Frame 1: measured-only and `confidence_weighted` both selected `n0001`,
  grid `[18,12,11]`, best `n0249`; `cap25` shadow selected `n0196`, grid
  `[15,11,11]`, best `n0196`, and was not executed.
- Frame 2: measured-only selected `n0001`, grid `[17,16,11]`, best `n0112`;
  `confidence_weighted` selected `n0127`, grid `[11,15,11]`, best `n0162`;
  `cap25` shadow also selected `n0127`, best `n0162`.
- Frame 2 winning-branch effective SC gain:
  `confidence_weighted=31.506256222724915`, `cap25=50.0`, raw gain_sc
  `75.0`, accumulated cost `0.5872281406276059`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65s_gated_sc_tree_two_frame_smoke.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65s_gated_sc_tree_two_frame_smoke_test.log`,
  plus py_compile and Stage 4A-6.5r/4A-6.5p regressions.
- Safety passed: prediction stayed read-only and information-gain-only, no
  rollout, no open-ended loop, no training/RL/PPO/BC/IL, no checkpoint change,
  no existing observed_state modification, no prediction writeback, no
  prediction traversability/collision/ray blocking, no target/ground-truth
  scoring, no external source modification/build, and no coverage-improvement
  claim.
- Next small task can be repeated gated two-frame smoke or a short gated SC
  tree smoke, still tightly staged.
- Still not next: direct rollout, online open-ended loop, third-frame/second
  action in this smoke, RL/PPO/BC/IL training, prediction writeback,
  observed_map prediction fusion, target/ground-truth scoring, checkpoint
  changes, coverage-improvement claims, or external source build.

Stage 4A-6.5t update / current next:

- Alternate-tree-seed repeated confidence-weighted/cap25 gated SC tree
  two-frame smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_gated_sc_tree_seed_repeat_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_gated_sc_tree_seed_repeat_smoke.py`.
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_tree_two_frame_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_gated_sc_tree_two_frame_smoke.py`
  to record seed-qualified profile names without changing seed-0 behavior.
- Only the tree/random seed was changed to `1`; scene seed remained `0`,
  scene remained `medium_three_rooms`, primary formula remained
  `confidence_weighted`, and cap25 remained shadow-only.
- Result: one Isaac startup, exactly two captured frames, exactly two
  map_predict calls, exactly one `confidence_weighted` selected-child move,
  no third frame, and no second action.
- Frame 1: measured-only, `confidence_weighted`, and cap25 shadow all selected
  `n0001`, grid `[18,12,11]`, best `n0245`, grid `[33,14,11]`; frame-1 cap25
  was no longer branch-more-aggressive under seed `1`.
- Frame 2: measured-only selected `n0057`, grid `[12,16,11]`, best `n0118`,
  grid `[12,19,11]`; `confidence_weighted` also selected `n0057 -> n0118`;
  cap25 shadow matched `confidence_weighted`.
- Frame 2 did not reproduce the exact Stage 4A-6.5s `n0127 -> n0162` branch,
  but it was spatially close to that reference branch (`0.14142135623730964m`
  selected-child delta and `0.4472135954999583m` best-descendant delta).
- Frame 2 effective SC gain:
  `confidence_weighted=35.51751762628555`, `cap25=50.0`, raw gain_sc `82.0`,
  accumulated cost `0.620156278894175`.
- Prediction stayed read-only and information-gain-only; prediction did not
  write observed_state, did not affect traversability/collision, did not block
  rays, and did not use target/ground-truth scoring.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65t_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke_test.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65t_stage4a65s_gated_sc_tree_two_frame_regression.log`.
- Current next small task should be seed robustness diagnosis before extending
  beyond two frames.
- Still not next: rollout, online open-ended loop, third-frame/second action
  in this smoke, RL/PPO/BC/IL training, prediction writeback, observed_map
  prediction fusion, target/ground-truth scoring, checkpoint changes,
  coverage-improvement claims, or external source build.

Stage 4A-6.5u update / current next:

- Offline seed robustness diagnosis is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65u_seed_robustness_diagnosis`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_gated_tree_seed_robustness.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_gated_tree_seed_robustness_diagnosis.py`.
- Result: seed1 Frame 2 `confidence_weighted` returned to measured-only in
  seed1 score/tree space (`n0057 -> n0118`) but remained spatially close to
  the seed0 SC branch (`n0127 -> n0162`): selected-child delta
  `0.14142135623730964m`, best-descendant delta `0.4472135954999583m`.
- Classification: seed1 confidence and cap25 are both `same_as_measured` and
  `spatially_same_as_seed0_sc`; seed0 confidence is
  `spatially_same_as_seed0_sc`.
- Rank/margin: seed0 confidence margin was narrow (`2.3578116606364574`,
  normalized `0.012879002637364282`); seed1 margin was healthier
  (`31.643506691543223`, normalized `0.1563440683986311`).
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65u_seed_robustness_diagnosis.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65u_seed_robustness_diagnosis_test.log`.
- Current next small task should be multi-seed offline replay / seed robustness
  sweep before any longer smoke.
- Still not next: rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, or external source
  build.

Stage 4A-6.5v update / current next:

- Multi-seed offline replay / seed robustness sweep is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65v_multi_seed_offline_replay`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_multi_seed_gated_tree_replay.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_multi_seed_gated_tree_replay.py`.
- Completed 10 tree seeds (`0..9`) for `measured_only`,
  `confidence_weighted`, `cap25`, and `raw_count` on the saved Stage 4A-6.5p
  Frame 2 observed_state + prediction.
- Seed 0 confidence replay exactly matched Stage 4A-6.5s
  (`n0127 -> n0162`); seed 1 confidence replay exactly matched Stage 4A-6.5t
  (`n0057 -> n0118`).
- `confidence_weighted`: exact seed0 SC fraction `0.1`, spatial seed0 SC
  basin fraction `0.3`, same-as-measured fraction `0.7`,
  measured-but-seed0-SC-basin fraction `0.1`, distinct SC branch fraction
  `0.1`.
- `cap25` spatial seed0 SC basin fraction `0.5`;
  confidence/cap25 exact agreement `0.8`.
- `raw_count` matched confidence's broad behavior in this sweep
  (`0.3` spatial seed0 SC basin, `0.7` same-as-measured).
- Confidence margins were not mostly narrow: normalized min/median/max
  `0.012879002637364282 / 0.12317906042561935 /
  0.22865634393797724`; narrow seeds were `[0,2,8]`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65v_multi_seed_gated_tree_replay.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65v_multi_seed_gated_tree_replay_test.log`.
- Current next small task should be tree sampling stabilization or SC gain
  design review.
- Still not next: rollout, online open-ended loop, 3-frame smoke, another
  start/scene two-frame smoke, RL/PPO/BC/IL training, prediction writeback,
  observed_map prediction fusion, target/ground-truth scoring, checkpoint
  changes, coverage-improvement claims, or external source build.

Stage 4A-6.5w update / current next:

- Source-faithful RRTStar root-rewire / tree-persistence stabilization
  prototype is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65w_source_faithful_rewire_persistence`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_source_faithful_rewire_persistence.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_source_faithful_rewire_persistence.py`.
- Completed 10 tree seeds (`0..9`) for `measured_only`,
  `confidence_weighted`, `cap25`, and `raw_count` across five configs:
  `fresh_random_256_baseline`,
  `persistent_rewire_256_no_density`,
  `persistent_rewire_512_no_density`,
  `persistent_rewire_256_source_density`, and
  `persistent_rewire_512_source_density`.
- Fresh baseline reproduced Stage 4A-6.5v:
  confidence spatial seed0 SC basin `0.3`, same-as-measured `0.7`,
  confidence/cap25 agreement `0.8`.
- Persistent no-density preserved many branches but did not stabilize the
  seed0-SC basin:
  256 confidence spatial basin `0.0`, same-as-measured `0.5`;
  512 confidence spatial basin `0.0`, same-as-measured `0.7`.
- Preservation diagnostics:
  256 no-density preserved-subtree winner fraction `0.875`, newly-expanded
  selected fraction `0.1`; 512 no-density preserved-subtree winner fraction
  `0.85`, newly-expanded selected fraction `0.125`.
- Source density evidence found exact source config values
  (`max_density_range: 1.0`), but source-like density was too restrictive in
  this mini-RRT/profile and produced no valid preserved roots/winners.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65w_source_faithful_rewire_persistence.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65w_source_faithful_rewire_persistence_test.log`.
- Current next small task should be SC gain design review, not another runtime
  smoke.
- Still not next: rollout, online open-ended loop, 3-frame smoke, another
  start/scene two-frame smoke, RL/PPO/BC/IL training, prediction writeback,
  observed_map prediction fusion, target/ground-truth scoring, checkpoint
  changes, coverage-improvement claims, or external source build.

Stage 4A-6.5x update / current next:

- Source-faithful SC gain design review and visible-voxel decomposition is
  complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65x_sc_gain_design_review`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/review_source_sc_gain_design.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_source_sc_gain_design_review.py`.
- Source evidence: active `sc_explorer.yaml` rewards predicted occupied and
  predicted free voxels with weights `1.0 / 1.0`, sets
  `unobserved_weight: 0.0`, uses `ssc_confidence_threshold: 0.05`, and does
  not enable direct `weight_by_confidence`.
- Current formulas: `raw_count` is only source-inspired unless restricted to
  source OCC/FREE threshold semantics; `confidence_weighted`, `cap25`,
  `occupied_only`, `occupied_margin`, `calibrated_occupied`,
  `novelty_discounted`, and branch normalization are diagnostic or
  source-inspired, not exact source-faithful formulas.
- Decomposition result: seed0 SC `n0127 -> n0162` has raw/source OCC+FREE SC
  `136.0 / 135.0`, so its advantage is not dominated by source-unknown
  predicted voxels. It is a short low-cost local branch with root-overlap /
  low-novelty prediction visibility. The seed0 measured branch sees more total
  source OCC+FREE prediction (`569.0`) but has much higher path cost.
- Candidate proxy result: source OCC+FREE and source-thresholded OCC/FREE
  select measured in the seed0 proxy; parent-visible-cleared and
  spatial-normalized diagnostics can still keep the short SC branch; frontier
  local selects measured in the seed0 proxy.
- Current next small task should be offline source OCC+FREE plus
  parent-visible-cleared/frontier-local seed replay over saved Frame2
  artifacts.
- Still not next: runtime smoke, rollout, online open-ended loop, 3-frame
  smoke, another start/scene two-frame smoke, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, or external
  source build.

Stage 4A-6.5y update / current next:

- Offline source OCC+FREE plus parent-visible-cleared/frontier-local seed
  replay is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65y_source_gain_seed_replay`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_source_gain_seed_replay.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_source_gain_seed_replay.py`.
- Completed 10 seeds (`0..9`) and 11 formulas:
  `measured_only`, `current_confidence_weighted`, `current_cap25`,
  `current_raw_count`, `source_occ_free`, `source_occ_free_thresholded`,
  `parent_visible_cleared_source_occ_free`,
  `root_visible_cleared_source_occ_free`,
  `frontier_local_source_occ_free`,
  `parent_cleared_frontier_local_source_occ_free`, and
  `branch_normalized_source_occ_free`.
- Prediction NPZ inventory and source OCC/FREE mapping reports were written.
  Mapping status is `source-faithful-approx`, because the saved simulator NPZ
  has the needed probability/class fields but is not the exact C++ SSCMap
  log-odds layer.
- Seed0 `current_confidence_weighted` reproduced Stage 4A-6.5s
  `n0127 -> n0162`.
- Seed0 `source_occ_free` did not select measured; it also selected
  `n0127 -> n0162`. Source OCC+FREE alone therefore did not remove the short
  low-cost local SC branch in full-tree replay.
- Main fractions:
  `current_confidence_weighted` spatial seed0 SC basin `0.3`,
  same-as-measured `0.7`; `source_occ_free` spatial seed0 SC basin `0.4`,
  same-as-measured `0.6`; `parent_visible_cleared_source_occ_free` spatial
  seed0 SC basin `0.5`, same-as-measured `0.7`;
  `frontier_local_source_occ_free` spatial seed0 SC basin `0.4`,
  same-as-measured `0.6`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65y_source_gain_seed_replay.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65y_source_gain_seed_replay_test.log`.
- Current next small task should remain offline: inspect source OCC/FREE
  mapping and source-inspired novelty filters before any runtime smoke.
- Still not next: runtime smoke, rollout, online open-ended loop, 3-frame
  smoke, another start/scene two-frame smoke, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, or external
  source build.

Stage 4A-6.5z update / current next:

- Offline decoupled SC utility sweep is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65z_decoupled_sc_utility_sweep`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_decoupled_sc_utility_sweep.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_decoupled_sc_utility_sweep.py`.
- Tested `value = gain_exp / cost + lambda * normalized_sc`, with SC outside
  the cost denominator.
- Completed 10 seeds (`0..9`), 3 SC bases, fixed lambdas
  `0,1,2,4,8,12,16,24,32`, and adaptive lambdas
  `0.25,0.5,1.0,2.0 * (p90(base_exp_value)-p50(base_exp_value))`.
- Decision rows: `390`.
- Seed0 base gap: measured `139.50115962835548`, seed0 SC
  `129.42159059130623`, gap `10.079569037049254`.
- Result: all tested fixed/adaptive lambda variants and all three SC bases had
  spatial seed0 SC basin fraction `0.0` and same-as-measured fraction `1.0`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65z_decoupled_sc_utility_sweep.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65z_decoupled_sc_utility_sweep_test.log`.
- Current next small task should remain offline: inspect the decoupled sweep
  tables and decide whether the diagnostic utility deserves a later explicit
  runtime flag.
- Still not next: runtime smoke, rollout, online open-ended loop, 3-frame
  smoke, another start/scene two-frame smoke, Pareto dominance gate, new
  runtime planner, RL/PPO/BC/IL training, prediction writeback, observed_map
  prediction fusion, target/ground-truth scoring, checkpoint changes,
  coverage-improvement claims, or external source build.

Stage 4A-6.5z.1 update / current next:

- Offline decoupled SC signal-strength and normalization diagnosis is
  complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65z1_decoupled_signal_strength_diagnosis`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_decoupled_sc_signal_strength.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_decoupled_sc_signal_strength.py`.
- The diagnosis read saved 6.5z tables and regenerated debug-only per-path
  components from saved 6.5y raw trees plus saved 6.5p Frame2
  observed_state/prediction NPZ. It did not start Isaac, capture frames, rerun
  map_predict, run SSCNet inference, execute actions, run runtime/two-frame
  smoke, run rollout, train, modify checkpoints/observed_state/prediction NPZ,
  write prediction into observed_map, use prediction for collision/
  traversability/ray blocking, use target/ground truth, implement Pareto or a
  runtime planner, or claim coverage improvement.
- Outputs include all required z.1 CSV/JSON/MD/PNG files (`38` required files
  checked), including near-miss, required-lambda, normalization,
  adaptive-gap, measured-vs-nonmeasured rank, low-cost follow-up, and
  debug-regeneration reports.
- Debug regeneration produced `7650` candidate path rows, `3003` near-miss
  rows, `4320` non-measured required-lambda rows, and `1443`
  impossible-under-positive-lambda rows.
- Important correction: the 6.5z lambda-sweep summary claims all decoupled
  variants are measured-only, but corrected per-formula/per-seed
  classification rows are not all measured-only. This inconsistency is
  explicitly recorded in z.1 `missing_fields_report.json` and the summary.
- Required-lambda result: finite p50/p90/max
  `229.31585862120286 / 627.9926880897762 / 34462.89245592027`; only `111`
  finite rows were `<=32`, and `173` were `<= adaptive 2x`.
- Seed0 reference SC branch `n0127 -> n0162` is impossible to promote with
  positive lambda under all three SC bases because its normalized SC is lower
  than the measured-like branch.
- Normalization result: measured winners were top-SC-quartile in `15/30`
  seed/basis rows, max SC belonged to measured in `18/30`, and normalized-SC
  IQR was `<0.10` in `14/30`.
- Recommended current next small task: larger offline lambda diagnostic sweep
  only, because some non-measured candidates can theoretically flip but most
  require lambda values above `32`.
- Still not next: runtime smoke, rollout, online open-ended loop, 3-frame
  smoke, another start/scene two-frame smoke, Pareto dominance gate
  implementation, new runtime planner implementation, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, or external
  source build.

Stage 4A-6.5aa update / current next:

- Controlled synthetic SC validation scene smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65aa_synthetic_sc_validation`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_synthetic_sc_validation_scene.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_synthetic_sc_validation_scene.py`.
- Scene variant:
  `synthetic_hidden_room_frontier`.
- Exactly one frame was captured. No selected action, two-frame runtime,
  rollout, open-ended loop, training/RL/PPO/BC/IL, prediction writeback,
  prediction traversability/collision/ray blocking, target/ground-truth
  planning/scoring, checkpoint modification, existing observed_state
  modification, external source modification/build, or coverage-improvement
  claim occurred.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aa_synthetic_sc_validation.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aa_synthetic_sc_validation_test.log`.
- Key result:
  measured-only chose measured-frontier in `5/5` seeds; Oracle source OCC+FREE
  over-cost chose hidden-room in `5/5`; map_predict source OCC+FREE over-cost
  also chose hidden-room in `5/5`; low-cost artifact flags were `0/45`; and
  map/oracle direction agreement was `0.95`.
- Decoupled minmax result:
  lambda `8` and `16` stayed measured-frontier for both Oracle and map_predict;
  lambda `32` selected hidden-room for Oracle in `4/5` seeds and map_predict
  in `3/5` seeds.
- Current next small task:
  repeat a tiny controlled map_predict calibration smoke before any runtime
  smoke.
- Still not next:
  rollout, online open-ended loop, RL/PPO/BC/IL training, prediction writeback,
  observed_map prediction fusion, target/ground-truth scoring, checkpoint
  changes, coverage-improvement claims, external source modification/build, or
  a new runtime planner.

Stage 4A-6.5ab update / current next:

- Tiny controlled map_predict calibration smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ab_synthetic_calibration_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_synthetic_map_predict_calibration_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_synthetic_map_predict_calibration_smoke.py`.
- It used only the saved Stage 4A-6.5aa synthetic frame, Oracle NPZ,
  map_predict NPZ, and raw mini-RRT trees. No Isaac startup, new capture,
  map_predict rerun, SSCNet inference, action execution, two-frame runtime,
  rollout, training/RL/PPO/BC/IL, checkpoint edit, existing observed_state or
  prediction NPZ edit, prediction writeback, prediction collision/
  traversability/ray-blocking use, target/ground-truth scoring, external
  source build/edit, or coverage-improvement claim occurred.
- Completed `96` configs and `485` decision rows over seeds `0..4`.
- Best candidate:
  `map_predict|source_occ_free|decoupled_minmax_lambda48|tau0p1|occ0p5|free0p5`.
- Best candidate result:
  hidden-room `5/5`, Oracle/map_predict agreement `1.0`, low-cost artifact
  fraction `0.0`, mean selected SC gain `5018.4`, and median margin
  `21.285778495568792`.
- Decoupled lambda `48` was stable for map_predict source OCC+FREE minmax
  across tau `0.1`, `0.4`, and `0.8`; lambda `32` was not stable enough.
- Across all calibration rows there were `3/485` low-cost artifact flags, but
  the recommended config had `0.0`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ab_synthetic_calibration_smoke.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ab_synthetic_calibration_smoke_test.log`.
- Current next small task:
  saved-frame one-step formula smoke only.
- Still not next:
  runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, external source
  modification/build, Pareto gate implementation, or runtime planner
  implementation.

Stage 4A-6.5ai update / current next:

- Staged one-frame lambda48 runtime smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ai_one_frame_lambda48_runtime_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65ai_one_frame_lambda48_runtime_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ai_one_frame_lambda48_runtime_smoke.py`.
- Runtime setup:
  one Isaac startup, deterministic `medium_three_rooms` scene seed `0`,
  canonical start pose `[-4.65, -4.65, 1.2]`, yaw
  `0.38710316317995463`, exactly one RGB/depth frame, exactly one
  measured-only observed_state update, exactly one map_predict call with
  `code_consistent_v1`, and no selected action execution, second frame,
  two-frame runtime, or rollout.
- Observed_state result:
  shape `(120,120,30)`, observed_ratio `0.0425462962962963`.
- map_predict result:
  prediction shape aligned to observed_state, valid predictions `57382`,
  predicted unmeasured OCC+FREE `40328` (`9494` occupied, `30834` free).
- Formula:
  `gain_exp / cost + 48 * minmax(source_occ_free)` with tau `0.1` and
  occ/free thresholds `0.5/0.5`; over-cost was not executed as runtime
  primary.
- Decisions:
  measured-only shadow selected `n0013 -> n0159`; lambda48 primary selected
  `n0001 -> n0228`; lambda32 shadow selected `n0001 -> n0228`.
- Lambda48 classification:
  `distinct_nonmeasured_branch`, healthy non-measured `true`,
  low-cost artifact `false`, historical prior basin `false`.
- Hardware use was explicitly recorded:
  `os_cpu_count=32`, requested/actual `--max_workers 32/32`,
  `parallel_backend=single_process_runtime_stage_no_process_pool`,
  OMP `1`, OPENBLAS `32`, MKL `1`, NUMEXPR `1`, VECLIB `1`,
  GPU `NVIDIA GeForce RTX 5080`, total wall time `29.64396972299801s`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ai_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ai_one_frame_lambda48_runtime_smoke.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ai_one_frame_lambda48_runtime_smoke_test.log`.
- Safety passed:
  no action, no second frame, no rollout/open-ended loop, no
  training/RL/PPO/BC/IL, checkpoint unchanged, no existing observed_state or
  prediction NPZ modified, no prediction writeback/fusion, no prediction
  traversability/collision/ray blocking/candidate sampling/edge validity, no
  target/ground-truth/future-observed scoring, no external source
  modification/build, no over-cost runtime promotion, and no coverage claim.
- Current next small task:
  `Stage 4A-6.5aj staged two-frame one-action lambda48 runtime smoke design
  review only`, or controlled capture-only additional saved-frame collection.
- Still not next:
  rollout, online open-ended loop, RL/PPO/BC/IL training, prediction
  writeback, observed_map prediction fusion, prediction traversability/
  collision/ray blocking, target/ground-truth or future-observed scoring,
  checkpoint changes, coverage-improvement claims, external source
  modification/build, Pareto gate implementation, runtime planner
  implementation, or over-cost runtime promotion.
- Future offline analysis commands should continue to include `--max_workers
  32`.

Stage 4A-6.5aj update / current next:

- Staged two-frame one-action lambda48 runtime smoke design review is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65aj_two_frame_one_action_runtime_design_review`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/design_stage4a65aj_two_frame_one_action_runtime_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65aj_two_frame_one_action_design_review.py`.
- This was design review only: no Isaac startup, no RGB/depth capture, no
  map_predict call, no SSCNet inference, no selected action execution, no
  two-frame runtime, no rollout/open-ended loop, no training/RL/PPO/BC/IL, no
  checkpoint or existing observed_state/prediction NPZ modification, no
  prediction writeback/fusion, no external source build, no over-cost runtime
  primary, and no coverage-improvement claim.
- Future Stage 4A-6.5ak design:
  exactly two frames, exactly two measured-only observed_state updates,
  exactly two map_predict calls, exactly one Frame 1 lambda48-selected action
  if safety gates pass, no second action, no third frame, no rollout.
- Future Stage 4A-6.5ak formula:
  `gain_exp / cost + 48 * minmax(source_occ_free)`; SC stays outside the cost
  denominator and over-cost remains prohibited for runtime primary.
- Future safety gates:
  no low-cost artifact, no historical prior basin unless action is blocked,
  finite lambda48/measured-only decisions, prediction read-only, no prediction
  traversability/collision/ray blocking/candidate sampling/edge validity, and
  no target/ground-truth/future-observed planning or scoring.
- Hardware plan:
  future command includes `--max_workers 32`, with process-pool helper workers
  keeping BLAS/OMP inner threads at `1`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aj_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aj_two_frame_one_action_design_review.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aj_two_frame_one_action_design_review_test.log`.
- Current next small task:
  `Stage 4A-6.5ak staged two-frame one-action lambda48 runtime smoke execution`
  only if explicitly requested by user.
- Still not next:
  rollout, online open-ended loop, RL/PPO/BC/IL, prediction writeback/fusion,
  prediction traversability/collision/ray blocking, target/ground-truth or
  future-observed scoring, checkpoint changes, coverage-improvement claims,
  external source modification/build, Pareto gate implementation, runtime
  planner implementation, or over-cost runtime promotion.

Stage 4A-6.5ah update / current next:

- Hardware-aware multi-scene/start saved-frame discovery or staged
  runtime-smoke design review is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ah_multiscene_or_runtime_design_review`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65ah_multiscene_or_design_review.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ah_multiscene_or_design_review.py`.
- Discovery searched existing saved artifacts under
  `/home/ubuntu22/sc_explorer_ws/outputs` and compared complete real-medium
  frame identities against Stage 4A-6.5ag's seven selected frames. It found
  `217` candidate rows: `7` already in Stage 4A-6.5ag, `55` duplicates or
  same-frame prediction variants, `127` missing prediction, `26` missing
  pose/camera, `2` synthetic/forbidden, and `0` new complete frames.
- Because there were no additional complete saved real frames, no offline
  replay was run. Stage 4A-6.5ah produced design-review-only outputs:
  `runtime_smoke_design_review.md`,
  `runtime_smoke_safety_checklist.*`,
  `future_stage4a65ai_command_sketch.md`,
  `additional_frame_discovery_inventory.*`,
  `additional_frame_duplicates.*`,
  `new_complete_frame_manifest.*`,
  `skipped_frame_candidates.*`,
  `hardware_utilization_report.*`, and final summary/recommendation files.
- Hardware use was explicitly recorded:
  `os_cpu_count=32`, requested `--max_workers 32`,
  `actual_max_workers=32`, `parallel_backend=ProcessPoolExecutor`,
  `task_count=62`, and
  `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ah_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ah_multiscene_or_design_review.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ah_multiscene_or_design_review_test.log`.
- Current next small task if runtime is desired:
  `Stage 4A-6.5ai staged one-frame lambda48 runtime smoke, no action execution`.
  Future command sketches must include `--max_workers 32`.
- Alternative next small task:
  collect additional saved frames in a controlled capture-only stage, still
  no rollout.
- Still not next:
  rollout, online open-ended loop, RL/PPO/BC/IL training, prediction
  writeback, observed_map prediction fusion, target/ground-truth or
  future-observed scoring, checkpoint changes, coverage-improvement claims,
  external source modification/build, Pareto dominance gate implementation,
  runtime planner implementation, or over-cost runtime promotion.

Hardware utilization policy for future offline replay/analysis stages:

- Use maximum available CPU parallelism by default. On this workstation,
  CPU-bound offline stages should request `--max_workers 32` and use
  `actual_max_workers=min(32, os.cpu_count() or 1)`.
- Process-pool workers should set BLAS/OMP inner threads to `1` to avoid
  oversubscription.
- Single-process numeric runs may set BLAS/OMP/torch threads to `32`.
- Every stage should log requested/actual workers, `os.cpu_count()`, parallel
  backend, task count, wall time, worker/process/thread mode, and thread
  environment variables.

Stage 4A-6.5ag update / current next:

- Stage 4A-6.5ag offline saved-frame-only multi-frame lambda48 replay over
  all available saved real `medium_three_rooms` frames is complete.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_multi_frame_lambda48_replay.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_multi_frame_lambda48_replay.py`.
- Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ag_multi_frame_lambda48_replay`.
- Discovery result:
  `20` candidate rows, `17` valid saved real-frame candidates, `7` unique
  real medium frames selected after deduplication, `10` duplicate rows, and
  incomplete/synthetic candidates skipped.
- Replay result:
  `7` frames, seeds `0..9`, `6` modes, `420` total decision rows, and `70`
  map_predict lambda48 seed-frame rows.
- map_predict lambda48 aggregate:
  same-as-measured `33/70`, distinct non-measured `35/70`, local jitter
  `2/70`, healthy non-measured `35/70`, historical prior basin `0/70`, and
  low-cost artifact `0/70`.
- Lambda32-vs-lambda48:
  branch-class agreement `62/70`, selected-child agreement `61/70`, and
  best-descendant agreement `41/70`; keep lambda48 primary because synthetic
  calibration still favors lambda48.
- Over-cost remains diagnostic-only:
  source_occ_free_over_cost historical prior-basin fraction `24/70`, low-cost
  artifact `0/70`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ag_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ag_multi_frame_lambda48_replay.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ag_multi_frame_lambda48_replay_test.log`.
- Current next small task:
  multi-scene/start saved-frame replay if available, or staged one-frame
  runtime-smoke design review only; do not execute runtime here and do not
  recommend rollout.
- Still not next:
  rollout, online open-ended loop, RL/PPO/BC/IL training, prediction
  writeback, observed_map prediction fusion, target/ground-truth or
  future-observed scoring, checkpoint changes, coverage-improvement claims,
  external source modification/build, Pareto dominance gate implementation,
  or runtime planner implementation.

Stage 4A-6.5ac update / current next:

- Saved-frame one-step lambda48 formula smoke is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ac_saved_frame_lambda48_formula_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_saved_frame_lambda48_formula_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_saved_frame_lambda48_formula_smoke.py`.
- It used only the saved Stage 4A-6.5aa synthetic frame, Oracle NPZ,
  map_predict NPZ, Stage 4A-6.5ab best-config summaries, and raw mini-RRT
  trees. No Isaac startup, new capture, map_predict rerun, SSCNet inference,
  action execution, two-frame runtime, rollout, training/RL/PPO/BC/IL,
  checkpoint edit, existing observed_state or prediction NPZ edit, prediction
  writeback, prediction collision/traversability/ray-blocking use,
  target/ground-truth scoring, external source build/edit, or
  coverage-improvement claim occurred.
- Formula:
  `gain_exp / cost + 48 * minmax(source_occ_free)` with tau `0.1` and
  occ/free thresholds `0.5/0.5`.
- Result:
  measured_only reproduced measured-frontier `5/5`; Oracle lambda48 selected
  hidden-room `5/5`; map_predict lambda48 selected hidden-room `5/5`;
  map_predict/Oracle agreement was `1.0`; map_predict lambda48 low-cost
  artifact fraction was `0.0`; median margin was `21.285778495568792`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ac_saved_frame_lambda48_formula_smoke.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ac_saved_frame_lambda48_formula_smoke_test.log`.
- Current next small task:
  saved-frame formula smoke on one real `medium_three_rooms` frame only.
- Still not next:
  runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, external source
  modification/build, Pareto gate implementation, or runtime planner
  implementation.

Stage 4A-6.5ad update / current next:

- Saved-frame lambda48 formula smoke on one real `medium_three_rooms` Frame2
  is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ad_real_frame_lambda48_formula_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_real_frame_lambda48_formula_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_real_frame_lambda48_formula_smoke.py`.
- It used saved Stage 4A-6.5p Frame2 observed_state / prediction / pose /
  camera info, saved Stage 4A-6.5y raw mini-RRT trees, and Stage 4A-6.5z/z.1
  comparison outputs. No Isaac startup, new capture, map_predict rerun,
  SSCNet inference, selected action execution, two-frame runtime, rollout,
  training/RL/PPO/BC/IL, checkpoint edit, existing observed_state or
  prediction NPZ edit, prediction writeback, prediction collision/
  traversability/ray-blocking use, target/ground-truth scoring, external
  source build/edit, or coverage-improvement claim occurred.
- Formula:
  `gain_exp / cost + 48 * minmax(source_occ_free)` with tau `0.1`, occ/free
  thresholds `0.5/0.5`, and seeds `0..9`.
- Result:
  `60` decision rows over `10` seeds and `6` modes; measured_only reproduced
  seed0 Frame2 measured reference `n0001 -> n0112`; map_predict lambda48 was
  same-as-measured in `6/10`, distinct non-measured in `4/10`, healthy
  non-measured fraction `0.4`, prior low-cost SC basin fraction `0.0`, and
  low-cost artifact fraction `0.0`.
- Lambda32 matched lambda48 at branch-class level in this replay:
  same-as-measured `6/10`, distinct non-measured `4/10`, prior basin `0.0`.
- Over-cost diagnostics reproduced the old risk shape:
  spatial prior basin flag fraction `0.5`, primary spatial-prior
  classification `3/10`, and seed0 returned to `n0127`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ad_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ad_real_frame_lambda48_formula_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ad_real_frame_lambda48_formula_smoke_test.log`.
- Current next small task:
  saved-frame formula smoke on another real medium frame only.
- Still not next:
  runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, external source
  modification/build, Pareto gate implementation, or runtime planner
  implementation.

Stage 4A-6.5ae update / current next:

- Saved-frame lambda48 formula smoke on another real `medium_three_rooms`
  frame is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ae_real_frame1_lambda48_formula_smoke`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_real_frame_lambda48_formula_smoke_another.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_real_frame_lambda48_formula_smoke_another.py`.
- It used saved Stage 4A-6.5p Frame1 observed_state / prediction / pose /
  camera info only as the evaluated frame. No fallback to Stage 4A-6.5ad
  Frame2 occurred. No Isaac startup, new capture, map_predict rerun, SSCNet
  inference, selected action execution, two-frame runtime, rollout,
  training/RL/PPO/BC/IL, checkpoint edit, existing observed_state or
  prediction NPZ edit, prediction writeback, prediction collision/
  traversability/ray-blocking use, target/ground-truth scoring, external
  source build/edit, or coverage-improvement claim occurred.
- Formula:
  `gain_exp / cost + 48 * minmax(source_occ_free)` with tau `0.1`, occ/free
  thresholds `0.5/0.5`, and seeds `0..9`.
- Result:
  `60` decision rows over `10` seeds and `6` modes; measured_only reproduced
  seed0 Frame1 measured reference `n0001 -> n0249`; map_predict lambda48 was
  same-as-measured in `8/10`, distinct non-measured in `2/10`, healthy
  non-measured fraction `0.2`, historical prior low-cost SC basin fraction
  `0.0`, and low-cost artifact fraction `0.0`.
- Lambda32 matched lambda48 at branch-class level in this replay:
  same-as-measured `8/10`, distinct non-measured `2/10`, prior basin `0.0`.
- Over-cost diagnostics on Frame1:
  same-as-measured `5/10`, distinct non-measured `5/10`, prior basin fraction
  `0.0`, low-cost artifact fraction `0.0`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ae_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ae_real_frame1_lambda48_formula_smoke.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ae_real_frame1_lambda48_formula_smoke_test.log`.
- Current next small task:
  offline saved-frame-only consolidation/design review or additional saved
  real-frame formula smoke if more real frames are needed.
- Still not next:
  runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, external source
  modification/build, Pareto gate implementation, or runtime planner
  implementation.

Stage 4A-6.5af update / current next:

- Offline saved-frame-only lambda48 consolidation / design review is complete:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65af_lambda48_saved_frame_consolidation`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/consolidate_lambda48_saved_frame_review.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_lambda48_saved_frame_consolidation.py`.
- It read only saved Stage 4A-6.5ab / 6.5ac / 6.5ad / 6.5ae CSV, JSON, and
  MD summaries. No Isaac startup, new capture, map_predict rerun, SSCNet
  inference, selected action execution, two-frame runtime, rollout,
  training/RL/PPO/BC/IL, checkpoint edit, existing observed_state or
  prediction NPZ edit, prediction writeback, prediction collision/
  traversability/ray-blocking use, target/ground-truth scoring, external
  source build/edit, runtime planner implementation, or coverage-improvement
  claim occurred.
- Consolidated lambda48 result:
  synthetic hidden-room `5/5`, oracle agreement `1.0`, low-cost artifact
  `0.0`; real aggregate same-as-measured `14/20`, healthy non-measured
  `6/20`, historical prior basin `0/20`, low-cost artifact `0/20`.
- Lambda32 matched lambda48 on real branch class (`20/20`) but synthetic
  calibration still favors lambda48 because lambda32 was only `3/5`
  hidden-room.
- Over-cost remains diagnostic-only: Frame2 had prior basin fraction `0.5`;
  Frame1 was more aggressive but this is not enough to recommend over-cost
  runtime.
- Generated required manifest, missing-field report, unified config table,
  lambda48 cross-frame summary, real aggregate, lambda32-vs-lambda48
  comparison, over-cost comparison, low-cost artifact summary, readiness
  matrix, design findings, final summary, recommended next step, and 8 PNG
  plots.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65af_lambda48_saved_frame_consolidation.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65af_lambda48_saved_frame_consolidation_test.log`.
- Current next small task:
  offline saved-frame-only multi-frame lambda48 replay over all available
  saved real `medium_three_rooms` frames.
- Still not next:
  runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, external source
  modification/build, Pareto gate implementation, or runtime planner
  implementation.

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
