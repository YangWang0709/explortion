Codex Log
Updated: 2026-06-04

Stage 4A-6.8 map_predict/lambda48 expert pilot actions:

- Re-read required project context and Git docs:
  `.project_context/CURRENT_STATE.md`, `.project_context/TODO.md`,
  `.project_context/CODEX_LOG.md`, `README.md`, `ARTIFACTS.md`,
  `ENVIRONMENT.md`, and `GIT_INITIALIZATION_REPORT.md`.
- Re-read required Stage 4A-6.6c camera-pose-fix artifacts, including
  `stage4a66c_usd_camera_pose_fix_summary.json`,
  `start_variants_interior.json`, selected validation/inspection manifests,
  `camera_info.json`, and `scene_metadata.json`.
- Re-read Stage 4A-6.7 measured-only pilot summary, dataset NPZ keys,
  dataset integrity report, and safety audit. Confirmed Stage 4A-6.7 was
  complete with `sample_count=10`, `capture_count=10`, exactly one action per
  start, no map_predict/SSCNet, no rollout, and no RL/training.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a68_map_predict_lambda48_expert_pilot.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a68_map_predict_lambda48_expert_pilot.py`.
- Ran py_compile in `env_isaaclab` for the new runner/test plus
  `scene_factory.py`, `isaac_map_predictor.py`, `offline_mini_rrt_tree.py`,
  and `sim_prediction_layer.py`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a68_py_compile.log`.
- Ran the Stage 4A-6.8 pilot on the fixed USD using the same 10 interior
  starts. Isaac started exactly once and completed all 10 start RGB/depth
  captures. `simulation_app.close()` then hung; after the capture files were
  finalized on disk, the process was terminated. The recovery run reused those
  captures and did not start Isaac again.
- The recovery run loaded SSCNet predictor once from
  `/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar`
  and ran exactly 10 map_predict/SSCNet calls, one per start, with
  `alignment_convention=code_consistent_v1` and `tau=0.1`.
- Lambda48 scoring used:
  `gain_exp / cost + 48 * minmax(source_occ_free)`.
  `source_occ_free` is the raw visible predicted-unmeasured voxel count from
  the read-only prediction layer. Prediction was not used for traversability,
  collision, ray blocking, candidate validity, edge validity, target/ground
  truth scoring, or future-observed scoring.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot`.
  Generated required dataset/report/visual package, including
  `expert_dataset.npz`, `expert_dataset_manifest.jsonl`,
  `expert_pilot_index.html`, `expert_action_flythrough.mp4`,
  per-sample RGB/depth/observed/prediction/top-candidate/action-quality
  files, dataset-level plots, prediction safety audit, expert data quality
  audit, and comparisons to Stage 4A-6.7.
- Result:
  `sample_count=10`, `capture_count=10`, `map_predict_calls=10`,
  `same_as_measured=4`, `local_jitter=4`,
  `distinct_nonmeasured_branch=2`, `no_valid_candidate=0`,
  `low_cost_artifact=0`, and `historical_prior_basin=0`.
  Stage 4A-6.8 vs Stage 4A-6.7 action changed count was `4`, mean action
  distance `0.3074937611088073m`, and mean yaw delta
  `0.6706520898196431rad`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a68_map_predict_lambda48_expert_pilot_test.log`
  reported `all_passed: true`. Dataset integrity, safety audit, prediction
  safety audit, and expert data quality audit all passed.
- Safety/negative scope stayed clean:
  no continuous rollout, no long rollout, no second action, no third frame, no
  full expert dataset, no RL/GDPO/PPO/BC/IL, no training, no replay buffer, no
  policy checkpoint, no checkpoint modification, no source USD modification,
  no fixed USD modification, no source observed_state modification, and no
  prediction writeback.
- Current next small task:
  review Stage 4A-6.7 vs Stage 4A-6.8 comparison and expert quality visuals,
  then decide whether to run a bounded two-frame pilot or start BC dataset
  design preparation. Do not jump directly to long rollout without explicit
  user approval.

Stage 4A-6.6c-usd-dependency-fix-env-corrected actions:

- Re-read `.project_context/CURRENT_STATE.md`, `.project_context/TODO.md`,
  `.project_context/CODEX_LOG.md`, and previous Stage 4A-6.6c dependency
  outputs. Confirmed the previous run was dependency-blocked, not scene
  validation success, and produced no RGB/depth, no `observed_state_final.npy`,
  no MP4, no Isaac retry, no Stage 4A-6.6d, and no Stage 4A-6.7.
- Applied the user's correction that the conda environment name in the pasted
  task was wrong. The effective conda environment for this run was
  `env_isaaclab`; report/script filenames retain the task namespace
  `env_isaacsim` for compatibility with the requested output list.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/localize_stage4a66c_usd_dependencies_env_isaacsim.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66c_usd_env_isaacsim_dependency_fix.py`.
- Ran py_compile in `env_isaaclab` for the new localization script, new test,
  and `scene_factory.py`.
- Probed `env_isaaclab`: Python
  `/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python`, Python 3.11.15;
  `isaacsim` import succeeded; `omni` namespace import succeeded; direct
  `pxr` import failed in the bare environment but succeeded when
  `omni.usd.libs` was added via `PYTHONPATH`/`LD_LIBRARY_PATH`.
- Searched the `env_isaaclab` CONDA_PREFIX, IsaacSim package/install roots,
  `/home/ubuntu22/.cache`, `/home/ubuntu22/.local/share/ov`,
  `/home/ubuntu22/.nvidia-omniverse`, and other requested IsaacSim/Omniverse
  roots for the 67 missing
  `Assets/Isaac/4.5/Isaac/...` USD dependencies using full relative path,
  exact basename, case-insensitive basename, same-stem USD variants, Isaac
  asset-root relative path, and cache/log evidence. Log/cache-only evidence
  was not treated as a trusted local asset match.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_env_isaacsim_dependency_fix`.
  Generated all 36 required blocked/search outputs, including probe reports,
  asset-root reports, asset search CSV/JSON/MD, localized package/hash
  manifests, path patch report, unresolved report, retry gate report, blocker,
  updated dependency package request, negative-scope reports, summary, and
  recommended next step.
- Result: 67/67 unique remote dependencies remain unresolved; 0 trusted exact
  local matches; 0 copied dependencies; 0 USD patches; no localized USD
  validation package; no Isaac retry. Source/staged USD SHA256 remained
  `11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b`.
- Validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_usd_env_isaacsim_dependency_fix_test.log`
  reported `all_passed: true`. Negative scope stayed clean: no procedural
  fallback, no cuboid scene, no old larger scene restoration, no rollout, no
  expert sampling/dataset, no selected action execution, no map_predict, no
  SSCNet inference, no prediction NPZ, no replay buffer, no checkpoint
  changes, and no RL/GDPO/PPO/BC/IL.
- Current blocker: `env_isaaclab` also does not contain the required local
  Isaac assets. The USD remains non-self-contained. Stage 4A-6.6d and
  Stage 4A-6.7 remain blocked until a complete dependency package is provided
  or the user allows downloading the exact missing URLs.

Stage 4A-6.6c-usd-dependency-fix actions:

- Re-read required project context files:
  `.project_context/CURRENT_STATE.md`, `.project_context/TODO.md`, and
  `.project_context/CODEX_LOG.md`, plus the previous Stage 4A-6.6c blocked
  output reports.
- Confirmed the previous Stage 4A-6.6c run was blocked by Isaac load
  (`LLVM ERROR: out of memory`) and produced no validation/inspection
  RGB/depth, no `observed_state_final.npy`, and no MP4.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/localize_stage4a66c_usd_dependencies.py`.
  The script uses the IsaacSim `omni.usd.libs` PXR bindings to inspect the
  binary staged USD, expands references/payloads/sublayers/asset attributes,
  searches local roots with allowed USD/mesh/image extensions, and only patches
  the staged USD if every missing dependency has a unique trusted exact local
  candidate.
- Ran py_compile and then the dependency localization audit with explicit
  `PYTHONPATH`/`LD_LIBRARY_PATH` for the IsaacSim USD bindings. The search used
  `rg --files --hidden --no-ignore`, indexed 672444 allowed-extension files,
  and covered the requested roots or their parent-root equivalents.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_home_like_scene_validation`.
  New outputs include `dependency_localization_input_summary.*`,
  `usd_dependency_expanded_report.*`, `missing_dependency_table.*`,
  `missing_dependency_unique_table.*`, `remote_dependency_table.*`,
  `local_dependency_candidates.*`, `dependency_localization_patch_report.*`,
  `dependency_localization_summary.*`, and `dependency_package_request.*`.
- Result:
  99 reference occurrences, 67 unique remote Omniverse/S3 dependencies,
  0 sublayers, 0 payloads, 0 material/texture asset paths, 0 absolute paths,
  0 local candidates, 0 copied files, and 0 staged USD patches. Source and
  staged SHA256 remained
  `11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b`.
- Because dependencies are still missing, Isaac retry was not allowed and was
  not executed. Stage 4A-6.6d and Stage 4A-6.7 remain blocked. Next faithful
  step is to provide the requested dependency package or a lighter fully local
  USD; no procedural fallback, random asset download, rollout, expert sampling,
  map_predict, SSCNet inference, prediction NPZ, replay buffer, checkpoint
  changes, or RL/GDPO/PPO/BC/IL was performed.

Stage 4A-6.6a larger_complex_scene_v1 scene complexity audit actions:

- Re-read local project context:
  `.project_context/CURRENT_STATE.md`, `.project_context/TODO.md`, and
  `.project_context/CODEX_LOG.md`, plus the user-provided Stage 4A-6.6a spec.
- Confirmed Stage 4A-6.6 was complete and that Stage 4A-6.6a was offline
  scene complexity audit only: no Isaac startup, no RGB/depth capture, no
  selected action execution, no rollout, no open-ended loop, no formal expert
  sampling, no expert dataset, no transitions.jsonl, no map_predict, no
  SSCNet inference, no prediction NPZ, no prediction writeback/fusion, no
  checkpoint changes, no replay buffer, and no RL/GDPO/PPO/BC/IL.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/audit_stage4a66a_scene_complexity.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66a_scene_complexity_audit.py`.
- Ran Python compile checks for both Stage 4A-6.6a scripts in `env_isaaclab`.
- Ran the offline audit with `--max_workers 32`, `--save_viz`, `--no_isaac`,
  `--no_capture`, `--no_rollout`, `--no_formal_expert_sampling`,
  `--no_map_predict`, and `--no_rl_gdpo`.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66a_scene_complexity_audit`
  with 92 top-level audit files, including required JSON/MD/CSV reports and
  plots.
- Audit result:
  all nine categories passed: scale, topology, starts, fixed views,
  observed_state, frontier/reachability, obstacle/occlusion, expert usability,
  and safety/negative-scope. Hard blockers: none.
- Decision:
  `scene_complexity_audit_passed: true`,
  `scene_ready_for_formal_expert_sampling_pilot: true`,
  `formal_expert_sampling_ready_full_dataset: false`.
- Main warnings:
  close topology among a few starts despite Euclidean spread; fixed views do
  not directly name `corridor_east_spur` and `room_j`; fixed-view
  observed_ratio is intentionally low; measured-only fixed views create
  multiple observed-free components; spur rooms have higher obstacle density;
  Stage 4A-6.7 should start measured-only before any lambda48 read-only
  map_predict pilot.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66a_scene_complexity_audit_test.log`
  reported `all_passed: true`. The test verified required outputs, required
  plots, max worker reporting, no forbidden rollout/expert/map_predict/RL
  artifacts, and unchanged Stage 4A-6.6 input hashes including
  `observed_state_final.npy`.
- Recommended next faithful step:
  Stage 4A-6.7 bounded formal expert sampling pilot design/execution,
  measured-only first, small/qualified start subset or all qualified starts,
  not full dataset. No rollout scaling, no full expert dataset, and no
  RL/GDPO/PPO/BC/IL yet. Long-term GDPO remains future direction only.

Stage 4A-6.6 larger_complex_scene_v1 construction and validation actions:

- Re-read local project context:
  `.project_context/CURRENT_STATE.md`, `.project_context/TODO.md`, and
  `.project_context/CODEX_LOG.md`, plus the user-provided Stage 4A-6.6 spec.
- Confirmed Stage 4A-6.5av completed cleanly and that the next required gate
  is larger scene construction/validation followed by Stage 4A-6.6a scene
  complexity audit. Formal expert sampling remains blocked before 6.6a.
- Implemented `build_larger_complex_scene_v1` in:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py`.
  The scene is deterministic for `scene_seed=0`, uses x/y bounds `[-12, 12]`,
  z bounds `[0, 3]`, and has 10 rooms, 7 corridors, 21 openings, 69 wall
  cuboids, 52 obstacle cuboids, 9 start variants, 14 fixed validation camera
  poses, narrow passages, loop closures, dead-end branches, and topology graph
  cycle rank 5.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/build_stage4a66_larger_complex_scene_v1.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66_larger_complex_scene_v1.py`.
- Ran Python compile checks for:
  `scene_factory.py`,
  `build_stage4a66_larger_complex_scene_v1.py`, and
  `test_stage4a66_larger_complex_scene_v1.py`.
- Ran the Stage 4A-6.6 validation in `env_isaaclab` with the project headless
  Vulkan settings:
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`,
  `__GLX_VENDOR_LIBRARY_NAME=nvidia`, and unset
  `DISPLAY`, `WAYLAND_DISPLAY`, `XAUTHORITY`, and `GNOME_SETUP_DISPLAY`.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66_larger_complex_scene_v1_validation`
- Runtime result:
  exactly one successful Isaac headless startup in the clean validation run,
  scene loaded, 14 fixed validation poses captured, 14/14 RGB views nonblank,
  and 14/14 depth views had finite positive depth.
- Measured-only observed_state result:
  final shape `(240, 240, 30)`, observed_ratio `0.09458275462962963`,
  observed_count `163439`, free_count `154672`, occupied_count `8767`,
  unknown_count `1564561`, invalid_label_count `0`, and no prediction input or
  writeback.
- Generated required metadata, inventories, topology graph, start variants,
  validation pose manifests, preliminary complexity metrics, fixed capture
  summaries, observed_state summaries, no-rollout/no-expert/no-map_predict/
  no-RL reports, audit bundle manifest, future 6.6a command sketch, and
  topdown/capture/observed/connectivity/checklist visualizations.
- Ran:
  `python sim_explorer/test_stage4a66_larger_complex_scene_v1.py --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66_larger_complex_scene_v1_validation`
  in `env_isaaclab`; it reported `all_passed: true`.
- Negative scope held:
  no rollout, no open-ended loop, no selected action execution, no formal
  expert sampling, no expert dataset, no map_predict call, no SSCNet inference,
  no prediction NPZ, no prediction fusion/writeback, no target/ground-truth/
  future-observed planning/scoring, no replay buffer, no policy checkpoint, no
  checkpoint load except hash audit, no external active_3d_planning changes,
  and no RL/GDPO/PPO/BC/IL.
- Outcome:
  `larger_complex_scene_v1_constructed_and_fixed_capture_validated`. This is
  not a Stage 4A-6.6a scene complexity audit pass and does not make formal
  expert sampling ready. Recommended next faithful step: Stage 4A-6.6a scene
  complexity audit using the generated audit input bundle.

Stage 4A-6.5at start_corridor seed0/seed1 review and next-start design actions:

- Re-read local project context:
  `.project_context/CURRENT_STATE.md`, `.project_context/CODEX_LOG.md`, and
  `.project_context/TODO.md`, plus the user-provided Stage 4A-6.5at spec.
- Confirmed Stage 4A-6.5at is offline diagnosis/design only: no Isaac
  startup, no RGB/depth capture, no map_predict, no SSCNet inference, no
  selected action execution, no two-frame runtime execution, no rollout, no
  training/RL/GDPO/PPO/BC/IL, no checkpoint changes, and no existing
  observed_state or prediction NPZ modification.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/review_stage4a65at_start_corridor_seed01_next_start_design.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65at_start_corridor_seed01_next_start_design.py`.
- Ran py_compile:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65at_py_compile.log`.
- Ran the offline review:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65at_start_corridor_seed01_review_next_start_design.log`.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65at_start_corridor_seed01_review_next_start_design`
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65at_start_corridor_seed01_review_next_start_design_test.log`
  reported `all_passed: true`, with 76 required files and 10 plots present.
- Reverified Stage 4A-6.5aq tree_seed `0` and Stage 4A-6.5as tree_seed `1`:
  both used `start_corridor`, pose `[0.0, -4.45, 1.2]`, yaw
  `1.5707963267948966`, exactly two frames, exactly two map_predict calls,
  exactly one selected action, no second action, no third frame, and no
  rollout, with formula `gain_exp / cost + 48 * minmax(source_occ_free)`.
- aq/as comparison:
  aq Frame1/Frame2 lambda48 stayed `same_as_measured`; as Frame1/Frame2
  lambda48 was `distinct_nonmeasured_branch`. Frame1 selected/best deltas were
  `0.2m` / `1.6881943016134136m`; Frame2 selected/best deltas were
  `0.458257569495584m` / `2.4103941586387903m`. Action pose/yaw deltas were
  `0.20000000000000018m` / `2.7504672066207645rad`.
- observed_state/map_predict:
  aq observed_ratio delta `0.012087962962962964`, newly observed `5222`; as
  observed_ratio delta `0.006354166666666667`, newly observed `2745`;
  map_predict Frame1 exact match `61152 / 49164`, Frame2 aq `52988 / 43828`
  vs as `47866 / 41937`, both `code_consistent_v1`, no explosion/collapse.
- lambda32/lambda48:
  Frame1 matched selected/best for both seeds. Frame2 aq matched selected
  child only; as lambda48 diverged from lambda32/measured in a healthy
  diagnostic way.
- No low-cost artifact, no historical prior basin, no prediction
  writeback/fusion, no prediction traversability/collision/ray blocking, no
  candidate sampling/edge-validity use, no target/ground-truth/future-observed
  scoring, no over-cost runtime promotion, and no coverage-improvement claim.
- Combined outcome:
  `healthy_distinct_seed1_after_conservative_seed0`, also interpretable as
  start_corridor seed-sensitive but clean. start_corridor tree_seed `2` was
  not executed and is not automatically next. Evidence is still not
  rollout-ready.
- Selected future Stage 4A-6.5au:
  `start_room_b`, pose `[2.75, -2.55, 1.2]`, yaw `2.7052603405912112`, source
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_b_empty_astar/scene_metadata.json`,
  future tree_seed `0`, exactly two frames, exactly two map_predict calls if
  action executes, exactly one selected action, no second action, no third
  frame, no rollout, formula `gain_exp / cost + 48 * minmax(source_occ_free)`,
  and `--max_workers 32`. The future command sketch begins
  `DO NOT RUN IN STAGE 4A-6.5at.` and was not executed.
- Long-term note:
  NVIDIA GDPO-style multi-reward decoupled policy optimization remains a
  future direction only. RL/GDPO/PPO/BC/IL remains explicitly not next until
  bounded repeats and rollout data are ready.

Stage 4A-6.5as start_corridor tree_seed=1 bounded smoke actions:

- Re-read local project context:
  `.project_context/CURRENT_STATE.md`, `.project_context/CODEX_LOG.md`, and
  `.project_context/TODO.md`.
- Confirmed Stage 4A-6.5aq was complete at `start_corridor`, tree_seed `0`,
  and Stage 4A-6.5ar selected Stage 4A-6.5as as the next real bounded runtime
  smoke: same scene/start, current tree_seed `1`, exactly two frames, exactly
  two map_predict calls if action executes, exactly one selected action, no
  second action, no third frame, no rollout, formula
  `gain_exp / cost + 48 * minmax(source_occ_free)`.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65as_start_corridor_seed1_bounded_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65as_start_corridor_seed1_bounded_smoke.py`.
- Ran py_compile:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65as_py_compile.log`.
- Ran Stage 4A-6.5as runtime with the validated headless NVIDIA/Vulkan
  environment:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65as_start_corridor_tree_seed1_bounded_smoke.log`.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65as_start_corridor_tree_seed1_bounded_smoke`
- Runtime result:
  exactly one Isaac startup in the clean run, exactly two frames, exactly two
  map_predict calls, exactly one selected action execution, no second action,
  no third frame, and no rollout.
- Start/repeat:
  `medium_three_rooms`, scene seed `0`, start variant `start_corridor`, pose
  `[0.0, -4.45, 1.2]`, yaw `1.5707963267948966`, pose source
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_corridor_empty_astar/scene_metadata.json`,
  repeat variant `alternate_start_corridor_tree_seed1`, reference Stage
  4A-6.5aq tree_seed `0`, current tree_seed `1`.
- Frame 1 result:
  measured-only shadow `n0018 -> n0022`, lambda48 primary `n0001 -> n0135`,
  lambda32 shadow `n0001 -> n0135`, classification
  `distinct_nonmeasured_branch`, low-cost artifact `false`, historical prior
  basin `false`, all gates passed, action pose
  `[0.15000000000000036, -3.9499999999999997, 1.2]`, yaw
  `-0.29145679447786677`.
- Frame 2 result:
  measured-only shadow `n0036 -> n0106`, lambda48 diagnostic
  `n0008 -> n0137`, lambda32 shadow `n0036 -> n0106`, classification
  `distinct_nonmeasured_branch`, low-cost artifact `false`, historical prior
  basin `false`.
- Repeat comparison vs Stage 4A-6.5aq:
  Frame1 selected/best deltas `0.2m` / `1.6881943016134136m`, Frame2
  selected/best deltas `0.458257569495584m` / `2.4103941586387903m`,
  action pose/yaw deltas `0.20000000000000018m` /
  `2.7504672066207645rad`, observed_ratio delta difference
  `-0.005733796296296297`, map_predict OCC+FREE deltas `0` / `-1891`.
- observed_state:
  observed_ratio `0.03149537037037037 -> 0.037849537037037036`, delta
  `0.006354166666666667`, newly observed `2745`, unknown->free `2277`,
  unknown->occupied `468`, occupied->free `0`, invalid labels `0`.
- map_predict stability:
  Frame1 valid/OCC+FREE `61152 / 49164`, Frame2 `47866 / 41937`, density
  ratio `0.8530021967293141`, no explosion/collapse, both
  `code_consistent_v1`.
- lambda32/lambda48:
  Frame1 selected/best matched; Frame2 diverged, with lambda48
  `n0008 -> n0137` and lambda32 `n0036 -> n0106`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65as_start_corridor_tree_seed1_bounded_smoke_test.log`
  reported `all_passed: true`.
- Outcome:
  `spatially_consistent_healthy_repeat`. Prediction remained read-only and
  information-gain-only. No prediction writeback/fusion,
  traversability/collision/ray blocking, candidate sampling, edge-validity
  use, target/ground-truth/future-observed scoring, checkpoint change,
  external source build, over-cost runtime primary, coverage-improvement
  claim, rollout, open-ended loop, RL/GDPO/PPO/BC/IL, or policy artifact was
  performed.
- Hardware:
  `os_cpu_count=32`, requested/actual workers `32/32`,
  OMP/OPENBLAS/MKL/NUMEXPR/VECLIB threads `1/1/1/1/1`, GPU
  `NVIDIA GeForce RTX 5080`, total runtime wall time
  `37.23983290199976s`.
- Recommendation:
  next small task is Stage 4A-6.5at start_corridor seed0/seed1
  repeat-comparison diagnosis and next-start design only, not rollout.
- Long-term note:
  NVIDIA GDPO-style multi-reward decoupled policy optimization remains a
  future direction only. RL/GDPO/PPO/BC/IL remains explicitly not next until
  bounded repeats and rollout data are ready.

Stage 4A-6.5ar alternate-start post-action diagnosis actions:

- Re-read local project context:
  `.project_context/CURRENT_STATE.md`, `.project_context/CODEX_LOG.md`, and
  `.project_context/TODO.md`.
- Confirmed Stage 4A-6.5aq was complete and that Stage 4A-6.5ar is
  offline diagnosis/design only: no Isaac startup, no capture, no
  map_predict, no SSCNet inference, no selected action execution, no
  two-frame runtime execution, no rollout, no training/RL/GDPO/PPO/BC/IL, no
  checkpoint changes, and no existing observed_state or prediction NPZ
  modification.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_stage4a65ar_alternate_start_post_action.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ar_alternate_start_post_action.py`.
- Ran py_compile:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ar_py_compile.log`.
- Ran the offline diagnosis:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ar_alternate_start_post_action_diagnosis.log`.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ar_alternate_start_post_action_diagnosis`
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ar_alternate_start_post_action_diagnosis_test.log`
  reported `passed: true`, with 68 required files and 10 plots present.
- Reverified Stage 4A-6.5aq sequence:
  exactly two frames, exactly two map_predict calls, exactly one selected
  action, no second action, no third frame, and no rollout.
- Reverified alternate start:
  `start_corridor`, pose `[0.0, -4.45, 1.2]`, yaw
  `1.5707963267948966`, matched Stage 4A-6.5ap design and metadata.
- Reverified action pose:
  `[-0.04999999999999982, -3.9499999999999997, 1.2]`, yaw
  `-3.0419240010986313`, matched Frame2 pose and the Frame1 lambda48
  selected child XY.
- observed_state diagnosis:
  observed_ratio `0.03149537037037037 -> 0.043583333333333335`, delta
  `0.012087962962962964`, newly observed `5222`, unknown->free `4876`,
  unknown->occupied `346`, occupied->free `0`, invalid labels `0`;
  measured-only update remained clean.
- map_predict diagnosis:
  Frame1 valid/OCC+FREE `61152 / 49164`, Frame2 `52988 / 43828`,
  density ratio `0.8914652998128713`, no explosion/collapse, both
  `code_consistent_v1`.
- Tree/branch diagnosis:
  Frame1 lambda48 matched measured-only exactly (`n0001 -> n0104`).
  Frame2 lambda48 shared selected child `n0001`, differed at best descendant
  (`n0127` vs measured `n0126`), and still classified `same_as_measured`.
  lambda32/lambda48 matched selected/best on Frame1 and matched selected
  child only on Frame2.
- No low-cost artifact and no historical prior basin were found. Prediction
  remained read-only/information-gain-only, with no prediction writeback/
  fusion, traversability/collision/ray blocking, candidate sampling,
  edge-validity use, target/ground-truth/future-observed scoring, over-cost
  runtime primary, or coverage-improvement claim.
- Outcome:
  `clean_same_as_measured`, conservative but safe. This is not coverage
  improvement evidence, not rollout-ready, and not RL/GDPO-ready.
- Selected future Stage 4A-6.5as:
  start_corridor tree_seed `1` bounded repeat-safety smoke, exactly two
  frames, exactly two map_predict calls if action executes, exactly one
  selected action, no second action, no third frame, no rollout, formula
  `gain_exp / cost + 48 * minmax(source_occ_free)`, and `--max_workers 32`.
  The future command sketch begins `DO NOT RUN IN STAGE 4A-6.5ar.` and was
  not executed.
- Long-term note:
  NVIDIA GDPO-style multi-reward decoupled policy optimization remains a
  future direction only. RL/GDPO/PPO/BC/IL remains explicitly not next until
  bounded repeats and rollout data are ready.

Stage 4A-6.5aq alternate-start bounded smoke actions:

- Re-read local project context:
  `.project_context/CURRENT_STATE.md`, `.project_context/CODEX_LOG.md`, and
  `.project_context/TODO.md`.
- Confirmed Stage 4A-6.5ap selected future alternate start `start_corridor`
  and that the current bounded runtime task is Stage 4A-6.5aq with
  `tree_seed=0`, no rollout, no second action, and no third frame.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65aq_alternate_start_bounded_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65aq_alternate_start_bounded_smoke.py`.
- Ran py_compile:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aq_py_compile.log`.
- Ran Stage 4A-6.5aq runtime with the validated headless NVIDIA/Vulkan
  environment:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aq_alternate_start_corridor_bounded_smoke.log`.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65aq_alternate_start_corridor_bounded_smoke`
- Runtime result:
  exactly one Isaac startup in the clean run, exactly two frames, exactly two
  map_predict calls, exactly one selected action execution, no second action,
  no third frame, and no rollout.
- Alternate start:
  `start_corridor`, pose `[0.0, -4.45, 1.2]`, yaw
  `1.5707963267948966`, source
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_corridor_empty_astar/scene_metadata.json`,
  distance `4.654299087940095m` from canonical start; pose matched Stage
  4A-6.5ap design and metadata.
- Frame 1 result:
  measured-only shadow `n0001 -> n0104`, lambda48 primary
  `n0001 -> n0104`, lambda32 shadow `n0001 -> n0104`,
  classification `same_as_measured`, low-cost artifact `false`, historical
  prior basin `false`, all pre-action gates passed, action pose
  `[-0.04999999999999982, -3.9499999999999997, 1.2]`, yaw
  `-3.0419240010986313`.
- Frame 2 result:
  measured-only shadow `n0001 -> n0126`, lambda48 diagnostic
  `n0001 -> n0127`, lambda32 shadow `n0001 -> n0126`,
  classification `same_as_measured`, low-cost artifact `false`, historical
  prior basin `false`.
- Observed_state delta:
  observed_ratio `0.03149537037037037 -> 0.043583333333333335`, delta
  `0.012087962962962964`, newly observed `5222`, unknown->free `4876`,
  unknown->occupied `346`, occupied->free `0`, invalid labels `0`.
- map_predict stability:
  Frame 1 valid/OCC+FREE `61152 / 49164`, Frame 2 valid/OCC+FREE
  `52988 / 43828`, density ratio `0.8914652998128713`, no
  explosion/collapse, both `code_consistent_v1`.
- lambda32/lambda48:
  Frame 1 selected/best matched; Frame 2 selected child matched but best
  descendant differed (`n0126` vs `n0127`), with both still
  `same_as_measured`.
- Comparison to Stage 4A-6.5ap design passed. Comparison to canonical-start
  seed0/1/2 was recorded as context only because exact branch/position match
  is not required after changing the start pose.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aq_alternate_start_corridor_bounded_smoke_test.log`
  reported `all_passed: true`.
- Outcome:
  `clean_same_as_measured`. Prediction remained read-only and
  information-gain-only. No prediction writeback/fusion,
  traversability/collision/ray blocking, candidate sampling, edge-validity
  use, target/ground-truth/future-observed scoring, checkpoint change,
  external source build, over-cost runtime primary, coverage-improvement
  claim, rollout, open-ended loop, RL/GDPO/PPO/BC/IL, or policy artifact was
  performed.
- Hardware:
  `os_cpu_count=32`, requested/actual workers `32/32`,
  OMP/OPENBLAS/MKL/NUMEXPR/VECLIB threads `1/1/1/1/1`, GPU
  `NVIDIA GeForce RTX 5080`, total runtime wall time
  `39.62141864500154s`.
- Recommendation:
  next small task is Stage 4A-6.5ar alternate-start post-action/two-frame
  diagnosis and repeat-safety review only, not rollout.
- Long-term note:
  NVIDIA GDPO-style multi-reward decoupled policy optimization remains a
  future direction only; RL/GDPO/PPO/BC/IL remains explicitly not next until
  bounded repeats and rollout data are ready.

Stage 4A-6.5ao bounded repeat-safety smoke actions:

- Re-read local project context:
  `.project_context/CURRENT_STATE.md`, `.project_context/CODEX_LOG.md`, and
  `.project_context/TODO.md`.
- Confirmed Stage 4A-6.5ak tree_seed `0`, Stage 4A-6.5am tree_seed `1`, and
  Stage 4A-6.5an repeat-comparison/next-design are complete and that the next
  bounded runtime task is Stage 4A-6.5ao with same scene/start and
  `tree_seed=2`.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65ao_bounded_repeat_safety_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ao_bounded_repeat_safety_smoke.py`.
- Ran py_compile:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ao_py_compile.log`.
- First runtime launch attempt failed at Isaac GLX initialization before any
  frame capture or bounded smoke output. Re-ran the clean smoke with the
  previously validated headless NVIDIA/Vulkan environment
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`,
  `__GLX_VENDOR_LIBRARY_NAME=nvidia`, and unset display variables.
- Ran Stage 4A-6.5ao runtime:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ao_bounded_repeat_safety_smoke_tree_seed2.log`.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ao_bounded_repeat_safety_smoke_tree_seed2`
- Runtime result:
  exactly one Isaac startup in the clean run, exactly two frames, exactly two
  map_predict calls, exactly one selected action execution, no second action,
  no third frame, and no rollout.
- Frame 1 result:
  measured-only shadow `n0001 -> n0248`, lambda48 primary `n0001 -> n0248`,
  lambda32 shadow `n0001 -> n0248`, classification `same_as_measured`,
  low-cost artifact `false`, historical prior basin `false`, all pre-action
  gates passed, action pose `[-4.25, -4.35, 1.2]`, yaw `2.2142974355881817`.
- Frame 2 result:
  measured-only shadow `n0126 -> n0186`, lambda48 diagnostic
  `n0003 -> n0227`, lambda32 shadow `n0003 -> n0227`, classification
  `distinct_nonmeasured_branch`, low-cost artifact `false`, historical prior
  basin `false`.
- Observed_state delta:
  observed_ratio `0.0425462962962963 -> 0.05556944444444444`, delta
  `0.013023148148148148`, newly observed `5626`, unknown->free `5078`,
  unknown->occupied `548`, occupied->free `0`, invalid labels `0`.
- map_predict stability:
  Frame 1 valid/OCC+FREE `57382 / 40328`, Frame 2 valid/OCC+FREE
  `32890 / 24936`, density ratio `0.6183296964887919`, no
  explosion/collapse, both `code_consistent_v1`.
- Repeat comparison:
  Frame 1 selected deltas vs seed0/seed1 `0.223606797749979m` /
  `0.41231056256176607m`; Frame 2 selected deltas vs seed0/seed1 `0.5m` /
  `0.632455532033676m`; Frame 2 best-descendant deltas vs seed0/seed1
  `4.036087214122113m` / `1.2083045973594573m`.
- Validation passed:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ao_bounded_repeat_safety_smoke_tree_seed2_test.log`
  reported `all_passed: true`.
- Outcome:
  `spatially_consistent_healthy_repeat`, spatially consistent with seed1 and
  seed-sensitive but clean. Still not rollout-ready.
- Recommendation:
  next small task is Stage 4A-6.5ap repeat-comparison review /
  alternate-start design only, not rollout.
- No rollout, open-ended loop, RL/GDPO/PPO/BC/IL, prediction writeback/fusion,
  prediction traversability/collision/ray blocking/candidate-sampling/
  edge-validity use, target/ground-truth/future-observed scoring, checkpoint
  changes, external source build, runtime planner implementation, over-cost
  runtime promotion, or coverage-improvement claim was performed.

Stage 2A actions:

- Read local context and existing SSC network notes.
- Checked `env_isaaclab`, Python, torch, CUDA, checkpoints, and NYU real `.npz` data.
- Saved pre-change git status/diff to:
  `/home/ubuntu22/sc_explorer_ws/logs/stage2a_git_status_before.txt`
  `/home/ubuntu22/sc_explorer_ws/logs/stage2a_changes_before.patch`
- Inspected `test.py`, `train.py`, `models/SSCNet.py`, `models/__init__.py`, `dataloaders/dataloader.py`, `dataloaders/__init__.py`, `utils/ssc_metrics.py`, and `config.py`.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/offline_infer_npz.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/prediction_layer.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_offline_prediction_layer.py`
- Ran `py_compile` in `env_isaaclab`.
- Ran single-sample offline inference with `--save_probs`.
- Ran PredictionLayer smoke test.
- Ran offline inference on first 5 NYUtest samples only.
- Updated `ssc_network_training_notes.md`.

No planner, RL, PPO, imitation learning, Unreal, AirSim, retraining, dataset download, or observed_map write was performed.

Stage 2B strict paper-faithful actions:

- Re-read local context and Stage 2A notes.
- Inspected dataloader TSDF/position conventions.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/sc_explorer_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/run_paper_expert_offline.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_paper_expert.py`
- Disabled earlier prototype entry files that used a target-label mock observed map:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/sc_explorer_expert.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/run_sc_explorer_expert_offline.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_sc_explorer_expert.py`
- Built measured S only from sensor-derived `tsdf_lr` and/or `position`.
- Defined P as `PredictionLayer.confidence >= tau` and not measured.
- Implemented non-blocking ray casting by default and optional `sc_blocking`.
- Implemented paper gains `I_exp`, `I_sc`, `I_hybrid`, `I_occ`, and `I_conf`.
- Implemented approximate position/yaw time cost and per-candidate utilities.
- Ran py_compile checks.
- Ran strict paper expert smoke test:
  `/home/ubuntu22/sc_explorer_ws/logs/stage2b_paper_expert_test.log`
- Ran single-sample strict paper expert scorer:
  `/home/ubuntu22/sc_explorer_ws/logs/stage2b_paper_expert_single.log`
- Ran batch5 strict paper expert scorer:
  `/home/ubuntu22/sc_explorer_ws/logs/stage2b_paper_expert_batch5.log`
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/paper_expert`
  `/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_batch5`
- Updated `ssc_network_training_notes.md` and local context.

No target_lr/target_hr expert-scoring use, RL, PPO, imitation learning training, Unreal, AirSim, retraining, robot execution, planner integration, or observed_map write was performed.

Stage 2C paper expert dataset actions:

- Re-read local context and Stage 2B notes.
- Confirmed Stage 2B is complete and the next step is dataset-format
  conversion from strict expert outputs, not imitation-learning training.
- Inspected:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/sc_explorer_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/run_paper_expert_offline.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/offline_infer_npz.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/prediction_layer.py`
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/generate_paper_expert_dataset.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_paper_expert_dataset.py`
- The generator writes:
  `/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/samples`
  `/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/predictions`
  `/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/logs`
  `/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/manifest.jsonl`
  `/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/metadata.json`
- The generator can optionally call `offline_infer_npz.run_inference` for
  missing predictions when `--generate_missing_predictions` is passed.
- Without `--generate_missing_predictions`, it prioritizes samples with
  matching existing prediction files so the existing Stage 2A batch5
  predictions can be reused for smoke runs.
- Ran py_compile checks for the new Stage 2C scripts.
- Ran Stage 2C smoke generation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage2c_dataset_smoke.log`
- Smoke output:
  `/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke`
- Smoke result:
  5 total samples, 5 ok, 0 failed, 5 per-sample expert `.npz` files,
  `manifest.jsonl`, `metadata.json`, and `combined_smoke.npz`.
- Ran Stage 2C dataset validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage2c_dataset_test.log`
- Validation result:
  manifest records 5, ok samples 5, failed samples 0, first sample shape
  `N=16 D=15`, forbidden target fields check passed.
- Updated `ssc_network_training_notes.md` and local context.

No target_lr/target_hr expert-scoring use, target-derived features, RL, PPO,
imitation-learning training, Unreal, AirSim, retraining, robot execution,
planner integration, or observed_map write was performed in Stage 2C.

Stage 3A IL Dataset/DataLoader actions:

- Re-read local context and `ssc_network_training_notes.md`.
- Confirmed Stage 2C is complete and Stage 3A is only Dataset/DataLoader,
  feature stats, policy skeleton, and forward-only smoke tests.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/__init__.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/paper_expert_dataset.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/policy.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/train_bc.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/test_dataset.py`
- `PaperExpertDataset` reads only Stage 2C expert sample `.npz` files from
  manifest ok rows. It does not open original `sample_npz` or `prediction_npz`.
- Dataset checks:
  expert action range, `valid_mask[expert_action]`, finite
  `candidate_features`, finite `expert_scores`, `strict_no_target_lr`, and no
  forbidden `target_lr`, `target_hr`, `gt`, or `ground_truth` fields.
- `collate_paper_expert_batch` stacks fixed candidate counts and pads variable
  candidate counts defensively.
- `compute_feature_stats` uses only valid candidates and clamps std to `1e-6`.
- `CandidateMLPPolicy` is a shared MLP over candidate features and masks invalid
  candidate logits to `-1e9`.
- `train_bc.py` prints a disabled-training message unless `--dry_run` is used.
  In dry-run mode it loads one batch, runs policy forward, computes CE loss,
  and exits without optimizer step or model save.
- Ran py_compile checks for the new Stage 3A files.
- Ran dataset smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/stage3a_il_dataset_test.log`
- Dataset smoke result:
  dataset size 5, first `candidate_features` shape `(16, 15)`, batch shape
  `(2, 16, 15)`, feature stats shape `(15,)`, logits shape `(2, 16)`,
  CE loss `0.000060`, forbidden target fields none, optimizer step no.
- Ran BC dry-run:
  `/home/ubuntu22/sc_explorer_ws/logs/stage3a_bc_dry_run.log`
- BC dry-run result:
  dataset size 5, `B,N,D: 2,16,15`, expert actions `[0, 0]`, logits shape
  `(2, 16)`, loss `0.165347`, optimizer step no, model saved no.
- Saved feature stats:
  `/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/feature_stats.npz`
- Updated `ssc_network_training_notes.md` and local context.

No behavior cloning training, optimizer step, model save, RL, PPO, Unreal,
AirSim, SSCNet retraining, robot execution, planner integration, target_lr /
target_hr loading, or observed_map write was performed in Stage 3A.

Stage 4A-1 Isaac depth observation smoke actions:

- Re-read local context and `ssc_network_training_notes.md`.
- Confirmed SSCNet full training, PredictionLayer, paper expert scorer, paper
  expert dataset, and IL Dataset/DataLoader smoke tests are complete.
- Confirmed the current task is simulator continuous exploration sensing, not
  NYU static rollout and not RL/BC training.
- Checked `env_isaaclab`:
  Python 3.11.15, torch 2.7.0+cu128, isaaclab 0.54.3, isaacsim installed,
  omni import OK, direct pre-launch pxr import failed.
- Found Isaac Lab repo:
  `/home/ubuntu22/IsaacLab`
- Found Isaac Lab commit:
  `090aed18163b2194d5551c7919f7539283677743`
- Found Isaac Sim pip install:
  `/home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim`
- Found Isaac Sim package metadata:
  `isaacsim 5.1.0.0`; VERSION file
  `5.1.0-rc.19+release.26219.9c81211b.gl`.
- Ran official empty scene smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/isaac_empty_scene_smoke.log`
- Ran official USD camera/depth smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/isaac_sensor_smoke.log`
- Working headless camera environment:
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`,
  `__GLX_VENDOR_LIBRARY_NAME=nvidia`, and unset
  `DISPLAY WAYLAND_DISPLAY XAUTHORITY GNOME_SETUP_DISPLAY`.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/README.md`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/minimal_depth_scene.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_depth_to_voxel.py`
- Ran pure Python depth_to_voxel tests:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a1_depth_to_voxel_test.log`
- Ran minimal depth scene:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a1_minimal_depth_scene.log`
- Saved depth outputs to:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke`
- Ran depth-to-voxel conversion:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a1_depth_to_voxel.log`
- Observed voxel map result:
  shape `(80, 80, 30)`, `unknown_count=143335`, `free_count=44435`,
  `occupied_count=4230`, `observed_ratio=0.2534635416666667`.
- Updated `simulator_notes.md`, `ssc_network_training_notes.md`, and project
  context files.

No RL, PPO, behavior cloning training, imitation-learning training, SSCNet
inference, PredictionLayer integration, expert scoring, AirSim, Unreal, target
label use, ground-truth map use, or prediction write into observed_map was
performed in Stage 4A-1.

Stage 4A-1-viz visualization actions:

- Re-read `simulator_notes.md`, `ssc_network_training_notes.md`, and project
  context.
- Confirmed Stage 4A-1 is complete and this step is visualization only.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_observed_map.py`
- The visualization script reads existing Stage 4A-1 `depth_*.npy`,
  `observed_state_step*.npy`, pose JSON files, `camera_info.json`, and
  `scene_metadata.json`.
- The script does not modify observed_state and records
  `observed_state_modified: false`, `prediction_used: false`,
  `expert_used: false`, and `rl_or_training_used: false` in
  `viz_summary.json`.
- Ran visualization command in `env_isaaclab`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a1_visualize_observed_map.log`
- Generated output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke_viz`
- Generated:
  `depth_000.png`, `depth_001.png`, `depth_002.png`, `depth_grid.png`,
  `observed_topdown_step0.png`, `observed_topdown_step1.png`,
  `observed_topdown_step2.png`, `observed_topdown_compare.png`,
  `occupied_voxels_3d_step2.png`, `free_occupied_voxels_3d_step2.png`,
  `slices_step2.png`, `index.html`, and `viz_summary.json`.
- Verified required files exist with nonzero sizes using `ls -lh` and `file`.
- Counts:
  step0 unknown 170910, free 19335, occupied 1755;
  step1 unknown 143439, free 44515, occupied 4046;
  step2 unknown 143335, free 44435, occupied 4230,
  observed ratio 0.2534635416666667.
- Open3D PLY export was skipped because `open3d` is not installed.

No RL, PPO, behavior cloning training, imitation-learning training, SSCNet
inference, PredictionLayer integration, expert scoring, target label use,
ground-truth map use, or observed_state modification was performed in
Stage 4A-1-viz.

Stage 4A-1-scene-viz scripted scene visualization actions:

- Re-read `simulator_notes.md`, `sim_explorer/README.md`,
  `.project_context/CURRENT_STATE.md`, and `.project_context/TODO.md`.
- Confirmed Stage 4A-1 is complete, Isaac headless camera/depth works, and
  this step is only scene visualization.
- Inspected:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/minimal_depth_scene.py`
- Confirmed the existing minimal scene requests only
  `distance_to_image_plane`, saves `depth_*.npy`, uses `GroundPlaneCfg` plus
  `CuboidCfg` walls/obstacles, and has three fixed camera poses.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/render_minimal_scene_views.py`
- The scene-viz script recreates the same 8m x 8m floor, side walls, back wall,
  and two cuboid obstacles, requests camera data types `rgb` and
  `distance_to_image_plane`, saves true RGB PNGs, saves matplotlib depth-color
  PNGs with colorbars and min/max titles, saves an overview render, and saves a
  topdown scene layout from structured scene parameters.
- Ran py_compile for the new script.
- Ran the headless Isaac render command:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a1_scene_viz.log`
- Generated output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_scene_viz`
- Generated:
  `camera_rgb_000.png`, `camera_rgb_001.png`, `camera_rgb_002.png`,
  `camera_depth_color_000.png`, `camera_depth_color_001.png`,
  `camera_depth_color_002.png`, `scene_overview_rgb.png`,
  `scene_overview_depth_color.png`, `scene_layout_topdown.png`,
  `scene_metadata.json`, and `scene_viz_summary.json`.
- Verified required files exist with nonzero sizes.
- Verified RGB PNGs are true RGB mode:
  camera RGB images `(160, 120)`, overview RGB image `(640, 480)`.
- Verified summary JSON contains no NaN.
- Verified camera output keys:
  `rgb`, `distance_to_image_plane`.
- Depth stats:
  pose0 min 1.3499999 max 3.9250004;
  pose1 min 1.6134452 max 3.9250004;
  pose2 min 0.3499999 max 2.9250004;
  overview min 3.1672034 max 11.9707413.
- Headless render still prints GLFW/default-display warnings, but Vulkan
  rendering succeeds and generated images are non-empty.

No RL, PPO, behavior cloning training, imitation-learning training, SSCNet
inference, PredictionLayer integration, expert scoring, target label use,
ground-truth map use, observed_map write, or observed_state modification was
performed in Stage 4A-1-scene-viz.

Stage 4A-2 simulator observed-map expert step actions:

- Re-read:
  `/home/ubuntu22/sc_explorer_ws/simulator_notes.md`
  `/home/ubuntu22/sc_explorer_ws/ssc_network_training_notes.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CURRENT_STATE.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/TODO.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/IMPLEMENTATION_PLAN.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/DECISIONS.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CODEX_LOG.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CHATGPT_SUMMARY.md`
- Confirmed Stage 4A-1 is complete and the Stage 4A-2 input is:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step2.npy`
- Inspected metadata:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_summary.json`
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/camera_info.json`
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/pose_002.json`
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/scene_metadata.json`
- Confirmed `pose_002.json` contains `position [1.0, 0.0, 1.2]`,
  `yaw_deg 0.0`, and `yaw_rad 0.0`; no yaw fallback was needed.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py`
- `sim_paper_expert.py` implements `EmptyPredictionLayer`, x/y/z grid-world
  conversion, frontier detection, frontier-adjacent FREE detection, candidate
  sampling, observed-map raycasting with conservative UNKNOWN blocking,
  paper-style gains, utility scoring, and top-N expert selection.
- Ran py_compile for all new Stage 4A-2 files.
- Ran Stage 4A-2 expert step:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a2_sim_expert_step.log`
- Expert step output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke`
- Generated:
  `expert_step_decision.npz`, `expert_step_decision.json`,
  `expert_step_candidates.jsonl`, `expert_topdown.png`, and
  `expert_score_bar.png`.
- Observed map stats:
  shape `(80, 80, 30)`, unknown `143335`, free `44435`, occupied `4230`,
  observed_ratio `0.2534635416666667`.
- Frontier/candidate stats:
  frontier_count `5929`, frontier_adjacent_free_count `5876`,
  candidates `64`, top_n `16`.
- Best candidate:
  expert_action `0`, candidate id `63`, score `88.83270299135849`,
  gain_exp `73.0`, gain_sc `0.0`, gain_hybrid `73.0`,
  path_cost `0.8217694333482268`, grid `(51, 38, 14)`,
  world `(1.15, -0.15, 1.45)`, yaw `-0.7030942394487684`.
- Ran Stage 4A-2 smoke test:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a2_sim_expert_test.log`
- Smoke test passed and verified output existence/format, EmptyPredictionLayer
  gain invariants, finite scores, valid expert_action, PNG visualization files,
  observed_state hash/content unchanged, and no RL/optimizer/policy training.
- Updated `simulator_notes.md`, `ssc_network_training_notes.md`, and project
  context files. `DECISIONS.md` was not updated because no new long-term
  design decision was added.

No RL, PPO, behavior cloning training, imitation-learning training, optimizer
step, policy training, SSCNet inference on Isaac depth, SSCNet retraining,
NYU target_lr/target_hr use, scene ground-truth use, simulator ground-truth use,
prediction write into observed_map, or modification of `observed_state_step*.npy`
was performed in Stage 4A-2.

Stage 4A-3 empty-prediction expert rollout actions:

- Re-read:
  `/home/ubuntu22/sc_explorer_ws/simulator_notes.md`
  `/home/ubuntu22/sc_explorer_ws/ssc_network_training_notes.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CURRENT_STATE.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/TODO.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/IMPLEMENTATION_PLAN.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/DECISIONS.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CODEX_LOG.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CHATGPT_SUMMARY.md`
- Confirmed Stage 4A-2 is complete and Stage 4A-3 should only implement a
  deterministic multi-step EmptyPredictionLayer expert rollout.
- Confirmed no RL, PPO, behavior cloning training, imitation-learning training,
  SSCNet training, SSCNet inference on Isaac depth, PredictionLayer/SSCNet
  prediction connection, NYU target labels, scene ground truth, simulator
  ground truth, or prediction writes into observed_map should be used.
- Added:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py`
- `depth_to_voxel.py` now exposes
  `update_observed_state_from_depth(...)` for single-frame rollout updates,
  while the original CLI remains intact.
- `run_sim_expert_rollout.py` recreates the minimal indoor-like Isaac scene,
  captures RGB/depth at each current camera pose, updates a measured-only
  observed_state, runs `select_sim_expert_action(...)` with
  `EmptyPredictionLayer`, converts the best candidate into planar camera
  motion, and repeats.
- Stage 4A-3 default motion mode is planar teleport:
  candidate x/y, fixed camera height, candidate yaw, level pitch/roll.
- The rollout saves per-step `.npz` transitions, `transitions.jsonl`,
  `observed_state_step*.npy`, `observed_state_final.npy`,
  `episode_summary.json`, and global `manifest.jsonl`.
- `visualize_sim_rollout.py` generates the final topdown path,
  observed_ratio curve, frontier_count curve, per-step topdown images, and
  `rollout_index.html`.
- Ran py_compile for the new/updated Stage 4A-3 files.
- Ran synthetic transition serialization smoke validation.
- Ran real Isaac headless rollout:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a3_rollout_empty_pred.log`
- Real rollout output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000`
- Real rollout result:
  steps_completed `10`, done_reason `max_steps`, observed_ratio
  `0.0 -> 0.21754166666666666`, final unknown/free/occupied
  `150232 / 35873 / 5895`, final pose
  `[3.549999952316284, 3.25, 1.2000000476837158]`.
- Expert rollout stats:
  average frontier_count `4525.6`, average candidates `64.0`,
  best_score min/mean/max `29.41531522194122 / 105.48766454499457 /
  190.10038228379815`, gain_exp min/mean/max `37.0 / 63.8 / 89.0`,
  gain_sc min/mean/max `0.0 / 0.0 / 0.0`.
- Ran Stage 4A-3 smoke test:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a3_rollout_test.log`
- Smoke test passed:
  synthetic transition serialization ok, real episode validation ok,
  observed_ratio non-decreasing, EmptyPredictionLayer `gain_sc=0`,
  prediction did not write observed_map, and no RL/optimizer/BC/IL training ran.
- Updated `simulator_notes.md`, `ssc_network_training_notes.md`, and project
  context files.
- Updated `DECISIONS.md` with the Stage 4A-3 planar teleport motion decision.

No RL, PPO, behavior cloning training, imitation-learning training, optimizer
step, policy training, SSCNet inference on Isaac depth, SSCNet retraining,
NYU target_lr/target_hr use, scene ground-truth use, simulator ground-truth use,
PredictionLayer/SSCNet map_predict connection, prediction write into
observed_map, A* planning, full SC-Explorer RRT tree planning, or physical
robot collision-checked path execution was performed in Stage 4A-3.

Stage 4A-3.2 medium-complexity scripted scene actions:

- Re-read:
  `/home/ubuntu22/sc_explorer_ws/simulator_notes.md`
  `/home/ubuntu22/sc_explorer_ws/ssc_network_training_notes.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CURRENT_STATE.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/TODO.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/IMPLEMENTATION_PLAN.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/DECISIONS.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CODEX_LOG.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CHATGPT_SUMMARY.md`
- Confirmed Stage 4A-3 is complete and the current task is a more complex
  scripted Isaac scene plus depth/observed-map smoke, not RL, PPO, behavior
  cloning training, imitation-learning training, SSCNet training, SSCNet
  inference on Isaac depth, PredictionLayer connection, prediction writes into
  observed_map, target labels, or ground truth.
- Added:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/medium_complex_depth_scene.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/render_medium_complex_scene_views.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_medium_complex_scene_metadata.py`
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py`
- `scene_factory.py` defines a deterministic `three_rooms` medium scene with
  bounds `x/y=[-6,6]`, `z=[0,3]`, 12m x 12m floor, 2.2m walls, 3 rooms,
  1 corridor, 3 openings, 13 wall segments, 13 cuboid obstacles, fixed camera
  poses, and JSON metadata.
- `depth_to_voxel.py` now accepts explicit bounds CLI arguments:
  `--x_min --x_max --y_min --y_max --z_min --z_max`, while preserving the old
  minimal-scene defaults.
- Ran py_compile checks for new/updated Stage 4A-3.2 files.
- Ran pure Python metadata test:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_metadata_test.log`
- Ran real Isaac headless medium scene capture:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_depth.log`
- Generated:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_smoke`
- Ran measured-only depth_to_voxel fusion:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_depth_to_voxel.log`
- Observed map result:
  shape `(120, 120, 30)`, unknown/free/occupied `339813 / 86064 / 6123`,
  observed_ratio `0.21339583333333334`.
- Ran medium scene visualization:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_viz.log`
- Generated:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_viz`
- Visualization files include:
  `scene_overview_rgb.png`, `scene_overview_depth_color.png`,
  `scene_layout_topdown.png`, `camera_rgb_grid.png`, `camera_depth_grid.png`,
  `observed_topdown_compare.png`, `free_occupied_voxels_3d_final.png`, and
  `slices_final.png`.
- Ran validation checks for required files, nonblank RGB/images, finite
  positive depth, observed map UNKNOWN/FREE/OCCUPIED values, observed_ratio
  greater than 0.05, and topology counts.
- Ran optional one-step expert smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_expert_step.log`
- Optional expert result:
  `frontier_count=20919`, `frontier_adjacent_free_count=21637`,
  `candidates=64`, `top_n=16`, best score `53.62160777611031`,
  best grid `[64, 91, 13]`, best world `[0.45, 3.15, 1.35]`, and
  EmptyPredictionLayer `gain_sc=0.0`.
- Updated `simulator_notes.md`, `ssc_network_training_notes.md`, and project
  context files.
- Updated `DECISIONS.md` with the decision to increase scripted scene
  complexity before scaling rollouts.

No RL, PPO, behavior cloning training, imitation-learning training, optimizer
step, policy training, SSCNet inference on Isaac depth, SSCNet retraining,
NYU target_lr/target_hr use, scene ground-truth use, simulator ground-truth use,
PredictionLayer/SSCNet map_predict connection, prediction write into
observed_map, large-scale rollout dataset generation, A* planning, full
SC-Explorer RRT tree planning, or physical robot collision-checked path
execution was performed in Stage 4A-3.2.

Stage 4A-3.5 observed-free A* path-cost actions:

- Re-read:
  `/home/ubuntu22/sc_explorer_ws/simulator_notes.md`
  `/home/ubuntu22/sc_explorer_ws/ssc_network_training_notes.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CURRENT_STATE.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/TODO.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/IMPLEMENTATION_PLAN.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/DECISIONS.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CODEX_LOG.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CHATGPT_SUMMARY.md`
- Confirmed Stage 4A-3.2 is complete and Stage 4A-3.5 should only add A*
  path-cost scoring over observed FREE space for the simulator expert.
- Confirmed no RL, PPO, behavior cloning training, imitation-learning
  training, SSCNet inference on Isaac depth, PredictionLayer connection,
  prediction writes into observed_map, target labels, or ground truth should be
  used.
- Added:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/astar_planner.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_astar_planner.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_astar.py`
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
- `astar_planner.py` implements `build_traversability_grid`, `astar_2d`,
  `path_length_m`, `summarize_traversability`, and optional visualization.
- Traversability uses only `observed_state`: FREE support in the robot-height
  band is traversable, OCCUPIED is blocked and inflated by robot radius,
  UNKNOWN is not traversable.
- `sim_paper_expert.py` now accepts `path_cost_mode=euclidean|astar`.
  The default remains `euclidean`.
- In A* mode, unreachable candidates are marked invalid with
  `unreachable_astar:<reason>`, `path_cost=inf`, and `final_score=-inf`; no
  Euclidean fallback is used.
- Candidate features now append:
  `astar_reachable`, `astar_path_length_m`, and `astar_num_expanded`.
- `run_sim_expert_step.py` now accepts `--path_cost_mode`.
- `run_sim_expert_rollout.py` now accepts `--path_cost_mode`,
  `--scene_variant minimal|medium_three_rooms`, `--scene_seed`,
  `--map_bound_mode`, and optional explicit bounds.
- Rollout visualization overlays selected A* paths when available and saves
  `reachable_candidates_curve.png`.
- Ran py_compile for Stage 4A-3.5 files.
- Ran A* planner tests:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_astar_planner_test.log`
- Ran one-step medium A* expert:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_medium_expert_step_astar.log`
- One-step result:
  traversable/blocked/unknown `4316 / 1907 / 8177`,
  reachable/unreachable candidates `12 / 52`, best score
  `51.651363679237036`, best gain_exp `110.0`, best gain_sc `0.0`,
  best path_cost `2.129663036258191`, best A* path length
  `1.2656854249492382m`, best grid `[64, 91, 13]`, best world
  `[0.45, 3.15, 1.35]`.
- Ran medium A* rollout:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_medium_rollout_astar_empty_pred.log`
- Rollout result:
  episode `medium_three_rooms_astar_empty_pred_000`, `steps_completed=5`,
  `done_reason=no_valid_candidate`, observed_ratio
  `0.0 -> 0.04308796296296296`, final unknown/free/occupied
  `413386 / 15863 / 2751`, average reachable candidates `18.4`, average
  best path_cost `0.9421159585855353`.
- Main blocker:
  at expert step 5 all 64 candidates were unreachable under conservative
  observed-free A* traversability (`traversable=338`, `blocked=918`,
  `unknown=13144`). No fallback was used.
- Ran simulator A* validator:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_sim_expert_astar_test.log`
- Ran Euclidean regression tests:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_regression_sim_paper_expert_euclidean_test.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a35_regression_sim_expert_rollout_test.log`
- Updated `simulator_notes.md`, `ssc_network_training_notes.md`, and project
  context files.
- Updated `DECISIONS.md` with the decision to build the paper expert
  step-by-step before IL/RL and to use observed-free A* as the Stage 4A-3.5
  path-cost upgrade.

No RL, PPO, behavior cloning training, imitation-learning training, optimizer
step, policy training, SSCNet inference on Isaac depth, SSCNet retraining,
NYU target_lr/target_hr use, scene ground-truth use, simulator ground-truth use,
PredictionLayer/SSCNet map_predict connection, prediction write into
observed_map, large-scale rollout dataset generation, full SC-Explorer RRT tree
planning, or physical robot collision-checked path execution was performed in
Stage 4A-3.5.

Stage 4A-3.6 reachability-aware A* candidate sampling actions:

- Re-read:
  `/home/ubuntu22/sc_explorer_ws/simulator_notes.md`
  `/home/ubuntu22/sc_explorer_ws/ssc_network_training_notes.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CURRENT_STATE.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/TODO.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/IMPLEMENTATION_PLAN.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/DECISIONS.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CODEX_LOG.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CHATGPT_SUMMARY.md`
- Confirmed Stage 4A-3.5 is complete and Stage 4A-3.6 should only fix A*
  candidate sampling / reachability diagnostics / rollout robustness.
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/astar_planner.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_astar_planner.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_astar.py`
- Added:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_reachable_candidate_sampling.py`
- Implemented `connected_component_from_start`, `nearest_traversable_cell`,
  `frontier_reachable_candidate_mask`, and
  `compute_reachable_frontier_candidate_cells`.
- Added `candidate_sampling_mode=frontier|reachable_frontier|auto`, with A*
  `auto` resolving to `reachable_frontier` and Euclidean `auto` preserving old
  frontier sampling.
- Added `--snap_start_to_traversable` and `--max_snap_radius_cells` to
  one-step and rollout CLIs.
- Added reachable component diagnostics, candidate source labels, snapped
  current xy fields, reachable frontier-adjacent counts, candidate source
  counts, and rollout reachable-component curves.
- Excluded the current/snap start cell from reachable candidates when other
  candidates exist, preventing trivial same-pose choices during rollout.
- Ran py_compile for Stage 4A-3.6 files.
- Ran reachable candidate sampling tests:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_reachable_candidate_sampling_test.log`
- Ran one-step medium reachable A* expert:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_medium_expert_step_astar_reachable.log`
- One-step result:
  reachable/unreachable candidates `64 / 0`, reachable component count
  `1196`, reachable frontier-adjacent count `1196`, top_n `16`, best score
  `88.24634362636618`, best gain_exp `66.0`, gain_sc `0.0`, best path_cost
  `0.7479063413600806`, best A* path length `0.28284271247461906m`, best
  grid `[58, 82, 11]`.
- Ran medium reachable A* rollout:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_medium_rollout_astar_reachable_empty_pred.log`
- Rollout result:
  episode `medium_three_rooms_astar_reachable_empty_pred_000`,
  `steps_completed=10`, `done_reason=max_steps`, observed_ratio
  `0.0 -> 0.10147453703703704`, final unknown/free/occupied
  `388163 / 36017 / 7820`, average reachable candidates `64.0`, average
  reachable component count `238.8`, average reachable frontier-adjacent
  count `238.8`, `no_valid_candidate_steps=[]`.
- Ran A* planner regression:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_astar_planner_regression_test.log`
- Ran simulator A* reachable-output validator:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_sim_expert_astar_reachable_test.log`
- Ran Euclidean regression smoke checks:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_regression_sim_paper_expert_euclidean_test.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a36_regression_sim_expert_rollout_test.log`
- Updated `simulator_notes.md`, `ssc_network_training_notes.md`, and project
  context files.
- Updated `DECISIONS.md` with the decision to sample A* candidates from the
  current reachable observed-free component before scaling rollout datasets.

No RL, PPO, behavior cloning training, imitation-learning training, optimizer
step, policy training, SSCNet inference on Isaac depth, SSCNet retraining,
NYU target_lr/target_hr use, scene ground-truth use, simulator ground-truth use,
PredictionLayer/SSCNet map_predict connection, prediction write into
observed_map, Euclidean fallback, full SC-Explorer RRT tree planning, or
physical robot collision-checked path execution was performed in Stage 4A-3.6.

Stage 4A-4 multi-episode EmptyPredictionLayer A* rollout dataset actions:

- Re-read:
  `/home/ubuntu22/sc_explorer_ws/simulator_notes.md`
  `/home/ubuntu22/sc_explorer_ws/ssc_network_training_notes.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CURRENT_STATE.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/TODO.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/IMPLEMENTATION_PLAN.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/DECISIONS.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CODEX_LOG.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CHATGPT_SUMMARY.md`
- Confirmed Stage 4A-3.6 is complete and Stage 4A-4 should only generate a
  deterministic multi-episode EmptyPredictionLayer expert rollout dataset.
- Added:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_batch.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_rollout_dataset.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_rollout_dataset_batch.py`
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
- The batch runner creates episode ids of the form
  `medium_three_rooms_seed{seed}_{start_variant}_empty_astar`, supports
  continue-on-error, skip-existing, per-episode logs, and a batch-controlled
  manifest.
- The rollout runner now saves start-variant/start-pose metadata and supports
  `--no_manifest` so batch manifests are not duplicated.
- Transitions now expose selected expert fields directly:
  `gain_exp`, `gain_sc`, `gain_hybrid`, `path_cost`, and `final_score`.
- Ran py_compile for the updated/new Stage 4A-4 files.
- Ran lightweight regressions:
  `test_reachable_candidate_sampling.py`, `test_astar_planner.py`, and
  `test_sim_expert_rollout.py`.
- Ran the requested 9-episode headless batch:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_batch_empty_pred_astar.log`
- Dataset output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar`
- Batch setup:
  scene_variant `medium_three_rooms`, seeds `0,1,2`, start variants
  `start_room_a,start_corridor,start_room_b`, max_steps `10`,
  num_candidates `64`, top_n `16`, gain_mode `hybrid`, prediction_mode
  `empty`, path_cost_mode `astar`, candidate_sampling_mode
  `reachable_frontier`, motion_mode `planar`.
- Batch result:
  ok episodes `9`, failed episodes `0`, total transitions `90`, all episodes
  ended with `done_reason=max_steps`.
- Ran summarizer:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_dataset_summary.log`
- Summary result:
  steps min/mean/max `10 / 10 / 10`, observed_ratio_end min/mean/max
  `0.08587037037037037 / 0.11455118312757204 / 0.16534722222222223`,
  average reachable candidates `64.0`, average reachable component count
  `570.3444444444444`, average best_score `163.2387554327081`, average
  gain_exp `49.15555555555556`, average gain_sc `0.0`, average path_cost
  `0.45623051832594874`, no_valid_candidate episodes `0`.
- Generated:
  `manifest.jsonl`, `dataset_summary.json`, `dataset_summary.md`,
  `rollout_dataset_index.html`, aggregate plots, and per-episode rollout
  outputs.
- Ran batch dataset validator:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_dataset_test.log`
- Validator result:
  ok episodes `9`, preferred ok episodes met, failed episodes `0`, total
  transitions `90`, observed_ratio non-decreasing, EmptyPredictionLayer
  `gain_sc=0`, leakage checks passed, no RL/optimizer/BC/IL training,
  prediction writes observed_map `no`, UNKNOWN traversability shortcut `no`,
  Euclidean fallback `no`.
- Updated `simulator_notes.md`, `ssc_network_training_notes.md`, and project
  context files.
- Updated `DECISIONS.md` with the decision that Stage 4A-4 generates a
  multi-episode measured-only expert rollout dataset before connecting
  map_predict.

No RL, PPO, behavior cloning training, imitation-learning training, optimizer
step, policy training, SSCNet inference on Isaac depth, SSCNet retraining,
NYU target_lr/target_hr use, scene ground-truth use, simulator ground-truth use,
PredictionLayer/SSCNet map_predict connection, prediction write into
observed_map, UNKNOWN traversability shortcut, Euclidean fallback, full
SC-Explorer RRT tree planning, or physical robot collision-checked path
execution was performed in Stage 4A-4.

Stage 4A-5 Isaac single-frame map_predict alignment smoke actions:

- Re-read:
  `/home/ubuntu22/sc_explorer_ws/simulator_notes.md`
  `/home/ubuntu22/sc_explorer_ws/ssc_network_training_notes.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CURRENT_STATE.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/TODO.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/IMPLEMENTATION_PLAN.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/DECISIONS.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CODEX_LOG.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CHATGPT_SUMMARY.md`
- Confirmed Stage 4A-4 is complete and Stage 4A-5 should only run
  single-frame Isaac map_predict preprocessing, inference, alignment, and
  read-only artifact validation.
- Inspected:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/utils/projection_layer.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/dataloaders/dataloader.py`
- Checked real NYU `position` samples under:
  `/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz`
- Wrote position convention log:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a5_position_convention_check.log`
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_sscnet_preprocess.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_prediction_layer.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_map_predict_single.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_isaac_prediction_alignment.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_map_predict_single.py`
- Selected first ok Stage 4A-4 manifest episode:
  `medium_three_rooms_seed0_start_room_a_empty_astar`, step `0`.
- Input files:
  `depth_000.npy`, `pose_000.json`, `camera_info.json`,
  `observed_state_step000.npy`, `episode_summary.json`, and
  `scene_metadata.json`.
- Preprocessing result:
  depth input `(480,640)`, position `(480,640)`, valid position pixels
  `166888`.
- Loaded best checkpoint strictly:
  `/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar`
- Ran one SSCNet inference:
  logits `(1,12,60,36,60)`, local prediction `(60,36,60)`,
  inference time `0.1617s`.
- Aligned local prediction into a global read-only simulator layer with shape
  `(120,120,30)`.
- Output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke`
- Generated:
  `sscnet_input_debug.npz`, `sscnet_depth_input.npy`,
  `sscnet_position.npy`, `valid_position_mask.npy`,
  `local_prediction.npz`, `global_prediction_layer.npz`,
  `prediction_alignment_summary.json`, `isaac_depth_input.png`,
  `local_prediction_slices.png`, `global_prediction_topdown.png`,
  `observed_vs_prediction_topdown.png`, and
  `prediction_not_measured_topdown.png`.
- Run stats:
  global valid prediction voxels `56602`, predicted occupied voxels `15664`,
  predicted_unmeasured voxels `39400`, observed_state modified `false`.
- Ran py_compile for all new Stage 4A-5 files.
- Ran Stage 4A-5 smoke test:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a5_isaac_map_predict_single_test.log`
- Smoke test passed and verified selected inputs exist, observed_state hash
  unchanged, preprocessing/inference/alignment shapes, finite probabilities,
  SimPredictionLayer API, predicted_unmeasured count, no forbidden target or
  ground-truth artifact fields, no observed_state writeback, no
  traversability/collision/A* prediction use, no RL/optimizer/BC/IL training,
  and no expert/rollout prediction use.
- Updated `simulator_notes.md`, `ssc_network_training_notes.md`, and project
  context files.
- Updated `DECISIONS.md` with the decision that map_predict is currently only
  a read-only layer and must not affect observed_map, traversability,
  collision, A*, expert decisions, or rollout decisions.

No RL, PPO, behavior cloning training, imitation-learning training, optimizer
step, policy training, SSCNet training, checkpoint modification,
target_lr/target_hr use, scene ground-truth use, simulator ground-truth use,
observed_map prediction writeback, prediction-based traversability/collision/A*,
expert prediction use, rollout prediction use, full rollout, full RRT planning,
or physical robot collision-checked execution was performed in Stage 4A-5.

Stage 4A-5.1 one-step SC-aware expert scoring actions:

- Re-read:
  `/home/ubuntu22/sc_explorer_ws/simulator_notes.md`
  `/home/ubuntu22/sc_explorer_ws/ssc_network_training_notes.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CURRENT_STATE.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/TODO.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/IMPLEMENTATION_PLAN.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/DECISIONS.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CODEX_LOG.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CHATGPT_SUMMARY.md`
- Confirmed Stage 4A-5 is complete and Stage 4A-5.1 should only use the
  read-only simulator prediction layer for one-step expert information gain.
- Inspected:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_prediction_layer.py`
- Updated `sim_paper_expert.py` to accept a prediction-layer-compatible object
  in `prediction_mode=sim_npz`.
- Preserved `prediction_mode=empty` default behavior with
  `EmptyPredictionLayer`.
- Kept prediction out of candidate sampling, observed-free A* traversability,
  A* path validity, collision checking, observed_state writeback, and ray
  blocking.
- Added diagnostics for prediction layer shape/counts and leakage flags.
- Updated `run_sim_expert_step.py` with:
  `--prediction_mode empty|sim_npz`, `--prediction_npz`, `--tau`, and
  `--episode_summary`.
- Added observed_state hash-before/hash-after diagnostics in the one-step
  runner.
- Updated `visualize_sim_expert_step.py` with:
  `prediction_overlay_topdown.png` and
  `predicted_unmeasured_visible_topdown.png`.
- Added:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_with_prediction.py`
- Ran py_compile:
  `sim_prediction_layer.py`, `sim_paper_expert.py`,
  `run_sim_expert_step.py`, `visualize_sim_expert_step.py`, and
  `test_sim_expert_with_prediction.py`.
- Ran empty baseline:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_empty_baseline.log`
- Ran SC prediction expert:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_sc_prediction.log`
- Ran Stage 4A-5.1 smoke test:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_with_prediction_test.log`
- Outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/empty_baseline`
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/sc_prediction`
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/comparison_summary.json`
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/comparison_summary.md`
- Empty baseline best:
  id `11`, score `331.3448560321166`, gain_exp `55.0`, gain_sc `0.0`,
  gain_hybrid `55.0`, path_cost `0.1659902032541859`, grid `[13,13,11]`.
- SC prediction best:
  id `11`, score `662.6897120642332`, gain_exp `55.0`, gain_sc `55.0`,
  gain_hybrid `110.0`, gain_occ `13.0`, gain_conf `19.406008422374725`,
  path_cost `0.1659902032541859`, grid `[13,13,11]`.
- Comparison:
  best candidate changed `false`, score delta `331.3448560321166`,
  gain_hybrid delta `55.0`, top-N overlap `16/16`, candidates with
  `gain_sc>0` `64/64`, max/mean gain_sc `174.0/71.59375`.
- Test verified observed_state hash unchanged, prediction layer shape matches
  observed_state, empty gain_sc zero, prediction gain_sc nonzero,
  gain_hybrid identity, finite gain_occ/gain_conf, no target/ground-truth
  leakage, no prediction traversability/collision/A*/ray-blocking/writeback,
  no RL/optimizer/BC/IL training, and no rollout.

No rollout, RL, PPO, behavior cloning training, imitation-learning training,
optimizer step, policy training, SSCNet training, checkpoint modification,
target_lr/target_hr use, scene ground-truth use, simulator ground-truth use,
observed_map prediction writeback, prediction-based traversability/collision/A*,
prediction ray blocking, full RRT planning, or physical robot
collision-checked execution was performed in Stage 4A-5.1.

Stage 4A-6 short multi-step SC-aware rollout actions:

- Re-read:
  `/home/ubuntu22/sc_explorer_ws/simulator_notes.md`
  `/home/ubuntu22/sc_explorer_ws/ssc_network_training_notes.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CURRENT_STATE.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/TODO.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/IMPLEMENTATION_PLAN.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/DECISIONS.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CODEX_LOG.md`
  `/home/ubuntu22/sc_explorer_ws/.project_context/CHATGPT_SUMMARY.md`
- Confirmed Stage 4A-5.1 is complete and Stage 4A-6 should run a short
  dynamic SC-aware rollout only.
- Added:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_map_predictor.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/compare_sc_pred_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_sc_aware_rollout.py`
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
- `IsaacMapPredictor` loads the SSCNet checkpoint once, runs per-step Isaac
  depth preprocessing, performs GPU inference on RTX 5080, aligns local
  `(60,36,60)` prediction to global observed_state shape `(120,120,30)`, and
  returns a read-only `SimPredictionLayer`.
- `run_sim_expert_rollout_sc_pred.py` keeps the loop measured-only for
  observed_state update and observed-free A*:
  Isaac depth -> measured observed_state update -> map_predict -> read-only
  SimPredictionLayer -> SC-aware information gain -> planar teleport.
- Prediction is used only for `gain_sc`, `gain_hybrid`, `gain_occ`,
  `gain_conf`, and final_score. It is not used for A* traversability,
  collision, candidate reachability, observed_state writes, or ray blocking.
- First Stage 4A-6 run exposed a visualization metadata bug:
  `prediction_alignment_summary.json` was missing `observed_state_source`.
- Fixed `isaac_map_predictor.py` to write `observed_state_source`,
  `depth_source`, `pose_source`, and `camera_info_source`.
- Ran final 5-step rollout:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a6_sc_pred_dynamic_rollout.log`
- Rollout output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke/episodes/medium_three_rooms_seed0_start_room_a_sc_pred_dynamic_000`
- Rollout result:
  steps_completed `5`, done_reason `max_steps`, observed_ratio
  `0.0 -> 0.05899768518518519`, final unknown/free/occupied
  `406513 / 21226 / 4261`, final pose
  `[-4.25, -4.150000095367432, 1.2000000476837158]`.
- SC-aware stats:
  average gain_exp `49.6`, gain_sc `49.4`, gain_hybrid `99.0`,
  gain_occ `8.8`, gain_conf `16.96283725500107`, average best_score
  `441.9845465468916`, candidates_with_gain_sc_positive min/mean/max
  `63 / 63.6 / 64`.
- map_predict performance:
  model_loaded_once `true`, average inference_time `0.020522295199771178s`,
  average total prediction time `0.14326694260016665s`, average expert_time
  `1.026360238399866s`, GPU memory peak `794354176` bytes.
- Ran comparison:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a6_sc_vs_empty_comparison.log`
- Comparison output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke/comparison_to_empty_baseline`
- Comparison result:
  compared_steps `5`, empty final observed_ratio `0.06896296296296296`,
  SC final observed_ratio `0.05899768518518519`, SC-empty delta
  `-0.009965277777777774`, changed selected actions `5`, mean score delta
  `233.79287700349096`, mean SC gain_sc `49.4`.
- Ran py_compile for:
  `isaac_map_predictor.py`, `run_sim_expert_rollout_sc_pred.py`,
  `compare_sc_pred_rollout.py`, `test_sim_sc_aware_rollout.py`,
  `sim_paper_expert.py`, `sim_prediction_layer.py`,
  `isaac_sscnet_preprocess.py`, and `visualize_sim_rollout.py`.
- Ran Stage 4A-6 test:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a6_sc_aware_rollout_test.log`
- Test verified:
  episode summary and transitions exist, steps_completed `5`,
  prediction artifacts exist for every step, observed_ratio is non-decreasing,
  prediction_mode `sim_dynamic`, path_cost_mode `astar`,
  candidate_sampling_mode `reachable_frontier`, gain_sc nonzero,
  candidates_with_gain_sc_positive > 0, gain_hybrid identity, observed_state
  hash unchanged by prediction, no prediction traversability/collision/A*/ray
  leakage, no target/ground-truth leakage, no RL/optimizer/BC/IL/SSCNet
  training, checkpoint not modified, and comparison summary exists.
- Closeout verification checked required episode outputs, per-step NPZ safety
  fields, prediction directories, rollout plots, comparison plots, logs, and
  checkpoint stat/hash. Required files exist and are non-empty.
- Per-step selected `gain_sc` was positive on all five completed steps:
  `53, 52, 48, 53, 41`; selected `gain_hybrid` equaled
  `gain_exp + gain_sc` on all steps.
- Prediction safety fields stayed false on every step:
  observed_state modified by prediction, prediction writeback,
  traversability, collision, A*, and ray blocking.
- Stage 4A-6 comparison showed lower observed_ratio than the measured-only
  baseline at the 5-step horizon:
  empty `0.06896296296296296`, SC-aware `0.05899768518518519`, delta
  `-0.009965277777777774`. This is recorded as an integration-correctness
  success, not a performance improvement.
- Log scan for Stage 4A-6 final logs found no current Traceback, Error, CUDA
  unavailable, checkpoint reload-every-step, prediction writeback, target_lr,
  ground_truth, optimizer, PPO/RL, behavior-cloning, or imitation-learning
  training issue beyond expected false/no status lines.

No RL, PPO, behavior cloning training, imitation-learning training, optimizer
step, policy training, SSCNet training, checkpoint modification,
target_lr/target_hr use, scene ground-truth use, simulator ground-truth use,
observed_map prediction writeback, prediction-based traversability/collision/A*,
prediction ray blocking, full RRT planning, or physical robot
collision-checked execution was performed in Stage 4A-6.

Stage 4A-6.1 SC-aware rollout analysis/ablation actions:

- Re-read simulator notes, SSCNet notes, and project context.
- Confirmed Stage 4A-6 is complete and the current task is Stage 4A-6.1
  analysis/ablation/tuning, not rollout scaling and not RL/IL.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/analyze_sc_rollout_behavior.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sc_pred_ablation_sweep.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_sc_pred_ablation.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sc_pred_ablation.py`
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
- Added optional `sc_gain_weight`, `sc_gain_cap`, and
  `score_gain_mode=hybrid_raw|hybrid_weighted`. Default raw hybrid behavior is
  unchanged.
- Ran existing SC-vs-empty analysis:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a61_existing_sc_analysis.log`
- Existing analysis output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/existing_sc_vs_empty`
- Existing analysis result:
  empty final observed_ratio `0.06896296296296296`, SC final observed_ratio
  `0.05899768518518519`, delta `-0.009965277777777774`, first SC lag step
  `1`, changed actions `5/5`, mean path_cost empty/SC
  `0.36998367643136965 / 0.2768163156997422`, mean gain_exp empty/SC
  `54.8 / 49.6`, mean SC gain_sc `49.4`.
- Ran sequential ablation sweep:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a61_ablation_sweep.log`
- Completed configs:
  `dynamic_w025_tau01`, `dynamic_w05_tau01`, `dynamic_w1_tau03`,
  `dynamic_w1_tau01_cap50`, `static_step0_weight_1p0_tau_0p1`; failed
  configs `[]`.
- All ablations completed 5 steps and ended at observed_ratio
  `0.05899768518518519`, still below the empty baseline by
  `-0.009965277777777774`.
- All ablations selected the same `5/5` actions as the original SC rollout.
- Ablation resource profile:
  dynamic wall time `27.75568118299998s` to `30.416639261s`, average
  map_predict inference `0.02070385000006354s` to `0.03013971940004012s`,
  dynamic GPU peak `794296320` bytes; static step0 wall time
  `22.448690754999916s`, average inference `0.0s`, GPU peak `None`.
- Ran ablation summary:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a61_ablation_summary.log`
- Summary output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_ablation/summary`
- Qualitative output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/qualitative_inspection`
- Ran py_compile and Stage 4A-6.1 validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a61_ablation_test.log`
- Validation passed:
  at least two ablations completed, observed_ratio non-decreasing, weighted
  gain formula correct, default weighted identity preserved, prediction
  read-only and information-gain-only, no prediction traversability/collision
  /A*/ray blocking/writeback, checkpoint not modified, no target arrays, and
  no RL/optimizer/BC/IL/SSCNet training.
- Log scan for Stage 4A-6.1 logs found no Traceback, Error, CUDA unavailable,
  checkpoint modified, target_lr/target_hr, ground_truth, optimizer, PPO/RL,
  behavior-cloning, imitation-learning training, prediction writeback, UNKNOWN
  traversable, or Euclidean fallback issue.

Stage 4A-6.1 conclusion:

- SC-aware scoring is active and safe, but gain_sc is dense and did not become
  selective enough to improve measured coverage in this seed/start.
- Weighting, tau increase, and cap reduced score magnitude but did not change
  selected actions.
- The next step should inspect map_predict preprocessing, global alignment,
  confidence calibration, and NYU-to-Isaac domain shift before longer
  SC-aware rollout scaling.
- No RL, PPO, behavior cloning training, imitation-learning training,
  optimizer step, SSCNet training, checkpoint modification, target_lr/target_hr
  use, scene/simulator ground-truth scoring, prediction writeback,
  prediction-based traversability/collision/A*, prediction ray blocking, full
  RRT planning, or physical robot collision-checked execution was performed in
  Stage 4A-6.1.

Stage 4A-6.2 map_predict diagnostics actions:

- Re-read simulator notes, SSCNet notes, and project context.
- Confirmed Stage 4A-6 and Stage 4A-6.1 are complete and Stage 4A-6.2 is
  prediction diagnostics, not policy tuning, rollout scaling, RL/IL, or
  training.
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_isaac_sscnet_preprocess.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_prediction_global_alignment.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/evaluate_prediction_against_future_observed.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_alignment_variant_sweep.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_map_predict_diagnostics.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_map_predict_diagnostics.py`
- Ran preprocessing diagnostics:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_preprocess_stats.log`
- Ran global alignment diagnostics:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_global_alignment.log`
- Ran future observed evaluation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_future_observed_eval.log`
  Future observations are post-hoc sensor validation only, not planning or
  expert scoring.
- Ran diagnostic alignment variant sweep:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_alignment_variant_sweep.log`
- Ran summary and candidate-score decomposition:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_diagnostic_summary.log`
- Ran py_compile and validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_py_compile.log`
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a62_diagnostics_test.log`
- Diagnostics root:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics`
- Preprocessing finding:
  Isaac valid position ratio `0.565763671875` vs NYU position nonzero proxy
  `0.74495458984375`; this is suspicious and supports preprocessing/domain
  shift review.
- Alignment finding:
  direct global sanity has mean in-front ratio `0.9977230302200312`, but the
  alignment variant sweep ranked `xz_swap_variant` best, with
  `current_default` rank `7` and Brier improvement `0.0735458940774611`;
  primary suspected issue is alignment convention.
- Calibration finding:
  tau `0.1` mean predicted_unmeasured `35118.2`, later measured fraction
  `0.059004217437215026`, occupied precision `0.25632042463242544`, occupied
  Brier `0.2786559495144023`, ECE-like calibration `0.3405436085907938`;
  prediction is dense/unselective.
- Scoring finding:
  gain_exp/gain_sc correlation `0.9647202023737985`, final_score vs inverse
  path_cost correlation `0.9713818732156227`, all five Stage 4A-6.1 ablations
  matched original SC actions `5/5`.
- Validation passed:
  observed_state hashes unchanged, checkpoint not modified, prediction
  read-only and information-gain-only, no traversability/collision/A*/ray
  blocking/candidate reachability use, future observations evaluation-only,
  diagnostics ran without Isaac startup, and no RL/optimizer/BC/IL/SSCNet
  training ran.

Stage 4A-6.2 conclusion:

- Fix or reconcile local prediction to global projection convention before any
  longer SC-aware rollout scaling.
- After alignment is fixed, address confidence calibration / I_sc selectivity
  if dense predictions remain.
- Do not jump to RL/IL.

Stage 4A-6.3 SSCNet alignment convention fix actions:

- Re-read simulator notes, SSCNet notes, and project context.
- Confirmed Stage 4A-6.2 is complete and Stage 4A-6.3 is an alignment
  convention fix/reconciliation, not rollout tuning, RL, PPO, BC, IL,
  optimizer work, SSCNet training, checkpoint modification, prediction fusion,
  or prediction writeback.
- Inspected SSCNet projection/training/inference code:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/utils/projection_layer.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/dataloaders/dataloader.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/models/SSCNet.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/offline_infer_npz.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/src/ssc_network_node.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/utils/utils.py`
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/voxel_utils/voxel_util.cpp`
- Implemented:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/document_sscnet_axis_convention.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/fix_prediction_alignment_convention.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_alignment_convention_fix.py`
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_sscnet_preprocess.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_map_predictor.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_map_predict_single.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_isaac_prediction_alignment.py`
- Added named conventions:
  `current_default_v0`, `xz_swap_diagnostic`, and `code_consistent_v1`.
- Axis audit finding:
  `Project2Dto3D` scatters flat indices into `view(W,H,D)` then permutes to
  `(D,H,W)`. The raw Python dataloader branch uses
  `np.ravel_multi_index((x,y,z),(240,144,240))`, but the C++/ROS projection
  path uses `z*(240*144)+y*240+x`. With that path, SSCNet output axes are
  `(x_right,y_up,z_forward)` after the Project2Dto3D permute.
- Ran axis audit:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a63_axis_convention_audit.log`
- Ran convention evaluation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a63_convention_eval.log`
- Convention eval result:
  `current_default_v0` occupied Brier `0.2786559495144023`,
  `code_consistent_v1` occupied Brier `0.20511005543694122`, Brier
  improvement `0.0735458940774611`, ECE-like `0.3405436085907937 ->
  0.22427722861569463`. Best diagnostic convention was
  `xz_swap_diagnostic`; recommended fixed convention is `code_consistent_v1`.
- Ran fixed single-frame map_predict smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a63_single_map_predict_fixed.log`
  output `/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke_alignment_fixed`
- Fixed single-frame result:
  global valid predictions `56602`, predicted occupied `16792`,
  predicted_unmeasured `39400`, observed_state unchanged.
- Ran fixed one-step expert smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a63_one_step_expert_fixed.log`
  output `/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_alignment_fixed_smoke`
- Fixed one-step result:
  best candidate id `11`, score `662.6897120642332`,
  gain_exp/gain_sc/gain_hybrid `55.0 / 55.0 / 110.0`, observed_state
  unchanged.
- Ran fixed 5-step dynamic SC-aware rollout:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a63_fixed_alignment_rollout.log`
  output `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_alignment_fixed_smoke`
- Fixed rollout result:
  steps `5`, observed_ratio `0.0 -> 0.05899768518518519`, equal to original
  SC and below empty baseline `0.06896296296296296`. Changed actions vs empty
  `5`; changed actions vs original SC `0`.
- Ran fixed vs empty comparison:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a63_fixed_vs_empty_comparison.log`
- Created fixed vs original SC comparison:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_alignment_fixed_smoke/comparison_to_original_sc/compare_fixed_vs_original_sc.json`
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_alignment_fixed_smoke/comparison_to_original_sc/compare_fixed_vs_original_sc.md`
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_alignment_fixed_smoke/comparison_to_original_sc/compare_fixed_vs_original_sc.png`
- Ran py_compile and validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a63_alignment_fix_test.log`
- Validation passed:
  observed_state unchanged, checkpoint not modified, future observations
  evaluation-only, prediction read-only and information-gain-only, no
  traversability/collision/A*/ray blocking/candidate reachability use, and no
  RL/PPO/optimizer/BC/IL/SSCNet training.

Stage 4A-6.3 conclusion:

- Use `code_consistent_v1` for future Isaac map_predict runs.
- Alignment diagnostics are fixed/reconciled, but the short rollout still does
  not improve because gain_sc remains dense and poorly selective.
- Next recommended work is calibrated/confidence-gated `I_sc`, not RL/IL and
  not longer rollout scaling yet.

Stage 4A-6.4 calibrated / confidence-gated I_sc actions:

- Re-read project context, Stage 4A-6.3 notes, and existing Stage 4A-6.4
  scripts/outputs in the workspace.
- Confirmed the current task is calibrated/confidence-gated prediction gain,
  not RL, PPO, IL/BC training, SSCNet retraining, longer rollout scaling,
  prediction fusion into observed maps, or prediction-based traversability.
- Verified implementation in:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/calibrate_prediction_gain.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sc_gain_gating_ablation.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_sc_gain_gating.py`
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sc_gain_gating.py`
- Stage 4A-6.4 scoring now keeps raw `gain_sc` and `gain_hybrid` for
  regression visibility, and adds `effective_gain_sc`, `weighted_gain_sc`,
  `gain_hybrid_effective`, and `gain_hybrid_weighted` for selective scoring.
- Supported formulas are `raw_count`, `occupied_only`, `occupied_margin`,
  `confidence_weighted`, `entropy_weighted`, `calibrated_occupied`, and
  `novelty_discounted`.
- Ran/verified calibration output:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/calibration`
  with log
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a64_calibrate_prediction_gain.log`.
- Calibration result:
  sample count `11175`, occupied_prob weighted bin correlation
  `0.8699543518514645`, confidence weighted bin correlation
  `0.893222674245022`, recommended occupied/confidence thresholds
  `0.9 / 0.9`, calibrated_occupied usable `true`.
- Future observed maps were used only to estimate a fixed post-hoc reliability
  table. Runtime expert scoring did not read future observed maps.
- Verified one-step gating outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/one_step`
  for empty baseline, raw_count regression, occupied_only, occupied_margin,
  confidence_weighted, and calibrated_occupied.
- One-step result:
  best candidate id remained `11` for all cases; selective formulas reduced
  effective gain but did not change the one-step action.
- Verified sequential 5-step gating ablation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a64_gain_gating_ablation.log`
  output
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/ablation`.
- Completed configs:
  `occupied_only_occ07`, `occupied_only_occ08`,
  `occupied_margin_occ06_w05`, `confidence_weighted_conf05_cap30`;
  failed configs `[]`.
- Ran summary:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a64_gain_gating_summary.log`
  output
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/summary`.
- Summary result at the 5-step comparison horizon:
  empty final observed_ratio `0.06896296296296296`;
  fixed raw SC final observed_ratio `0.05899768518518519`;
  all completed gated configs final observed_ratio `0.05899768518518519`;
  all completed gated configs changed actions vs fixed raw SC `0/5`.
- Selectivity result:
  mean raw `gain_sc` stayed `49.4`, but mean `effective_gain_sc` became
  `4.2` for occupied_only_occ07, `3.2` for occupied_only_occ08,
  `1.7860426306724548` for occupied_margin_occ06_w05, and
  `36.19095666408539` for confidence_weighted_conf05_cap30.
- Ran py_compile:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a64_py_compile.log`
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a64_gain_gating_test.log`
- Validation passed:
  calibration outputs exist, one-step outputs exist, ablation manifest exists,
  four configs completed, observed_ratio is non-decreasing, raw and effective
  gains are logged, synthetic gain formulas are correct, weighted score formula
  is correct, `code_consistent_v1` is used, prediction is read-only, no
  prediction writeback/traversability/collision/A*/ray blocking, future
  observations are evaluation-only, no target/ground-truth leakage, no
  RL/PPO/BC/IL/optimizer training, and checkpoint was not modified.

Stage 4A-6.4 conclusion:

- Calibrated/confidence-gated scoring works mechanically and makes the SC term
  sparser, but it still does not change selected actions or improve measured
  coverage in the current 5-step medium_three_rooms seed/start.
- The remaining issue is rank sensitivity: even gated SC gain does not alter
  the candidate ordering enough against measured frontier gain and path cost.
- Next recommended work is candidate-level score/rank decomposition and
  spatial qualitative review of selected vs rejected candidates under gated
  formulas, still without RL/IL and without rollout scaling.

Stage 4A-6.5a candidate rank sensitivity diagnosis actions:

- Created one offline analysis script only:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/analyze_candidate_rank_sensitivity_small.py`.
- Loaded existing steps `0..4` from empty baseline, fixed raw SC, and the four
  completed Stage 4A-6.4 gated ablation episodes. No Isaac rollout, no
  counterfactual sweep, no spatial visualization, no scorer changes, and no
  training were run.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65a_rank_sensitivity`.
- Key result:
  selected gated candidate ids and positions are identical across all analyzed
  steps; top-1 remains stable vs fixed raw SC. Rank changes exist below top-1
  (mean top-5 Jaccard vs raw SC `0.9166666666666666`, mean top-16 Jaccard
  `0.869934640522876`).
- Component diagnosis:
  final_score vs inverse path_cost Pearson `0.8919154707376216`; final_score
  vs gain_exp Pearson `-0.46732096565152237`; final_score vs effective_gain_sc
  Pearson `0.03806071813182923`; gain_exp vs effective_gain_sc Pearson
  `0.17050928134932833`. Selected candidates have mean low-path-cost rank
  `1.0333333333333334`, mean gain_exp rank `14.4`, and mean effective_gain_sc
  rank `9.0`.
- Missing fields were reported instead of crashing: per-candidate ids are not
  stored in the step NPZ files; selected `best_candidate_id` is available from
  transitions.
- Validation:
  `py_compile` passed, summary JSON/MD exist, at least one config and step
  loaded, and missing fields were reported.

Stage 4A-6.5a conclusion:

- Candidate ranking is rank-insensitive because the selected candidate is
  almost always the lowest-cost candidate. Gating changes lower candidate ranks
  and reduces effective SC gain, but it does not move any gated alternative
  above the path-cost-dominant top-1.
- Recommended next small task is offline counterfactual score analysis if
  path-cost dominance is the focus. If candidate ids are needed, improve
  candidate logging only. If high-SC candidates need spatial interpretation,
  do spatial visualization only. Still not RL.

Stage 4A-6.5b offline counterfactual score analysis actions:

- Created one offline analysis script only:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_score_counterfactuals_small.py`.
- Loaded only existing Stage 4A-6.5a files:
  `candidate_rank_table.csv`, `selected_candidate_summary.csv`,
  `rank_correlation_summary.csv`, and
  `stage4a65a_rank_sensitivity_summary.json`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65b_counterfactual_scores`.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65b_counterfactual_scores.log`.
- Validation:
  `py_compile` passed; summary JSON/MD, action CSV/JSONL, formula summary,
  thresholds CSV, and skipped-formulas JSON exist; 6 configs, steps `0..4`,
  480 candidate rows, and 94 formula variants were analyzed; no rollout-like
  files were created in the output directory.

Stage 4A-6.5b conclusion:

- Any no-cost formula changes top-1: `exp_only_no_cost` changed 30/30 groups,
  and `cost_powered` with `alpha=0` changed all executable groups.
- Current over-cost formulas are stable: `exp_over_cost`,
  `raw_hybrid_over_cost`, and `effective_hybrid_over_cost` changed 0 groups.
- Reducing path-cost exponent matters: `alpha=0.5` changed vs `alpha=1` in
  80 grouped sweeps.
- `sc_only` changed 10/20 executable groups, indicating high-SC candidates are
  often different from current selected candidates.
- SC-specific lambda changes exist but are limited for decoupled scoring:
  global min SC-vs-lambda0 threshold `0.1`, median `0.5`; `decoupled_sc`
  changed one grouped sweep by `lambda <= 1` and more by larger lambda.
- Recommended next small task: if doing a formula smoke, keep it one-step only
  and use `decoupled_sc_lambda0p5`. If treating the no-cost changes as the main
  signal, inspect candidate generation / spatial placement instead. Still not
  RL.

Stage 4A-6.5d decoupled one-step spatial visualization actions:

- Created one offline visualization script only:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_decoupled_one_step_case.py`.
- Loaded the saved Stage 4A-6.5c case, observed_state, pose, prediction npz,
  one-step comparison JSON, decisions, and candidate JSONL files. No Isaac
  startup, rollout, map_predict rerun, scorer change, checkpoint edit,
  observed_state edit, prediction writeback, or training was run.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65d_decoupled_spatial_viz`.
- Generated:
  `observed_baseline_decoupled_topdown.png`,
  `prediction_overlay_topdown.png`, `candidate_score_components_topdown.png`,
  `baseline_vs_decoupled_local_zoom.png`,
  `stage4a65d_spatial_summary.json`, and
  `stage4a65d_spatial_summary.md`.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65d_spatial_viz.log`.
- Validation:
  `py_compile` passed; output dir exists; 4 PNGs exist; summary JSON/MD exist;
  observed_state was loaded read-only and its SHA-256 stayed unchanged; no
  rollout-like files were created.

Stage 4A-6.5d conclusion:

- Baseline top-1: `grid:15,16,11`, world `[-4.45,-4.35,1.15]`, score
  `168.01139097257172`, `gain_exp=53.0`, `effective_gain_sc=34.769909381866455`,
  `path_cost=0.49401412320638405`.
- Decoupled top-1: `grid:14,18,11`, world `[-4.55,-4.15,1.15]`, score
  `127.65878621375252`, `gain_exp=76.0`, `effective_gain_sc=56.02893930673599`,
  `path_cost=0.7627128432753507`.
- Displacement is `[-1,2,0]` cells / `0.22360679774997816 m`. The candidates
  are distinct logged candidates, but spatially adjacent. Decoupled moved
  toward higher gain/SC at higher cost, yet this still looks like local jitter
  rather than a new exploration branch.
- This is plausible for future one-step formula comparison, but not enough to
  justify rollout. Next recommended small task: candidate generation or
  path-level/tree utility diagnosis, still no rollout and no RL.

Stage 4A-6.5e offline candidate/path utility diagnosis actions:

- Created one offline diagnosis script only:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_candidate_generation_path_utility.py`.
- Loaded existing Stage 4A-6.5a rank tables, Stage 4A-6.5b counterfactual
  summaries, Stage 4A-6.5c one-step candidate JSONL/NPZ, Stage 4A-6.5d spatial
  summary, fixed SC episode step NPZs, empty baseline episode step NPZs, and
  Stage 4A-6.4 gated episode step NPZs. No Isaac startup, rollout,
  map_predict rerun, scorer change, checkpoint edit, observed_state edit,
  prediction writeback, or training was run.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65e_path_candidate_diagnosis`.
- Generated:
  `candidate_spread_summary.csv`,
  `candidate_rank_distance_summary.csv`,
  `selected_vs_high_gain_summary.csv`,
  `path_cost_dominance_summary.csv`,
  `path_level_proxy_summary.csv`,
  `stage4a65e_path_candidate_diagnosis_summary.json`,
  `stage4a65e_path_candidate_diagnosis_summary.md`, and four simple step001
  PNG plots.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65e_path_candidate_diagnosis.log`.
- Validation:
  `py_compile` passed; output dir exists; summary JSON/MD exist; 5 CSV files
  exist; no rollout-like files were created; selected observed_state SHA-256
  matched the recorded hash.

Stage 4A-6.5e conclusion:

- Full-64 runtime candidates are not all local: median candidate distance is
  `2.321601489576769 m`.
- Saved top-N candidate sets are much more local: median distance is
  `0.9196431793625341 m`.
- Selected candidates match the minimum path-cost candidate in `0.9375` of
  analyzed sets, and path-cost/inverse-cost is the strongest final-score
  component in `0.96875` of sets.
- High-gain alternatives are usually spatially different:
  selected-to-max-`gain_exp` median distance `1.7464247558356059 m`;
  selected-to-max-`effective_gain_sc` median distance
  `1.0630145543593017 m`.
- The Stage 4A-6.5c decoupled one-step change remains local jitter:
  baseline vs decoupled distance `0.22360679774997816 m`.
- The diagnostic 2-step proxy is not paper-equivalent and not a true
  counterfactual tree. Its fixed next-step estimate preserves the same top
  candidate in all computed cases, so it cannot solve the one-step locality
  problem.
- Recommended next small task: original SC-Explorer RRT/tree utility source
  code inspection. Still no rollout and no RL.

Stage 4A-6.5f original RRT/tree utility source-code inspection actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md` before doing source inspection.
- Created one static inspection script only:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/inspect_original_tree_utility.py`.
- Searched the tracked and local files under:
  `/home/ubuntu22/sc_explorer_ws/ssc_exploration`.
- Used git state to separate original tracked source evidence from untracked
  workspace additions, so previous local Python expert files were not treated
  as original SC-Explorer planner code.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65f_original_tree_utility_inspection`.
- Generated:
  `source_file_hits.csv`, `planner_file_candidates.txt`,
  `utility_formula_evidence.json`, `utility_formula_evidence.md`,
  `original_tree_utility_summary.json`, `original_tree_utility_summary.md`,
  `missing_or_ambiguous_items.md`, and
  `recommended_next_faithful_step.md`.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65f_original_tree_utility_inspection.log`.
- Validation:
  `py_compile` passed and was logged to
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65f_py_compile.log`; output dir
  and all required summary files exist; at least one planner candidate file was
  found; tracked `ssc_planning`, `ssc_mapping`, `ssc_msgs`, `README.md`, and
  `.rosinstall` had no source changes after inspection.

Stage 4A-6.5f conclusion:

- The local original source configures an RRT*/tree planner stack:
  `trajectory_generator.type: "RRTStar"`,
  `trajectory_evaluator.type: "RRTStarEvaluatorAdapter"`,
  `cost_computer.type: "SegmentTime"`,
  `value_computer.type: "GlobalNormalizedGain"`, and
  `next_selector.type: "SubsequentBest"`.
- The SC-specific code in this repo is primarily the map/evaluator integration:
  `SSCVoxbloxOccupancyMap`, `SSCExplorationEvaluator`,
  `SSCServer`, and the `SSCGrid` network bridge.
- `SSCExplorationEvaluator::computeGainFromVisibleVoxels` computes local
  trajectory-segment gain from ray-cast visible voxels, using measured ESDF
  observed checks plus SSC predicted occupied/free/unknown voxel types.
- In `sc_explorer.yaml`, SSC prediction is used for gain and collision
  fallback, while SSC information planning is disabled for ray blocking.
- Exact RRT node/tree fields, accumulated branch/path utility, exact
  `GlobalNormalizedGain` formula, and final best-node/best-branch/first-path
  action selection are not in the inspected repo; they live in external
  `mav_active_3d_planning` / `active_3d_planning_*` packages referenced by
  `.rosinstall` and `package.xml`.
- Recommended next small task: inspect/fetch the external active_3d_planning
  source manually, focusing on `RRTStar`, `RRTStarEvaluatorAdapter`,
  `SegmentTime`, `GlobalNormalizedGain`, `SubsequentBest`,
  `ContinuousYawPlanningEvaluator`, and `TrajectorySegment`. Still no rollout
  and no RL.

Stage 4A-6.5g external active_3d_planning source inspection actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md` before external source inspection.
- Checked local `.rosinstall`, recursive `package.xml` files, and planner
  configs. The active external dependency is
  `git@github.com:ethz-asl/mav_active_3d_planning.git`.
- Fetched/found the external source under:
  `/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning`
  at commit `11634e8325480ce5da36a78b23b917347c973613`.
- Created one static inspection script only:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/inspect_external_active_3d_planning.py`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65g_external_active3d_inspection`.
- Generated:
  `external_dependency_urls.json`, `external_source_inventory.csv`,
  `external_clone_status.json`, `external_source_hits.csv`,
  `planner_symbol_index.json`, `external_utility_formula_evidence.json`,
  `external_utility_formula_evidence.md`,
  `external_tree_utility_summary.json`,
  `external_tree_utility_summary.md`,
  `missing_or_ambiguous_external_items.md`,
  `recommended_next_faithful_step.md`, and `inspection_manifest.json`.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65g_external_active3d_inspection.log`.
- Validation:
  `py_compile` passed and was logged to
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65g_py_compile.log`; required
  outputs exist; external repo `git status --short` is clean; local
  `ssc_exploration` original planner/source files were not modified.

Stage 4A-6.5g conclusion:

- External source was available and contains the relevant packages:
  `active_3d_planning_core`, `active_3d_planning_ros`,
  `active_3d_planning_voxblox`, `active_3d_planning_mav`,
  `active_3d_planning_app_reconstruction`, and `mav_active_3d_planning`.
- `TrajectorySegment` is the tree node/edge object and stores trajectory
  points, `gain`, `cost`, `value`, `parent`, `children`, and `info`.
- `RRTStar` expands an RRT tree, computes gain for new segments, evaluates
  cost/value for candidate parent connections, rewires to the parent with
  highest value, and can preserve/reinsert branches after root changes.
- `SegmentTime` computes segment cost from trajectory timestamp duration,
  with optional parent-cost accumulation disabled by default in the inspected
  config path.
- `GlobalNormalizedGain` computes value as the maximum accumulated
  root-to-descendant `gain / cost` ratio in the segment subtree; no lambda,
  exponent, or discount appears in this class.
- `SubsequentBest` chooses the immediate child whose subtree contains the
  highest-value segment, and `OnlinePlanner` executes that child trajectory.
- `RRTStarEvaluatorAdapter` delegates gain/cost/value to the following
  evaluator and wraps next selection with `RRTStar::rewireRoot`.
- `ContinuousYawPlanningEvaluator` samples yaw orientations, selects the
  best FOV section by gain, sets trajectory yaw, and recomputes cost/value;
  `SegmentTime` itself does not directly price yaw rate/acceleration.
- SC-specific gain from `SSCExplorationEvaluator` enters as local
  `TrajectorySegment::gain`; measured ESDF and SSC prediction maps remain
  separate. In `sc_explorer.yaml`, SSC is enabled for collision fallback and
  gain classification, while SSC information planning is disabled for ray
  blocking.
- The current simulator expert is missing the real RRT tree, accumulated
  branch utility, `SubsequentBest`, root rewiring, and continuous yaw logic.
  Its path-cost/locality dominance is likely partly caused by collapsing the
  source planner to one-step local `gain / path_cost`.
- Recommended next small task: offline minimal tree-utility prototype over
  saved candidates, reproducing `GlobalNormalizedGain` and `SubsequentBest`.
  Still no rollout, no RL, no map_predict rerun, no training, and no planner
  implementation.

Stage 4A-6.5h offline tree utility prototype actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md` before implementing the prototype.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_tree_utility_prototype.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_offline_tree_utility_prototype.py`.
- The prototype reads saved Stage 4A-6.5a candidate rank tables, Stage 4A-6.5e
  diagnosis CSV fields, Stage 4A-6.5g source inspection outputs, and existing
  fixed-SC / empty-baseline transition logs only.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65h_offline_tree_utility_prototype`.
- Generated:
  `synthetic_tree_tests.json`, `synthetic_tree_tests.md`,
  `tree_formula_reference.md`, `loaded_candidate_fields.json`,
  `missing_fields_report.json`, `one_step_star_results.csv`,
  `recorded_episode_chain_results.csv`, `shallow_pseudo_tree_results.csv`,
  `subsequent_best_decisions.csv`, `tree_utility_comparison_summary.json`,
  `tree_utility_comparison_summary.md`, and
  `recommended_next_faithful_step.md`.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65h_offline_tree_utility_prototype.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65h_offline_tree_utility_test.log`,
  and `/home/ubuntu22/sc_explorer_ws/logs/stage4a65h_py_compile.log`.
- Validation:
  `py_compile` passed; test script passed; required outputs exist; synthetic
  tests pass; no rollout-like files were created in the output directory;
  external active_3d_planning repo `git status --short` is clean.

Stage 4A-6.5h conclusion:

- `OfflineSegment` now mirrors the source-level tree fields needed for this
  prototype: gain, cost, value, parent, children, metadata/info, and traceable
  candidate position fields.
- `GlobalNormalizedGain` is reproduced as the best accumulated
  root-to-descendant `gain / cost` value inside each segment subtree, skipping
  invalid near-zero accumulated costs with a warning.
- `SubsequentBest` is reproduced as selecting the root immediate child whose
  subtree contains the highest-value segment.
- Synthetic cases passed: low-cost trap selected `A` via descendant `A1`;
  subtree-does-not-help selected `B`; zero-cost safety emitted a warning and
  did not crash.
- Real saved data loaded 480 candidate rows over 6 configs and steps `0..4`;
  all candidate ids were surrogate ids because the saved table has empty
  `candidate_id`.
- `one_step_star` matched current one-step top-1 in `30/30` default trees and
  `30/30` runtime-like trees, confirming the star case degenerates to local
  ranking.
- Recorded chains for fixed raw SC and empty baseline show accumulated
  selected-path utility, but they are actual selected paths only and not
  counterfactual trees.
- Shallow pseudo-trees built 24 default-mode trees and changed
  `SubsequentBest` vs local one-step in `0/24`; the saved proxy children do
  not provide real branch alternatives.
- Recommended next small task: offline mini-RRT tree builder on saved observed
  map, still without Isaac, rollout, map_predict rerun, RL/IL, or planner
  implementation.
- No Isaac startup, rollout, new expert step, map_predict rerun, SSCNet
  inference/training, RL/PPO/BC/IL, checkpoint modification, observed_state
  modification, prediction writeback, target/ground-truth scoring, or external
  source build/modification occurred.

Stage 4A-6.5i offline mini-RRT tree actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md` before implementing the mini-RRT tree builder.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_mini_rrt_tree.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_offline_mini_rrt_tree.py`.
- The script reads the Stage 4A-6.5c selected case, fixed SC episode
  `observed_state_step001.npy`, pose, camera info, episode summary, and saved
  one-step comparison outputs only.
- Implemented real offline `MiniRRTSegment` parent/children tree expansion:
  reachable observed-free/frontier sampling, nearest-node steering, snapped
  traversable endpoints, measured-only line edge validity, per-node yaw
  sampling, measured-only `gain_exp` raycast, SegmentTime-like
  `segment_length_m / v_max` cost, `GlobalNormalizedGain`, and
  `SubsequentBest`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65i_offline_mini_rrt_tree`.
- Generated:
  `mini_rrt_tree_segments.jsonl`, `mini_rrt_tree_summary.json`,
  `mini_rrt_tree_summary.md`, `subsequent_best_decision.json`,
  `subsequent_best_decision.md`, `tree_vs_one_step_comparison.json`,
  `tree_vs_one_step_comparison.md`, `sampled_nodes.csv`,
  `rejected_samples.csv`, `gain_cost_value_table.csv`,
  `tree_formula_reference.md`, `missing_or_limited_features.md`,
  `recommended_next_faithful_step.md`, `mini_rrt_tree_topdown.png`,
  `selected_branch_topdown.png`, `gain_cost_scatter.png`, and
  `value_depth_histogram.png`.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65i_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65i_offline_mini_rrt_tree.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65i_offline_mini_rrt_tree_test.log`.
- Validation:
  `py_compile` passed; `test_offline_mini_rrt_tree.py` passed edge collision,
  UNKNOWN-not-traversable, parent/children, low-cost trap,
  deterministic-seed, required-output, and observed_state hash checks.

Stage 4A-6.5i conclusion:

- The mini-RRT accepted `255` non-root nodes (`256` total) on the saved fixed
  SC observed map; `100` samples were rejected (`41` edge through
  non-traversable/UNKNOWN, `59` target same as nearest).
- Root grid/world/yaw: `[13,13,11]`,
  `[-4.650000095367432,-4.650000095367432,1.2000000476837158]`,
  `0.38710316317995463`.
- `GlobalNormalizedGain` computed valid values with no warnings.
- `SubsequentBest` selected immediate child `n0140`, grid `[14,13,11]`,
  value `286.21642816261226`, accumulated gain/cost
  `32.0 / 0.11180350549906025`; the best descendant was also `n0140`.
- The selected child differs from the one-step baseline `grid:15,16,11` and
  decoupled `grid:14,18,11`, but it is only `0.11180350549906025 m` from root
  and equals the root-local best child. This run did not find a nonlocal
  high-gain branch and did not reduce local path-cost dominance.
- Recommended next small task: inspect gain/raycast or sampling strategy,
  still offline and still no rollout.
- No Isaac startup, rollout, online expert loop, map_predict rerun, SSCNet
  inference/training, RL/PPO/BC/IL, checkpoint modification, observed_state
  modification, prediction writeback, target/ground-truth scoring, or external
  source modification/build occurred.

Stage 4A-6.5j offline mini-RRT gain/raycast/sampling diagnosis actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md` before implementing the diagnosis.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_mini_rrt_gain_sampling.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_mini_rrt_gain_sampling_diagnosis.py`.
- The script reads existing Stage 4A-6.5i mini-RRT outputs, saved
  `observed_state_step001.npy`, camera info, selected case metadata, and
  external active_3d_planning source/inspection outputs only.
- Implemented segment length/cost/value diagnostics, selected node raycast
  audit, all-node visibility novelty/overlap analysis, sampling/steering
  diagnosis, offline filter/rerank sweeps, source anti-local mechanism search,
  and summary/recommendation files.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65j_gain_raycast_sampling_diagnosis`.
- Generated required files:
  `segment_length_cost_diagnosis.csv/json`,
  `selected_node_raycast_audit.json`,
  `selected_node_visible_voxels.csv`,
  `selected_node_visible_topdown.png`,
  `node_novelty_overlap_table.csv`,
  `novelty_rerank_summary.json/md`,
  `sampling_steering_diagnosis.csv/json`,
  `sampling_rejection_summary.md`,
  `offline_filter_rerank_table.csv`,
  `offline_filter_rerank_summary.json/md`,
  `source_anti_local_mechanisms.md`,
  `source_anti_local_hits.csv`,
  `stage4a65j_gain_raycast_sampling_summary.json/md`, and
  `recommended_next_faithful_step.md`, plus optional diagnostic plots.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65j_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65j_gain_raycast_sampling_diagnosis.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65j_gain_raycast_sampling_test.log`.
- Validation:
  `py_compile` passed; `test_mini_rrt_gain_sampling_diagnosis.py` passed;
  required outputs exist; selected gain recompute status present; novelty table
  contains the selected node; no rollout-like outputs were created;
  observed_state SHA-256 was unchanged; external source git status was
  unchanged.

Stage 4A-6.5j conclusion:

- Selected child `n0140` has logged/recomputed `gain_exp=32.0`, visible
  unknown `32`, visible free `44`, and visible occupied `0`.
- Root/parent overlap under recorded yaw is `0/32`; root overlap under the
  selected yaw is `16/32`. The recorded-yaw gain is therefore reproducible and
  not a root/parent duplicate-count mismatch.
- The immediate root-child collapse is explained by short-edge cost
  amplification: `32.0 / 0.11180350549906025 = 286.21642816261226`.
- Segment lengths are short overall: min `0.09999999999999964m`, median
  `0.141421356237309m`, p75 `0.20000000000000018m`, max
  `0.5656854249492381m`; local gain/cost vs inverse segment length correlation
  is `0.9282421006554769`.
- Sampling diagnostics show accepted `255`, rejected `100`;
  `target_same_as_nearest=59` and `edge_non_traversable_or_unknown=41`.
- Diagnostic filters show root-child min segment length `0.15m` or min root
  distance `0.25m` moves selection off `n0140`; root/parent novelty alone does
  not move the immediate root-child winner.
- External source evidence supports source-like minimum path/cropped segment
  controls, density control, root rewiring/reinsert, optional parent visible
  clearing, and continuous yaw. No mandatory root-visible overlap filtering or
  near-root gain discount was proven.
- Recommended next small task: offline mini-RRT minimum-edge-length variant,
  still no Isaac, no rollout, no online expert loop, no map_predict rerun, no
  SSCNet inference/training, no RL/PPO/BC/IL, no checkpoint modification, no
  observed_state modification, no prediction writeback, no target/ground-truth
  scoring, and no external source build/modification.

Stage 4A-6.5k offline mini-RRT minimum-edge-length variant actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md` before implementing the variant.
- Modified:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_mini_rrt_tree.py`.
- Added optional parameters with defaults preserving Stage 4A-6.5i behavior:
  `--min_edge_length_m 0.0`, `--min_root_child_length_m 0.0`,
  `--min_root_distance_m 0.0`, `--crop_min_length_m 0.0`,
  `--short_edge_policy allow`, `--density_radius_m 0.0`,
  `--max_nodes_per_density_radius 0`, and `--variant_name`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_mini_rrt_min_edge_variants.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_mini_rrt_min_edge_variants.py`.
- The runner executed 9 offline variants:
  `baseline_allow`, `reject_min_edge_0p15`, `reject_min_edge_0p25`,
  `reject_root_child_0p25`, `reject_root_distance_0p25`,
  `crop_min_length_0p15`, `crop_min_length_0p25`, `density_limited`, and
  `combined_source_like`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65k_min_edge_length_variant`.
- Generated aggregate files:
  `variants_manifest.jsonl`, `variants_summary.csv/json`,
  `variants_comparison.md`, `selected_child_comparison.csv`,
  `rejection_reason_comparison.csv`,
  `segment_length_distribution_by_variant.csv`,
  `gain_cost_correlation_by_variant.csv`,
  `source_min_length_reference.md`, `recommended_next_faithful_step.md`,
  `stage4a65k_min_edge_variant_summary.json/md`, plus four aggregate PNGs.
- Each variant subdirectory contains the required mini-RRT JSON/CSV/JSONL
  outputs and, with `--save_viz`, `variant_selected_branch_topdown.png`,
  `variant_tree_topdown.png`, and `variant_gain_cost_scatter.png`.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65k_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65k_min_edge_variants.log`, and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65k_min_edge_variants_test.log`.
- Also ran the original Stage 4A-6.5i regression:
  `test_offline_mini_rrt_tree.py` passed after the default-preserving change.

Stage 4A-6.5k conclusion:

- `baseline_allow` reproduced selected child `n0140`, value
  `286.21642816261226`, selected distance `0.11180350549906025m`, and
  selected grid `[14,13,11]`.
- All non-baseline variants moved off `n0140`.
- Nonlocal-branch variants by the Stage 4A-6.5k definition:
  `reject_root_child_0p25`, `reject_root_distance_0p25`, and
  `crop_min_length_0p25`.
- Best source-like variant selected for the next smoke:
  `crop_min_length_0p25`. It selected child `n0001` at grid `[18,12,11]`,
  distance `0.5123476174067144m`, with best descendant `n0249` at grid
  `[39,19,11]`, distance `2.6688013442592453m`, accumulated gain/cost
  `645.0 / 4.565369444959812`, accepted nodes `255`, rejected samples `916`,
  and median segment length `0.2999999999999998m`.
- `crop_min_length_0p15` moved off `n0140` but still selected a local child
  at `0.20615538536590458m`; `crop_min_length_0p25` was the first crop
  variant to find a clearly nonlocal branch.
- `density_limited` and `combined_source_like` reduced tiny-edge clustering
  but were too restrictive at radius `0.25m` / max nodes `1`, accepting only
  `86` nodes and selecting a local child at `0.287228140627606m`.
- Validation passed: required outputs exist, `baseline_allow` equals `n0140`,
  at least one min-edge/crop/density variant completed, child fields are
  present, no rollout-like outputs were created, observed_state SHA-256 stayed
  unchanged, external active_3d_planning git status stayed clean, and no
  prediction writeback occurred.
- Recommended next small task: no-prediction online one-step tree smoke using
  source-like crop/min-length settings, still no rollout and no RL.
- No Isaac startup, rollout, online expert loop, map_predict rerun, SSCNet
  inference/training, RL/PPO/BC/IL, checkpoint modification, observed_state
  modification, prediction writeback, prediction collision/traversability use,
  target/ground-truth scoring, or external source build/modification occurred.

Stage 4A-6.5l source-protected one-step tree smoke actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md` before implementing the smoke.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_source_protected_one_step_tree_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_source_protected_one_step_tree_smoke.py`.
- The runner wraps the existing saved-map mini-RRT builder with the
  Stage 4A-6.5k source-like crop/min-length profile:
  `source_like_crop_min_length_0p25`, `short_edge_policy=crop`,
  `crop_min_length_m=0.25`, `num_nodes=256`, `sample_mode=mixed`,
  `gain_mode=exp`, `path_cost_mode=segment_time`, and `num_yaw_samples=8`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65l_source_protected_one_step_tree_smoke`.
- Generated required files:
  `source_protected_tree_decision.json/md`,
  `source_protected_tree_segments.jsonl`,
  `source_protected_gain_cost_value_table.csv`,
  `source_protected_sampled_nodes.csv`,
  `source_protected_rejected_samples.csv`,
  `source_protection_checklist.json/md`,
  `tree_vs_baseline_comparison.json/md`,
  `one_step_tree_smoke_summary.json/md`,
  `recommended_next_faithful_step.md`, plus tree/branch/gain-cost PNGs.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65l_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65l_source_protected_one_step_tree_smoke.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65l_source_protected_one_step_tree_smoke_test.log`.

Stage 4A-6.5l conclusion:

- The source-protected one-step tree smoke ran successfully and exactly
  reproduced Stage 4A-6.5k `crop_min_length_0p25`.
- Selected child: `n0001`, grid `[18,12,11]`, world
  `[-4.15,-4.75,1.15]`, distance `0.5123476174067144m`.
- Best descendant: `n0249`, grid `[39,19,11]`, world
  `[-2.05,-4.05,1.15]`, distance `2.6688013442592453m`.
- Value: `141.28100864040323`; accumulated gain/cost:
  `645.0 / 4.565369444959812`; accepted nodes `255`; rejected samples `916`.
- Active source-like protections: crop/min-path-length at `0.25m` and
  8-sample continuous yaw approximation.
- Present but inactive: density limiting / `max_density_range`, because
  Stage 4A-6.5k showed radius `0.25m` / max nodes `1` was too restrictive;
  root rewiring/reinsert is represented only as a hook/checklist and is
  inactive in this one saved step.
- Intentionally inactive because source evidence was optional or missing:
  optional parent visible clearing, mandatory root-visible filtering, and
  near-root gain discount.
- The selected child avoids old short-edge winner `n0140`, differs from
  one-step baseline grid `[15,16,11]`, differs from decoupled grid
  `[14,18,11]`, and satisfies the Stage 4A-6.5k nonlocal branch definition.
- Validation passed: required outputs exist, crop profile active,
  selected/best grids match the reference crop variant, observed_state
  SHA-256 stayed unchanged, checkpoint SHA-256 stayed unchanged, external
  active_3d_planning git status stayed clean, no rollout-like outputs were
  created, no map_predict artifacts were created, and no prediction was used.
- Recommended next small task: no-prediction Isaac one-step capture + tree
  decision smoke, still no rollout.
- No Isaac startup, rollout, online multi-step loop, map_predict rerun,
  SSCNet inference/training, RL/PPO/BC/IL, checkpoint modification,
  observed_state modification, prediction writeback, prediction collision /
  traversability use, target/ground-truth scoring, or external source
  build/modification occurred.

Stage 4A-6.5m Isaac one-step capture + tree decision actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md` before implementing the Isaac one-step capture smoke.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_one_step_tree_capture_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_one_step_tree_capture_smoke.py`.
- The runner starts Isaac headless once, creates deterministic
  `medium_three_rooms` seed `0`, loads fixed episode `pose_001.json`, captures
  one RGB/depth frame, updates a copy of prior
  `observed_state_step000.npy`, and saves only the new output observed map
  `observed_state_isaac_capture_step001.npy`.
- The runner then invokes the source-protected mini-RRT tree decision on the
  new observed map using the Stage 4A-6.5l source-like profile:
  `short_edge_policy=crop`, `crop_min_length_m=0.25`, `num_nodes=256`,
  `max_extension_m=0.5`, `sample_mode=mixed`, `gain_mode=exp`,
  `path_cost_mode=segment_time`, `num_yaw_samples=8`, and `seed=0`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65m_isaac_one_step_tree_capture_smoke`.
- Generated required capture/report/tree files and plots, including
  `capture_rgb_001.png`, `capture_depth_001.npy/png`,
  `observed_state_isaac_capture_step001.npy`,
  `source_protected_tree_decision.json/md`,
  `source_protection_checklist.json/md`,
  `comparison_to_saved_tree_smoke.json/md`,
  `isaac_one_step_tree_capture_summary.json/md`, tree CSV/JSONL aliases, and
  the required topdown/scatter visualizations.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65m_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65m_isaac_one_step_tree_capture_smoke.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65m_isaac_one_step_tree_capture_smoke_test.log`.

Stage 4A-6.5m conclusion:

- Isaac one-frame capture succeeded at fixed pose `pose_001.json`: position
  `[-4.65,-4.65,1.2]`, yaw `0.38710316317995463`.
- Captured depth shape `(120,160)`, positive count `11772`, min/max
  `1.0084033012390137 / 4.979179859161377`; RGB was nonblank.
- Measured-only depth update succeeded. The prior
  `observed_state_step000.npy` hash stayed unchanged
  `3ea6746c8008633a810824407a3e5b4ab56b638c06aa4fa7c33baa3262b11c24`.
- New observed map:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65m_isaac_one_step_tree_capture_smoke/observed_state_isaac_capture_step001.npy`.
  Hash:
  `afacb32647bfa1b2ece34b75f19cb34cd15ec742b43e02b682af0fcd5f2bc59e`.
  It exactly matched saved fixed-episode `observed_state_step001.npy`.
- Source-protected tree decision succeeded and exactly matched Stage 4A-6.5l:
  selected child `n0001`, grid `[18,12,11]`, world
  `[-4.15,-4.75,1.15]`, distance `0.5123476174067144m`; best descendant
  `n0249`, grid `[39,19,11]`, world `[-2.05,-4.05,1.15]`, distance
  `2.6688013442592453m`; accumulated gain/cost
  `645.0 / 4.565369444959812`; value `141.28100864040323`; accepted nodes
  `255`; rejected samples `916`.
- Protection checklist: crop/min-path-length active at `0.25m`; density
  limiting implemented but inactive; continuous yaw approximation active with
  `8` fixed yaw samples; root rewiring/reinsert inactive; optional parent
  visible clearing and root-visible filtering / near-root discount inactive.
- Validation passed: `py_compile`,
  `test_isaac_one_step_tree_capture_smoke.py`, required outputs, prior hash,
  checkpoint hash, external source status, no prohibited rollout-like outputs,
  no map_predict artifacts, no prediction use, and no leakage flags.
- Recommended next small task: no-prediction two-frame tree smoke. Still not
  rollout.
- No selected action execution, online multi-step loop, rollout, map_predict
  rerun, SSCNet inference/training, RL/PPO/BC/IL, checkpoint modification,
  existing observed_state modification, prediction writeback, prediction
  collision/traversability use, target/ground-truth scoring, external source
  modification/build, or coverage-improvement claim occurred.

Stage 4A-6.5n two-frame tree smoke actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md` before implementing the no-prediction two-frame smoke.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_two_frame_tree_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_two_frame_tree_smoke.py`.
- The runner starts Isaac headless once, creates deterministic
  `medium_three_rooms` seed `0`, captures frame 1 at the Stage 4A-6.5m pose,
  updates a copy of prior `observed_state_step000.npy`, runs the
  source-protected no-prediction mini-RRT tree, moves once to the selected
  child x/y with fixed camera height `1.2m`, captures frame 2, updates from
  `observed_state_frame001.npy`, and runs the same tree decision again.
- The runner keeps prediction disabled: `prediction_npz=""`, `gain_mode=exp`,
  no map_predict rerun, no prediction writeback, and no prediction for
  collision/traversability.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65n_two_frame_tree_smoke`.
- Generated required frame capture files, observed maps, per-frame tree
  decisions, per-frame JSONL/CSV aliases, `two_frame_tree_summary.json/md`,
  `observed_ratio_two_frame.json`, `source_protection_checklist.json/md`,
  `frame001_tree_topdown.png`, `frame002_tree_topdown.png`,
  `two_frame_path_topdown.png`, and `recommended_next_faithful_step.md`.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65n_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65n_two_frame_tree_smoke.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65n_two_frame_tree_smoke_test.log`.

Stage 4A-6.5n conclusion:

- Frame 1 capture succeeded at position `[-4.65,-4.65,1.2]`, yaw
  `0.38710316317995463`; depth shape `(120,160)`, positive count `11772`,
  min/max `1.0084033012390137 / 4.979179859161377`.
- Frame 1 measured-only update added `3727` observed voxels and produced
  `observed_state_frame001.npy` hash
  `afacb32647bfa1b2ece34b75f19cb34cd15ec742b43e02b682af0fcd5f2bc59e`.
- Frame 1 tree decision exactly reproduced Stage 4A-6.5m / 6.5l: selected
  child `n0001`, grid `[18,12,11]`, world `[-4.15,-4.75,1.15]`; best
  descendant `n0249`, grid `[39,19,11]`, world `[-2.05,-4.05,1.15]`;
  accumulated gain/cost `645.0 / 4.565369444959812`; value
  `141.28100864040323`; accepted nodes `255`; rejected samples `916`.
- Exactly one action was executed: selected child `n0001` x/y
  `[-4.15,-4.75]`, fixed camera height `1.2m`, yaw `2.15879915042112`.
- Frame 2 capture succeeded at position `[-4.15,-4.75,1.2]`, yaw
  `2.15879915042112`; depth shape `(120,160)`, positive count `14997`,
  min/max `1.0084031820297241 / 4.885604381561279`.
- Frame 2 measured-only update added `6251` observed voxels and produced
  `observed_state_frame002.npy` hash
  `aeb1b990f783d2548c3f738fc5f4ba4ee922b8bda53fb32cf9b7818938c674a1`.
- Frame 2 tree decision succeeded and remained nonlocal: selected child
  `n0001`, grid `[17,16,11]`, world `[-4.25,-4.35,1.15]`, distance
  `0.502493918652551m`; best descendant `n0112`, grid `[8,27,11]`, world
  `[-5.15,-3.25,1.15]`, distance `1.4874475373705685m`; accumulated
  gain/cost `323.0 / 2.315392939101747`; value `139.50115962835548`;
  accepted nodes `255`; rejected samples `478`.
- Validation passed: `py_compile`, `test_isaac_two_frame_tree_smoke.py`,
  required outputs, active crop/min-length at `0.25m`, prediction disabled,
  no map_predict artifacts, no rollout manifest/plots, prior observed_state
  unchanged, frame1 observed_state unchanged during frame2 update, checkpoint
  unchanged, external source unchanged, selected action executed exactly once,
  no third frame, and no leakage flags.
- Recommended next small task: map_predict + source-protected tree one-step
  smoke. Still not rollout.
- No rollout, online open-ended loop, map_predict rerun during this stage,
  SSCNet inference/training, RL/PPO/BC/IL, checkpoint modification, existing
  observed_state modification, prediction writeback, prediction collision /
  traversability use, target/ground-truth scoring, external source
  modification/build, or coverage-improvement claim occurred.

Stage 4A-6.5o map_predict + source-protected tree one-step actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md` before implementing the map_predict tree one-step smoke.
- Confirmed Stage 4A-6.5n completed, the next task was map_predict +
  source-protected tree one-step smoke, and the hard boundaries remained:
  one frame only, no two-frame run, no rollout, no selected action execution,
  no training/RL/PPO/BC/IL, no checkpoint change, no existing observed_state
  modification, no prediction writeback, no prediction traversability /
  collision / ray blocking, no target/ground-truth scoring, and no external
  source build/modification.
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_mini_rrt_tree.py`
  to preserve default measured-only behavior while recording `gain_occ` and
  `gain_conf` for prediction-aware tree nodes and CSV rows.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_tree_one_step_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_map_predict_tree_one_step_smoke.py`.
- The runner starts Isaac headless once, creates deterministic
  `medium_three_rooms` seed `0`, captures exactly one frame at the Stage
  4A-6.5m/6.5n frame-1 pose, updates a copy of prior
  `observed_state_step000.npy`, runs map_predict once with
  `IsaacMapPredictor`, then runs the source-protected tree twice on the new
  observed map: measured-only baseline `gain_mode=exp` and SC-tree
  `prediction_mode=sim_dynamic`, `gain_mode=hybrid`.
- The runner uses the Stage 4A-6.5l source-like profile:
  `short_edge_policy=crop`, `crop_min_length_m=0.25`,
  `min_edge_length_m=0.0`, `min_root_child_length_m=0.0`,
  `min_root_distance_m=0.0`, `density_radius_m=0.0`,
  `max_nodes_per_density_radius=0`, `num_nodes=256`,
  `max_extension_m=0.5`, `sample_mode=mixed`,
  `path_cost_mode=segment_time`, `v_max=1.0`, `robot_radius_m=0.2`,
  `voxel_size=0.1`, `raycast_stride=2`, `num_yaw_samples=8`,
  `max_ray_length_m=4.8`, and `seed=0`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65o_map_predict_tree_one_step_smoke`.
- Generated required capture files, measured-only observed_state update
  summary, map_predict artifacts, prediction stats, measured-only and SC tree
  raw outputs, tree aliases, comparison report, source-protection checklist,
  topdown decision visualization, predicted-unmeasured visualization,
  `map_predict_tree_one_step_summary.json/md`, and
  `recommended_next_faithful_step.md`.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65o_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65o_offline_mini_rrt_tree_regression.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65o_source_protected_regression.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65o_map_predict_tree_one_step_smoke.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65o_map_predict_tree_one_step_smoke_test.log`.

Stage 4A-6.5o conclusion:

- Isaac one-frame capture succeeded at position `[-4.65,-4.65,1.2]`, yaw
  `0.38710316317995463`; depth shape `(120,160)`.
- Measured-only observed_state update succeeded and added `3727` observed
  voxels. New `observed_state_frame001.npy` hash:
  `afacb32647bfa1b2ece34b75f19cb34cd15ec742b43e02b682af0fcd5f2bc59e`.
- map_predict succeeded with checkpoint
  `/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar`,
  `alignment_convention=code_consistent_v1`, `tau=0.1`, and no dense
  `class_prob` save.
- Prediction layer was shape-aligned to observed_state `(120,120,30)`.
  Counts: `prediction_valid_count=57382`,
  `predicted_unmeasured_count=37537`,
  `predicted_occupied_count=16779`.
- Source-protected tree ran in `prediction_mode=sim_dynamic` through
  read-only `SimPredictionLayer` as information gain only.
- `gain_sc` was nonzero: `255/255` non-root tree nodes had `gain_sc > 0`;
  `gain_sc` min/mean/max `15.0 / 48.88235294117647 / 62.0`.
- `gain_hybrid = gain_exp + gain_sc` passed exactly with max error `0.0`.
- measured-only selected child: `n0001`, grid `[18,12,11]`;
  SC-tree selected child: `n0001`, grid `[18,12,11]`.
  The selected child did not differ from measured-only in this one-step smoke,
  so no spatially meaningful selected-child change was observed.
- SC-tree best descendant remained `n0249`, grid `[39,19,11]`;
  accumulated gain/cost `1258.0 / 4.565369444959812`; value
  `275.55491379794235`.
- Validation passed: `py_compile`, `test_map_predict_tree_one_step_smoke.py`,
  `test_offline_mini_rrt_tree.py`, and Stage 4A-6.5l
  `test_source_protected_one_step_tree_smoke.py` regression.
- Safety passed: one frame only, no selected action execution, no two-frame
  run, no rollout, checkpoint unchanged, external source unchanged, prior
  observed_state unchanged, new observed_state unchanged after map_predict and
  tree, no prediction writeback, no prediction traversability/collision/ray
  blocking, no target/ground-truth scoring, no SSCNet training, no RL/PPO/BC/IL,
  and no coverage-improvement claim.
- Recommended next small task: map_predict + source-protected tree two-frame
  smoke. Still not rollout.

Stage 4A-6.5p map_predict + source-protected tree two-frame actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md` before implementing the two-frame smoke.
- Confirmed Stage 4A-6.5n and Stage 4A-6.5o completed, and the hard
  boundaries remained: exactly two frames, exactly one selected action, no
  rollout/open-ended loop, no training/RL/PPO/BC/IL, no checkpoint change, no
  existing observed_state modification, no prediction writeback, no prediction
  traversability/collision/ray blocking, no target/ground-truth scoring, and
  no external source build/modification.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_tree_two_frame_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_map_predict_tree_two_frame_smoke.py`.
- The runner starts Isaac headless once, creates deterministic
  `medium_three_rooms` seed `0`, captures frame 1 at the Stage 4A-6.5n
  starting pose, updates a new measured-only observed map, runs map_predict,
  evaluates measured-only and SC/hybrid source-protected trees, executes
  exactly one move to the frame-1 SC selected child, captures frame 2, updates
  a new measured-only observed map, runs map_predict again, evaluates
  measured-only and SC/hybrid trees again, then stops.
- `IsaacMapPredictor` is instantiated once and reports two predicted steps.
  Prediction output is passed only as a read-only `SimPredictionLayer` to
  information-gain scoring; tree traversability, collision, and ray blocking
  still use only measured observed_state.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65p_map_predict_tree_two_frame_smoke`.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65p_map_predict_tree_two_frame_smoke.log`
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65p_map_predict_tree_two_frame_smoke_test.log`.

Stage 4A-6.5p conclusion:

- Isaac startup succeeded; exactly two frames were captured and exactly one
  selected-child action was executed.
- Frame 1 measured update added `3727` observed voxels. map_predict produced
  shape-aligned prediction `(120,120,30)` with
  `prediction_valid_count=57382`, `predicted_unmeasured_count=37537`, and
  `predicted_occupied_count=16779`.
- Frame 1 SC tree reproduced the current Stage 4A-6.5o artifact: selected
  child `n0001`, grid `[18,12,11]`; best descendant `n0249`, grid
  `[39,19,11]`; accumulated gain_exp/gain_sc/gain_hybrid/cost
  `645.0 / 638.0 / 1283.0 / 4.565369444959812`; value
  `281.02873501649196`. It did not change measured-only selected child.
- The single move used SC selected child `n0001`: pose
  `[-4.15,-4.75,1.2]`, yaw `2.15879915042112`.
- Frame 2 measured update added `6251` observed voxels. map_predict produced
  `prediction_valid_count=37258`, `predicted_unmeasured_count=26620`, and
  `predicted_occupied_count=6638`.
- Frame 2 measured-only tree matched Stage 4A-6.5n frame 2:
  selected child `n0001`, grid `[17,16,11]`; best descendant `n0112`, grid
  `[8,27,11]`; accumulated gain/cost `323.0 / 2.315392939101747`; value
  `139.50115962835548`.
- Frame 2 SC tree changed selected child and best descendant:
  selected child `n0127`, grid `[11,15,11]`; best descendant `n0162`, grid
  `[14,15,11]`; accumulated gain_exp/gain_sc/gain_hybrid/cost
  `76.0 / 75.0 / 151.0 / 0.5872281406276059`; value
  `257.14026551693735`.
- `gain_sc` was nonzero in both frames: frame 1 `255/255` nodes and frame 2
  `248/255` nodes had `gain_sc > 0`. gain_exp/gain_sc correlation was
  `0.9427480283392026` in frame 1 and `0.02431093087475427` in frame 2.
- Validation passed: `py_compile`, `test_map_predict_tree_two_frame_smoke.py`,
  required outputs, source-protection checklist, prediction-safety checklist,
  observed hash checks, gain_hybrid identity, no third frame, no rollout-like
  outputs, checkpoint unchanged, external source unchanged, and no leakage
  flags.
- Recommended next small task: controlled gated SC tree one-step smoke or a
  repeated two-frame smoke. Still not rollout.

Stage 4A-6.5q SC-tree branch-change diagnosis actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5p was complete.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_sc_tree_branch_change.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sc_tree_branch_change_diagnosis.py`.
- The diagnosis script reads only saved Stage 4A-6.5p artifacts and performs:
  branch-change decomposition, all-node rank sensitivity, gated replay on the
  saved frame2 SC tree, missing-field reporting, and diagnostic plots.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65q_sc_tree_branch_change_diagnosis`.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65q_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65q_sc_tree_branch_change_diagnosis.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65q_sc_tree_branch_change_test.log`.

Stage 4A-6.5q conclusion:

- Frame2 measured selected child was `n0001`, grid `[17,16,11]`; SC selected
  child was `n0127`, grid `[11,15,11]`; SC best descendant was `n0162`, grid
  `[14,15,11]`.
- The selected-child delta was `0.6082762530298217m`; the best-descendant
  delta was `1.3416407864998743m`. The change is a different immediate root
  child, but a local two-node branch rather than a long nonlocal branch.
- On the saved SC tree, the SC branch had lower exp-only value per cost than
  the measured branch (`129.42159059130623` vs `134.7503461425601`), but
  higher SC-only value per cost (`127.71867492563113` vs
  `121.79358209039086`) and much lower accumulated cost. Raw hybrid margin was
  narrow: `257.14026551693735` vs `256.543928232951`.
- Frame1 did not change selected child because gain_exp/gain_sc correlation
  was high (`0.9427480283392026`) and SC reinforced the measured branch.
  Frame2 changed because correlation dropped to `0.02431093087475427`.
- Gated replay: raw_count, weight `1.0`, cap `25`, cap `50`, and
  confidence-weighted gain kept `n0127`; weights `0.0`, `0.25`, `0.5`, and
  occupied-only returned to measured child `n0001`. Minimum SC weight for
  changing selected child was `0.899353934095411`.
- Validation passed: `py_compile` and
  `test_sc_tree_branch_change_diagnosis.py`.
- Safety passed: no Isaac startup, no map_predict rerun, no SSCNet inference,
  no rollout, no selected action execution, no training/RL, no checkpoint
  modification, no observed_state modification, no prediction writeback, no
  prediction traversability/collision/ray blocking, no target/ground-truth
  scoring, and no external source modification/build.
- Recommended next small task: gated SC tree one-step smoke. Still not
  rollout.

Stage 4A-6.5r gated SC tree one-step smoke actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5p and Stage 4A-6.5q were complete.
- Confirmed the hard boundary for this task: saved-frame one-step smoke only,
  no Isaac startup, no RGB/depth capture, no map_predict rerun, no SSCNet
  inference, no two-frame run, no selected action execution, no rollout, no
  training/RL/PPO/BC/IL, no checkpoint change, no observed_state modification,
  no prediction writeback, no prediction traversability/collision/ray
  blocking, no target/ground-truth scoring, and no external source
  modification/build.
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/offline_mini_rrt_tree.py`.
- The mini-RRT tree now records raw `gain_sc`, `gain_occ`, `gain_conf`,
  `effective_gain_sc`, and `gain_hybrid_effective`, and supports optional
  runtime `sc_gain_formula` values including `raw_count`, `weight_0p5`,
  `weight_1p0`, `cap25`, `cap50`, `confidence_weighted`,
  `occupied_only`, and `confidence_weighted_cap25`.
- Default behavior remains measured-only `gain_mode=exp` with prediction
  disabled unless SC/hybrid prediction scoring is explicitly requested.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_gated_sc_tree_one_step_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_gated_sc_tree_one_step_smoke.py`.
- The runner reads only Stage 4A-6.5p Frame 2 saved inputs:
  `observed_state_frame002.npy`,
  `frame002_prediction/global_prediction_layer.npz`,
  `frame002_pose.json`,
  `frame002_camera_info.json`,
  `frame002_measured_tree_decision.json`, and
  `frame002_sc_tree_decision.json`.
- The runner uses the same Stage 4A-6.5p source-protected mini-RRT profile
  and root convention, builds one tree per formula, writes per-formula
  decisions, gain/cost tables, tree segment JSONL, source/prediction safety
  reports, hash checks, formula comparison CSV/JSON, and visualizations.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65r_gated_sc_tree_one_step_smoke`.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_gated_sc_tree_one_step_smoke.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_gated_sc_tree_one_step_smoke_test.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_offline_mini_rrt_tree_regression.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_source_protected_one_step_regression.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_isaac_two_frame_tree_regression.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_map_predict_tree_one_step_regression.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_map_predict_tree_two_frame_regression.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65r_sc_tree_branch_change_regression.log`.

Stage 4A-6.5r conclusion:

- All expected gated runtime one-step selections passed.
- Measured-only selected `n0001`, grid `[17,16,11]`, best descendant `n0112`,
  value `139.50115962835548`.
- Raw-count reproduced Stage 4A-6.5p Frame 2: selected child `n0127`, grid
  `[11,15,11]`, best descendant `n0162`, value `257.14026551693735`,
  accumulated gain_exp/effective_gain_sc/cost
  `76.0 / 75.0 / 0.5872281406276059`.
- `weight_0p5` returned to measured `n0001`, with value
  `195.86308325529168`.
- `weight_1p0`, `cap25`, `cap50`, `confidence_weighted`, and
  `confidence_weighted_cap25` preserved `n0127`. For the winning branch
  `n0127 -> n0162`, cap `25` accumulated effective SC `50.0`; confidence
  weighted accumulated effective SC `31.506256222724915`.
- `occupied_only` returned to measured selected child `n0001`.
- Validation passed: `py_compile`, the new Stage 4A-6.5r test, offline
  mini-RRT regression, Stage 4A-6.5l source-protected regression, Stage
  4A-6.5n no-prediction two-frame regression, Stage 4A-6.5o map-predict
  one-step regression, Stage 4A-6.5p two-frame output regression, and Stage
  4A-6.5q branch-change regression.
- Safety passed: observed_state and prediction NPZ hashes unchanged,
  checkpoint hash unchanged, external source status unchanged, no prohibited
  rollout-like artifacts, and prediction remained read-only and
  information-gain-only.
- Recommended next small task: choose a conservative gated formula for a later
  staged smoke or repeat saved/two-frame validation if needed. Still not
  direct rollout.

Stage 4A-6.5s confidence-weighted / cap25 gated SC tree two-frame actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5r was complete.
- Confirmed the hard boundary for this task: exactly one Isaac headless
  startup, exactly two captured frames, exactly one selected-child action from
  the primary `confidence_weighted` formula, no third frame, no second action,
  no rollout, no open-ended loop, no RL/PPO/BC/IL/training, no checkpoint
  modification, no existing observed_state modification, no prediction
  writeback, no prediction traversability/collision/ray blocking, no
  target/ground-truth scoring, no external source modification/build, and no
  coverage-improvement claim.
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_tree_two_frame_smoke.py`
  so explicit gated `--sc_gain_formula` values are passed through to
  `offline_mini_rrt_tree.py`; default behavior remains `raw_count`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_gated_sc_tree_two_frame_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_gated_sc_tree_two_frame_smoke.py`.
- The 4A-6.5s runner invokes the Stage 4A-6.5p two-frame runner once with
  `--sc_gain_formula confidence_weighted`, then evaluates `cap25` as a
  saved-frame offline shadow on frame 1 and frame 2. The shadow tree never
  executes motion and never reruns Isaac or map_predict.
- Ran py_compile for:
  `run_map_predict_tree_two_frame_smoke.py`,
  `run_gated_sc_tree_two_frame_smoke.py`,
  `test_gated_sc_tree_two_frame_smoke.py`, and
  `offline_mini_rrt_tree.py`.
- Ran regressions:
  `test_gated_sc_tree_one_step_smoke.py` on Stage 4A-6.5r outputs and
  `test_map_predict_tree_two_frame_smoke.py` on Stage 4A-6.5p outputs.
- Ran the requested Isaac smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65s_gated_sc_tree_two_frame_smoke.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65s_gated_sc_tree_two_frame_smoke`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65s_gated_sc_tree_two_frame_smoke_test.log`.

Stage 4A-6.5s conclusion:

- Validation passed.
- Frame 1 capture / measured update / map_predict succeeded. Frame 1
  measured-only and `confidence_weighted` both selected child `n0001`, grid
  `[18,12,11]`, best descendant `n0249`, grid `[39,19,11]`.
- Frame 1 `confidence_weighted` winning path accumulated effective SC gain
  `240.77071565389633`, raw gain_sc `633.0`, and cost
  `4.565369444959812`.
- Frame 1 `cap25` shadow selected child `n0196`, grid `[15,11,11]`, best
  `n0196`; this shadow decision was not executed.
- Exactly one move was executed, using the Frame 1 `confidence_weighted`
  selected child `n0001`, pose `[-4.15,-4.75,1.2]`, yaw
  `2.15879915042112`.
- Frame 2 capture / measured update / map_predict succeeded. Frame 2
  measured-only selected `n0001`, grid `[17,16,11]`, best descendant `n0112`,
  matching Stage 4A-6.5n.
- Frame 2 `confidence_weighted` selected `n0127`, grid `[11,15,11]`, best
  descendant `n0162`, grid `[14,15,11]`, matching the Stage 4A-6.5p/6.5r SC
  branch.
- Frame 2 `cap25` shadow matched `confidence_weighted`: selected `n0127`,
  best `n0162`.
- Frame 2 winning branch accumulated raw gain_sc `75.0`, effective SC gain
  `31.506256222724915` for `confidence_weighted`, effective SC gain `50.0`
  for `cap25`, and cost `0.5872281406276059`.
- Prediction stats:
  frame 1 `prediction_valid_count=57382`,
  `predicted_unmeasured_count=37537`,
  `predicted_occupied_count=16779`;
  frame 2 `prediction_valid_count=37258`,
  `predicted_unmeasured_count=26620`,
  `predicted_occupied_count=6638`.
- Observed ratios:
  frame 1 `0.05104398148148148`; frame 2 `0.06551388888888889`.
- Safety passed: checkpoint unchanged, existing observed_state unchanged,
  prediction read-only and information-gain-only, no prediction
  traversability/collision/ray blocking, no target/ground-truth scoring, no
  external source modification/build, no rollout-like prohibited outputs, and
  no coverage-improvement claim.
- Recommended next small task: repeated gated two-frame smoke or short gated
  SC tree smoke if staged. Still not direct rollout.

Stage 4A-6.5t alternate-tree-seed gated SC tree two-frame actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5s was complete.
- Confirmed the hard boundary for this task: repeat Stage 4A-6.5s with only
  mini-RRT/tree seed changed to `1`; keep scene seed `0`, scene
  `medium_three_rooms`, primary executed formula `confidence_weighted`, cap25
  as shadow only, exactly two frames, exactly one selected-child action, no
  third frame, no second action, no rollout/open-ended loop, no
  RL/PPO/BC/IL/training, no checkpoint modification, no observed_state
  modification outside the new output dir, no prediction writeback, no
  prediction traversability/collision/ray blocking, no target/ground-truth
  scoring, no external source modification/build, and no coverage-improvement
  claim.
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_tree_two_frame_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_gated_sc_tree_two_frame_smoke.py`
  to record seed-qualified profile names while preserving seed-0 output
  behavior.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_gated_sc_tree_seed_repeat_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_gated_sc_tree_seed_repeat_smoke.py`.
- The Stage 4A-6.5t wrapper invokes the Stage 4A-6.5s runner once with
  `--seed 1`, `--scene_seed 0`, `--primary_sc_gain_formula
  confidence_weighted`, and `--shadow_sc_gain_formula cap25`, then writes a
  repeat/stability summary against the Stage 4A-6.5s seed-0 reference.
- The first attempt failed before capture because the headless Vulkan/NVIDIA
  environment was not set and Isaac exited with `GLXBadFBConfig`; reran the
  same command with `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`,
  `__GLX_VENDOR_LIBRARY_NAME=nvidia`, and unset display variables.
- Ran the requested Isaac smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65t_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke_test.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65t_stage4a65s_gated_sc_tree_two_frame_regression.log`.

Stage 4A-6.5t conclusion:

- Validation passed.
- The recorded profile name is `source_like_crop_min_length_0p25_seed1`.
- Isaac startup succeeded once on the successful run; exactly two frames were
  captured; map_predict ran exactly twice; exactly one action was executed
  from the `confidence_weighted` Frame 1 selected child; cap25 remained
  shadow-only.
- Frame 1 measured-only, `confidence_weighted`, and cap25 all selected
  `n0001`, grid `[18,12,11]`, best descendant `n0245`, grid `[33,14,11]`.
  Frame 1 `confidence_weighted` effective SC gain was
  `148.99908256530762`, cap25 effective SC gain was `175.0`, raw gain_sc was
  `368.0`, and cost was `2.4879349937152315`.
- Frame 1 executed move remained `n0001`, pose `[-4.15,-4.75,1.2]`, yaw
  `2.15879915042112`.
- Frame 2 measured-only selected `n0057`, grid `[12,16,11]`, best `n0118`,
  grid `[12,19,11]`. Frame 2 `confidence_weighted` also selected
  `n0057 -> n0118`; cap25 shadow matched it.
- Frame 2 did not select the exact Stage 4A-6.5s `n0127 -> n0162` id branch,
  but it was spatially close to that reference branch: selected-child delta
  `0.14142135623730964m`, best-descendant delta `0.4472135954999583m`.
  Because the selected branch returned to measured-only under seed `1`, this
  is not enough to extend to a 3-frame gated smoke.
- Frame 2 `confidence_weighted` effective SC gain was `35.51751762628555`,
  cap25 effective SC gain was `50.0`, raw gain_sc was `82.0`, and cost was
  `0.620156278894175`.
- Prediction stats matched Stage 4A-6.5s: frame 1
  `prediction_valid_count=57382`, `predicted_unmeasured_count=37537`,
  `predicted_occupied_count=16779`; frame 2
  `prediction_valid_count=37258`, `predicted_unmeasured_count=26620`,
  `predicted_occupied_count=6638`.
- Safety passed: checkpoint unchanged, existing observed_state unchanged,
  prediction read-only and information-gain-only, no prediction
  traversability/collision/ray blocking, no target/ground-truth scoring, no
  external source modification/build, no rollout-like prohibited outputs, and
  no coverage-improvement claim.
- Recommended next small task: seed robustness diagnosis before any longer
  smoke. Still not rollout.

Stage 4A-6.5u seed robustness diagnosis actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5s seed0 plus Stage 4A-6.5t seed1 were
  complete.
- Confirmed the hard boundary for this task: offline diagnosis only; no Isaac
  startup, no new RGB/depth capture, no map_predict rerun, no SSCNet
  inference, no action execution, no two-frame/rollout execution, no training
  or RL, no checkpoint/observed_state modification, no prediction writeback,
  no prediction traversability/collision/ray blocking, no target/ground-truth
  scoring, no external source modification/build, and no coverage-improvement
  claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_gated_tree_seed_robustness.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_gated_tree_seed_robustness_diagnosis.py`.
- The diagnosis script reads saved Stage 4A-6.5s/6.5t decision JSON, tree
  segments, gain/cost/value CSVs, observed_state hashes, prediction hashes,
  and checkpoint hash. It writes decision comparison, top-K branch spatial
  matching, rank/margin summaries, branch classification, five diagnostic
  plots, missing-fields report, safety summary, and recommended next step.
- Ran py_compile for the new scripts.
- Ran:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65u_seed_robustness_diagnosis.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65u_seed_robustness_diagnosis`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65u_seed_robustness_diagnosis_test.log`.

Stage 4A-6.5u conclusion:

- Validation passed.
- Seed1 Frame 2 `confidence_weighted` truly returned to measured-only in
  seed1 score/tree space: measured and confidence both selected
  `n0057 -> n0118`, and measured rank under confidence scoring was `1`.
- Seed1 remained spatially close to the seed0 SC branch
  `n0127 -> n0162`: selected-child delta `0.14142135623730964m`,
  best-descendant delta `0.4472135954999583m`.
- Top-K confidence branch clouds overlapped; the nearest seed1 confidence
  branch to seed0 `n0127 -> n0162` was seed1 `n0057 -> n0118`, rank `1`.
- Branch classification: seed0 confidence is `spatially_same_as_seed0_sc`;
  seed1 confidence and seed1 cap25 are both `same_as_measured` and
  `spatially_same_as_seed0_sc`.
- Rank/margin: seed0 confidence winner margin was `2.3578116606364574`
  (normalized `0.012879002637364282`, narrow); seed1 confidence winner margin
  was `31.643506691543223` (normalized `0.1563440683986311`). Effective SC was
  decisive for seed0 but not across both seeds; cost dominance was not
  supported by the saved correlations.
- Recommended next small task: multi-seed offline replay / seed robustness
  sweep before any longer smoke. Still not 3-frame smoke and still not rollout.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no action execution, no rollout, no training/RL, no
  checkpoint/observed_state modification, no prediction writeback, no
  prediction traversability/collision/ray blocking, no target/ground-truth
  scoring, no external source modification/build, and no coverage-improvement
  claim.

Stage 4A-6.5v multi-seed offline replay / seed robustness sweep actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5s seed0, Stage 4A-6.5t seed1, and
  Stage 4A-6.5u diagnosis were complete.
- Confirmed the hard boundary for this task: offline replay only; no Isaac
  startup, no new RGB/depth capture, no map_predict rerun, no SSCNet
  inference, no action execution, no two-frame/rollout execution, no training
  or RL, no checkpoint/observed_state/prediction NPZ modification, no
  prediction writeback, no prediction traversability/collision/ray blocking,
  no target/ground-truth scoring, no external source modification/build, and no
  coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_multi_seed_gated_tree_replay.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_multi_seed_gated_tree_replay.py`.
- The replay script reads the saved Stage 4A-6.5p Frame 2 observed_state and
  prediction NPZ, plus the saved source-protected replay context from the
  Stage 4A-6.5s reference summary so the offline root matches the existing
  Stage 4A-6.5p/6.5s artifacts.
- Ran `py_compile` for the new scripts and `offline_mini_rrt_tree.py`.
- Ran the requested 10-seed offline replay:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65v_multi_seed_gated_tree_replay.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65v_multi_seed_offline_replay`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65v_multi_seed_gated_tree_replay_test.log`.

Stage 4A-6.5v conclusion:

- Validation passed.
- The sweep completed 10 tree seeds (`0..9`) and 40 seed/formula replays for
  `measured_only`, `confidence_weighted`, `cap25`, and `raw_count`.
- Seed 0 confidence replay exactly matched Stage 4A-6.5s:
  `n0127 -> n0162`. Seed 1 confidence replay exactly matched Stage 4A-6.5t:
  `n0057 -> n0118`.
- `confidence_weighted` across seeds: exact seed0 SC fraction `0.1`, spatial
  seed0 SC basin fraction `0.3`, same-as-measured fraction `0.7`,
  measured-but-seed0-SC-basin fraction `0.1`, distinct SC branch fraction
  `0.1`, and local jitter `0.0`.
- `cap25` spatial seed0 SC basin fraction was `0.5`; confidence/cap25 exact
  selected-child agreement was `0.8`; confidence-vs-measured and
  cap25-vs-measured exact agreement were both `0.7`.
- `raw_count` was not more aggressive by the sweep summary; its spatial
  seed0 SC basin fraction and same-as-measured fraction matched
  `confidence_weighted` (`0.3` and `0.7`).
- Confidence normalized margin distribution: min `0.012879002637364282`,
  median `0.12317906042561935`, max `0.22865634393797724`; narrow seeds
  (`<0.02`) were `[0,2,8]`, so margins were not mostly narrow.
- Value/effective-SC correlation was high on average
  (`0.7175741643849938`), while value/inverse-cost correlation did not support
  cost dominance (mean `0.29905939149745453`, max `0.47222953967781184`).
- Recommendation: tree sampling stabilization or SC gain design review before
  any 3-frame smoke or another start/scene two-frame smoke. Still not rollout.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no rollout/open-ended loop,
  no training/RL, no checkpoint/observed_state/prediction NPZ modification, no
  prediction writeback, no prediction traversability/collision/ray blocking,
  no target/ground-truth scoring, no external source modification/build, and
  no coverage-improvement claim.

Stage 4A-6.5w source-faithful RRTStar rewire/persistence actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5v was complete and not ready for
  3-frame smoke or rollout.
- Confirmed hard boundary for this task: offline replay only; no Isaac
  startup, no new RGB/depth capture, no map_predict rerun, no SSCNet
  inference, no action execution, no runtime two-frame loop, no rollout, no
  training/RL/PPO/BC/IL, no checkpoint/observed_state/prediction NPZ
  modification, no prediction writeback, no prediction
  traversability/collision/ray blocking, no target/ground-truth scoring, no
  external source modification/build, and no coverage-improvement claim.
- Inspected external active_3d_planning source at:
  `/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning`
  commit `11634e8325480ce5da36a78b23b917347c973613`.
- Source evidence recorded:
  `RRTStar::rewireRoot`, `rewireRootSingle`, `rewireToBestParent`,
  `RRTStarEvaluatorAdapter::selectNextBest`, `TrajectorySegment`,
  `SegmentTime`, `GlobalNormalizedGain`, `SubsequentBest`,
  `ContinuousYawPlanningEvaluator`, `crop_min_length`, `min_path_length`,
  `max_density_range`, and source configs with `max_density_range: 1.0`.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_source_faithful_rewire_persistence.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_source_faithful_rewire_persistence.py`.
- The runner reads saved Stage 4A-6.5p Frame1/Frame2 observed_state,
  prediction NPZ, pose, and camera info files. Fresh baseline rows are read
  from Stage 4A-6.5v saved replay to prove reproduction. Persistent configs
  build a Frame1 mini-RRT tree, resolve the executed Frame2 root near
  `[18,12,11]`, make that node the new root, preserve descendants whose
  measured-only Frame2 edges remain traversable, approximately reinsert
  non-descendant old branches, recompute Frame2 local gain/cost/value, rerun
  `GlobalNormalizedGain`/`SubsequentBest`, and continue expanding to target
  node count.
- Ran `py_compile` for:
  `run_source_faithful_rewire_persistence.py`,
  `test_source_faithful_rewire_persistence.py`, and
  `offline_mini_rrt_tree.py`.
- Ran a one-seed smoke in:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65w_smoke_tmp`
  and validated it.
- Ran the requested full offline sweep:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65w_source_faithful_rewire_persistence.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65w_source_faithful_rewire_persistence`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65w_source_faithful_rewire_persistence_test.log`.

Stage 4A-6.5w conclusion:

- Validation passed.
- Completed 200 config/seed/formula decision rows:
  10 seeds (`0..9`), formulas `measured_only`, `confidence_weighted`,
  `cap25`, and `raw_count`, configs `fresh_random_256_baseline`,
  `persistent_rewire_256_no_density`, `persistent_rewire_512_no_density`,
  `persistent_rewire_256_source_density`, and
  `persistent_rewire_512_source_density`.
- Fresh baseline reproduced Stage 4A-6.5v: confidence spatial seed0 SC basin
  `0.3`, same-as-measured `0.7`, cap25 spatial seed0 SC basin `0.5`, and
  confidence/cap25 agreement `0.8`.
- `persistent_rewire_256_no_density`: confidence spatial seed0 SC basin
  `0.0`, same-as-measured `0.5`, confidence/cap25 agreement `0.8`.
  It preserved many branches: mean preserved/pruned/reinserted/new nodes
  `169.35 / 8.5 / 56.5 / 29.15`, preserved-subtree winner fraction `0.875`,
  newly-expanded selected fraction `0.1`, and confidence normalized margin
  min/median/max `0.02783581652828007 / 0.06829388113415004 /
  0.1961893002847853`.
- `persistent_rewire_512_no_density`: confidence spatial seed0 SC basin
  `0.0`, same-as-measured `0.7`, confidence/cap25 agreement `0.8`.
  Mean preserved/pruned/reinserted/new nodes
  `169.35 / 8.5 / 56.5 / 192.075`, preserved-subtree winner fraction `0.85`,
  newly-expanded selected fraction `0.125`, and confidence normalized margin
  min/median/max `0.004792637771781909 / 0.04041557135583554 /
  0.16734846406225848`.
- Source density evidence found exact config values (`max_density_range:
  1.0`), but the source-like density diagnostic was too restrictive for this
  mini-RRT/profile: 256/512 source-density configs had no valid preserved
  roots or winners and spatial seed0 SC basin `0.0`.
- Reinsert approximation was implemented and recorded `106` successful
  branch-root reinsert attempts. Exact C++ KD-tree parent rewiring, ESDF
  collision, ROS lifecycle, source ownership model, and full continuous-yaw
  orientation sections remain approximate/not implemented.
- Recommendation: SC gain design review before another runtime smoke. Still
  not rollout.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no rollout/open-ended loop,
  no training/RL, no checkpoint/observed_state/prediction NPZ modification, no
  prediction writeback, no prediction traversability/collision/ray blocking,
  no target/ground-truth scoring, no external source modification/build, and
  no coverage-improvement claim.

Stage 4A-6.5x source-faithful SC gain design review actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5v/6.5w were complete and not ready for
  runtime smoke or rollout.
- Confirmed hard boundary for this task: offline analysis only; no Isaac
  startup, no new RGB/depth capture, no map_predict rerun, no SSCNet
  inference, no action execution, no two-frame/rollout execution, no
  training/RL/PPO/BC/IL, no checkpoint/observed_state/prediction NPZ
  modification, no prediction writeback, no prediction
  traversability/collision/ray blocking, no target/ground-truth scoring, no
  external source modification/build, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/review_source_sc_gain_design.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_source_sc_gain_design_review.py`.
- Inspected local source:
  `SSCExplorationEvaluator`, `SSCVoxelEvaluator`, `SSCVoxbloxOccupancyMap`,
  `SSCVoxbloxCriteriaMap`, `sc_explorer.yaml`, `baseline.yaml`, and the
  current simulator gain implementations in `offline_mini_rrt_tree.py`,
  `sim_paper_expert.py`, and `sim_prediction_layer.py`.
- Inspected external active_3d_planning source for `SimulatedSensorEvaluator`,
  `IterativeRayCaster`, `RRTStar`, `GlobalNormalizedGain`, and
  `SubsequentBest`.
- Loaded saved Stage 4A-6.5p Frame 2 observed_state and prediction NPZ plus
  saved Stage 4A-6.5p/6.5r/6.5v/6.5w tree/gain/decision artifacts.
- Recomputed visible voxel sets diagnostically because saved Python tree
  artifacts do not store visible voxel ids; prediction was not used for
  ray blocking.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65x_sc_gain_design_review`.
- Ran `py_compile` for:
  `review_source_sc_gain_design.py`,
  `test_source_sc_gain_design_review.py`, and `offline_mini_rrt_tree.py`.
- Ran the review:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65x_sc_gain_design_review.log`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65x_sc_gain_design_review_test.log`.

Stage 4A-6.5x conclusion:

- Validation passed.
- Source SC gain evidence: the active `sc_explorer.yaml` profile rewards
  predicted occupied and predicted free voxels with weights `1.0 / 1.0`,
  sets `unobserved_weight: 0.0`, uses `ssc_confidence_threshold: 0.05`, and
  does not enable direct `weight_by_confidence`.
- Source raycast evidence: `use_ssc_information_planning: false`, so
  prediction does not block visibility raycasting in the source profile.
  Parent-visible clearing is supported by the base evaluator but inactive in
  `sc_explorer.yaml`.
- Current formula audit: no current formula name is exactly source-faithful.
  `raw_count` is source-inspired unless restricted to source OCC/FREE
  threshold semantics. `confidence_weighted`, `cap25`, `occupied_only`,
  `occupied_margin`, `calibrated_occupied`, `novelty_discounted`, and branch
  normalization are diagnostic or source-inspired only.
- Key decomposition:
  seed0 measured `n0001 -> n0112` recorded `gain_exp=323.0`, raw/source
  OCC+FREE SC `593.0 / 569.0`, parent/root-cleared source SC `315.0`,
  frontier-local source SC `512.0`, cost `2.315392939101747`;
  seed0 SC `n0127 -> n0162` recorded `gain_exp=76.0`, raw/source OCC+FREE SC
  `136.0 / 135.0`, parent/root-cleared source SC `102.0`, frontier-local
  source SC `114.0`, cost `0.5872281406276059`;
  seed1 confidence/measured branch `n0057 -> n0118` recorded
  `gain_exp=90.0`, raw/source OCC+FREE SC `153.0 / 153.0`, cost
  `0.620156278894175`.
- Interpretation: seed0 SC is not winning because source-unknown prediction
  voxels dominate; raw and source OCC/FREE are nearly identical for that
  branch. The issue is a short low-cost local branch with low-novelty/root
  overlap prediction visibility, while the measured branch sees much more
  total source OCC/FREE prediction but pays higher cost.
- Candidate variant proxy: source OCC+FREE and source-thresholded OCC/FREE
  select the measured branch in the seed0 proxy; parent-visible-cleared and
  spatial-normalized diagnostics can still keep the short SC branch because of
  low cost; frontier-local selects measured in the seed0 proxy.
- Recommended next small task: offline source OCC+FREE plus
  parent-visible-cleared/frontier-local seed replay over saved Frame2
  artifacts. Still no runtime smoke and no rollout.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no rollout/open-ended loop,
  no training/RL, no checkpoint/observed_state/prediction NPZ modification, no
  prediction writeback, no prediction traversability/collision/ray blocking,
  no target/ground-truth scoring, no external source modification/build, and
  no coverage-improvement claim.

Stage 4A-6.5y source gain seed replay actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5v/6.5w/6.5x were complete and not
  ready for runtime smoke or rollout.
- Confirmed hard boundary for this task: offline seed replay only; no Isaac
  startup, no new RGB/depth capture, no map_predict rerun, no SSCNet
  inference, no action execution, no runtime two-frame loop, no rollout, no
  training/RL/PPO/BC/IL, no checkpoint/observed_state/prediction NPZ
  modification, no prediction writeback, no prediction
  traversability/collision/ray blocking, no target/ground-truth scoring, no
  external source modification/build, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_source_gain_seed_replay.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_source_gain_seed_replay.py`.
- The runner reads saved Stage 4A-6.5p Frame2 observed_state, prediction NPZ,
  pose, and camera info. It replays the current mini-RRT formulas, then
  post-hoc rescores explicit source OCC+FREE / visible-clearing /
  frontier-local variants on the saved raw-count tree without changing
  `offline_mini_rrt_tree.py` default behavior and without using prediction for
  edge validity, traversability, collision, or ray blocking.
- Wrote prediction field inventory and source OCC/FREE mapping reports. The
  mapping is recorded as `source-faithful-approx`: the NPZ contains
  `global_pred_class`, `global_confidence`, `global_free_prob`,
  `global_occupied_prob`, and `global_prediction_valid`, but it is not the
  exact C++ SSCMap log-odds layer.
- Ran `py_compile` for:
  `run_source_gain_seed_replay.py`,
  `test_source_gain_seed_replay.py`, and `offline_mini_rrt_tree.py`.
- Ran a one-seed smoke output in:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65y_source_gain_seed_replay_smoke_tmp`.
- Ran the requested full offline replay:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65y_source_gain_seed_replay.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65y_source_gain_seed_replay`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65y_source_gain_seed_replay_test.log`.

Stage 4A-6.5y conclusion:

- Validation passed.
- Completed 110 decision rows: 10 seeds (`0..9`) times 11 formulas:
  `measured_only`, `current_confidence_weighted`, `current_cap25`,
  `current_raw_count`, `source_occ_free`, `source_occ_free_thresholded`,
  `parent_visible_cleared_source_occ_free`,
  `root_visible_cleared_source_occ_free`,
  `frontier_local_source_occ_free`,
  `parent_cleared_frontier_local_source_occ_free`, and
  `branch_normalized_source_occ_free`.
- Required output files/plots all exist (`43` required outputs checked).
- Seed0 `current_confidence_weighted` reproduced Stage 4A-6.5s:
  `n0127 -> n0162`.
- Seed0 `source_occ_free` and `source_occ_free_thresholded` also selected
  `n0127 -> n0162`; source OCC+FREE alone did not select measured in this
  full-tree replay.
- Multi-seed fractions:
  `current_confidence_weighted` spatial seed0 SC basin `0.3`,
  same-as-measured `0.7`; `source_occ_free` spatial seed0 SC basin `0.4`,
  same-as-measured `0.6`; `parent_visible_cleared_source_occ_free` spatial
  seed0 SC basin `0.5`, same-as-measured `0.7`;
  `root_visible_cleared_source_occ_free` spatial seed0 SC basin `0.4`,
  same-as-measured `0.6`; `frontier_local_source_occ_free` spatial seed0 SC
  basin `0.4`, same-as-measured `0.6`;
  `parent_cleared_frontier_local_source_occ_free` spatial seed0 SC basin
  `0.5`, same-as-measured `0.7`; `branch_normalized_source_occ_free` spatial
  seed0 SC basin `0.5`, same-as-measured `0.7`.
- Interpretation: source OCC+FREE and the tested source-inspired filters did
  not robustly remove the seed0 SC basin. The short low-cost branch remains a
  ranking problem in this offline tree setup.
- Recommended next faithful step: inspect source OCC/FREE mapping and
  source-inspired novelty filters offline before any runtime smoke. Runtime
  smoke is still not ready, and rollout is still not ready.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no runtime two-frame loop,
  no rollout/open-ended loop, no training/RL/PPO/BC/IL, no
  checkpoint/observed_state/prediction NPZ modification, no prediction
  writeback, no prediction traversability/collision/ray blocking, no
  target/ground-truth scoring, no external source modification/build, and no
  coverage-improvement claim.

Stage 4A-6.5z decoupled SC utility sweep actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5x/6.5y were complete and not ready for
  runtime smoke or rollout.
- Confirmed hard boundary for this task: offline utility sweep only; no Isaac
  startup, no new RGB/depth capture, no map_predict rerun, no SSCNet
  inference, no action execution, no runtime two-frame loop, no rollout, no
  training/RL/PPO/BC/IL, no checkpoint/observed_state/prediction NPZ
  modification, no prediction writeback, no prediction
  traversability/collision/ray blocking, no target/ground-truth scoring, no
  external source modification/build, no Pareto gate/new runtime planner, and
  no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_decoupled_sc_utility_sweep.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_decoupled_sc_utility_sweep.py`.
- The runner reads saved Stage 4A-6.5p Frame2 observed_state, prediction NPZ,
  pose/camera info, and saved Stage 4A-6.5y `current_raw_count` mini-RRT trees.
  It recomputes path-level SC bonuses for `source_occ_free`,
  `parent_visible_cleared_source_occ_free`, and
  `frontier_local_source_occ_free`, then scores each path with
  `gain_exp / cost + lambda * normalized_sc`.
- Fixed lambdas were exactly `0,1,2,4,8,12,16,24,32`.
- Adaptive lambdas used
  `lambda_base = p90(base_exp_value) - p50(base_exp_value)` with variants
  `0.25x`, `0.5x`, `1.0x`, and `2.0x`.
- Ran `py_compile` for:
  `run_decoupled_sc_utility_sweep.py` and
  `test_decoupled_sc_utility_sweep.py`.
- Ran the requested full offline sweep:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65z_decoupled_sc_utility_sweep.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65z_decoupled_sc_utility_sweep`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65z_decoupled_sc_utility_sweep_test.log`.

Stage 4A-6.5z conclusion:

- Validation passed.
- Completed 390 decision rows: 10 seeds (`0..9`) times 3 SC bases times 13
  lambda variants.
- Required outputs/plots all exist (`33` required outputs checked).
- Seed0 base value gap was confirmed:
  measured `323.0 / 2.315392939101747 = 139.50115962835548`;
  seed0 SC `76.0 / 0.5872281406276059 = 129.42159059130623`;
  measured-minus-SC gap `10.079569037049254`.
- The Stage 4A-6.5x context source OCC+FREE values are recorded as measured
  `569.0` and seed0 SC `135.0`. The Stage 4A-6.5y saved posthoc row fields
  are recorded separately in the gap report because this sweep reads saved raw
  tree artifacts.
- For all tested fixed lambdas and all three SC bases, spatial seed0 SC basin
  fraction was `0.0` and same-as-measured fraction was `1.0`.
- For all adaptive lambda variants and all three SC bases, spatial seed0 SC
  basin fraction was also `0.0` and same-as-measured fraction was `1.0`.
- Seed0 `source_occ_free` fixed lambda `0` selected measured
  `n0001 -> n0112`.
- Interpretation: putting SC outside the cost denominator eliminated the short
  low-cost seed0 SC basin in this saved-tree offline diagnostic. This is not
  source-faithful and does not prove coverage improvement.
- Recommended next diagnostic step: inspect the decoupled sweep tables offline;
  do not proceed directly to runtime smoke or rollout from this diagnostic
  alone.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no runtime two-frame loop,
  no rollout/open-ended loop, no training/RL/PPO/BC/IL, no
  checkpoint/observed_state/prediction NPZ modification, no prediction
  writeback, no prediction traversability/collision/ray blocking, no
  target/ground-truth scoring, no external source modification/build, no
  Pareto gate/new runtime planner, and no coverage-improvement claim.

Stage 4A-6.5z.1 decoupled SC signal-strength diagnosis actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5x/6.5y/6.5z were complete and runtime
  smoke/rollout were not ready.
- Confirmed hard boundary for this task: offline table/saved-tree diagnosis
  only; no Isaac startup, no new RGB/depth capture, no map_predict rerun, no
  SSCNet inference, no action execution, no runtime two-frame loop, no
  rollout, no training/RL/PPO/BC/IL, no checkpoint/observed_state/prediction
  NPZ modification, no prediction writeback, no prediction
  traversability/collision/ray blocking, no target/ground-truth scoring, no
  external source modification/build, no Pareto gate/runtime planner
  implementation, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_decoupled_sc_signal_strength.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_decoupled_sc_signal_strength.py`.
- The runner loads saved Stage 4A-6.5z tables, saved Stage 4A-6.5y raw
  `current_raw_count` mini-RRT trees, saved Stage 4A-6.5p Frame2
  observed_state/prediction NPZ/pose/camera info, and regenerates
  debug-only per-path SC components because 6.5z saved only top-k candidate
  rows.
- Ran `py_compile` for:
  `diagnose_decoupled_sc_signal_strength.py`,
  `test_decoupled_sc_signal_strength.py`,
  `run_decoupled_sc_utility_sweep.py`,
  `run_source_gain_seed_replay.py`, and `offline_mini_rrt_tree.py`.
- Ran the requested diagnosis:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65z1_decoupled_signal_strength_diagnosis.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65z1_decoupled_signal_strength_diagnosis`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65z1_decoupled_signal_strength_diagnosis_test.log`.

Stage 4A-6.5z.1 conclusion:

- Validation passed.
- Required outputs/plots all exist (`38` required files checked).
- Debug regeneration was `recomputed_debug_tree_nodes=true` and produced
  `7650` candidate path rows.
- Diagnostic output sizes:
  `3003` near-miss rows, `4320` non-measured required-lambda rows, `1443`
  impossible-under-positive-lambda rows, `30` normalization rows, and `30`
  adaptive-gap rows.
- The 6.5z `lambda_sweep_summary_by_basis_variant` claims all tested
  decoupled variants were measured-only, but corrected per-formula/per-seed
  classification rows do not support that claim. Example corrected rows:
  `decoupled_source_occ_free_fixed_0` same-as-measured `0.8`, spatial seed0
  SC basin `0.2`; `decoupled_source_occ_free_fixed_32`
  same-as-measured `0.6`, spatial seed0 SC basin `0.1`.
- Required-lambda distribution for finite rows:
  p50/p90/max `229.31585862120286 / 627.9926880897762 /
  34462.89245592027`. Only `111` finite rows were `<=32`; `173` were
  `<= adaptive 2x`.
- Seed0 reference SC branch `n0127 -> n0162` is impossible to promote with
  positive lambda under all three SC bases because its normalized SC is lower
  than the measured-like branch.
- Normalization diagnosis:
  measured winners are already top-SC-quartile in `15/30` seed/basis rows,
  max normalized SC belongs to measured in `18/30`, and normalized-SC IQR is
  `<0.10` in `14/30`.
- Interpretation:
  current Frame2 map_predict SC signal is not cleanly branch-selective. The
  seed0 short local SC artifact is suppressed by decoupling because its
  normalized SC is lower than measured, not because moderate lambda promotes
  it. Some non-measured branches can theoretically flip, but most require
  lambda above `32`.
- Recommended next small task:
  larger offline lambda diagnostic sweep only.
- Still not next:
  runtime smoke, rollout, online open-ended loop, Pareto gate implementation,
  runtime planner implementation, RL/PPO/BC/IL, prediction writeback,
  observed_map prediction fusion, target/ground-truth scoring, checkpoint
  changes, coverage-improvement claims, or external source build.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no runtime two-frame loop,
  no rollout/open-ended loop, no training/RL/PPO/BC/IL, no
  checkpoint/observed_state/prediction NPZ modification, no prediction
  writeback, no prediction traversability/collision/ray blocking, no
  target/ground-truth scoring, no external source modification/build, no
  Pareto gate/runtime planner implementation, and no coverage-improvement
  claim.

Stage 4A-6.5aa synthetic SC validation actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5x/6.5y/6.5z/6.5z.1 were complete and
  runtime smoke/rollout remained not recommended.
- Confirmed boundary for this task: one controlled synthetic validation only;
  no selected action execution, no two-frame runtime, no rollout/open-ended
  loop, no training/RL/PPO/BC/IL, no checkpoint modification, no existing
  observed_state or prediction NPZ modification, no prediction writeback, no
  prediction traversability/collision/ray blocking, no target/ground-truth
  planning/scoring, no external source modification/build, no Pareto gate/new
  runtime planner, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_synthetic_sc_validation_scene.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_synthetic_sc_validation_scene.py`.
- Added the deterministic synthetic scene variant
  `synthetic_hidden_room_frontier`, with one fixed camera frame in room A
  looking through a doorway toward a hidden corridor/room, plus a measured-only
  frontier branch for contrast.
- The runner generated a measured-only observed_state, a read-only diagnostic
  Oracle/SyntheticPredictionLayer NPZ, one optional real map_predict NPZ on the
  same saved frame, and source-protected one-step mini-RRT tree decisions for
  seeds `0..4`.
- Ran `py_compile` for:
  `run_synthetic_sc_validation_scene.py`,
  `test_synthetic_sc_validation_scene.py`, `offline_mini_rrt_tree.py`,
  `sim_prediction_layer.py`, and `scene_factory.py`.
- Ran the synthetic validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aa_synthetic_sc_validation.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65aa_synthetic_sc_validation`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aa_synthetic_sc_validation_test.log`.

Stage 4A-6.5aa conclusion:

- Validation passed.
- Exactly one frame was captured. No selected action was executed; no two-frame
  runtime or rollout ran.
- Observed map shape/counts:
  `[120, 120, 30]`, unknown/free/occupied `256908 / 158196 / 16896`.
- Oracle prediction was read-only diagnostic, with valid `52184`, predicted
  occupied `11354`, predicted free `40830`, and hidden valid `44990`.
- Real map_predict was available from one saved-frame call, with valid `59904`,
  predicted occupied `2530`, predicted free `34392`, and hidden valid `16280`.
- Completed `45` decision rows for `5` seeds.
- Measured-only selected the measured-frontier direction in all seeds.
- Oracle source OCC+FREE over-cost selected the hidden-room direction in all
  seeds. Oracle decoupled minmax lambda `32` selected hidden-room direction in
  `4/5` seeds; lambda `8` and `16` stayed measured-frontier.
- Map_predict source OCC+FREE over-cost selected the hidden-room direction in
  all seeds. Map_predict decoupled minmax lambda `32` selected hidden-room
  direction in `3/5` seeds; lambda `8` and `16` stayed measured-frontier.
- Map/oracle direction agreement fraction was `0.95`.
- Low-cost artifact flags were `0/45`.
- Interpretation: Oracle and map_predict both produced useful hidden-room
  signal in this controlled diagnostic. This is still not a coverage
  improvement claim and does not by itself justify rollout or broad runtime
  use.
- Recommended next small task:
  repeat a tiny controlled map_predict calibration smoke before any runtime
  smoke.
- Still not next:
  rollout, online open-ended loop, RL/PPO/BC/IL, prediction writeback,
  observed_map prediction fusion, target/ground-truth scoring, checkpoint
  changes, coverage-improvement claims, external source modification/build, or
  a new runtime planner.

Stage 4A-6.5ab synthetic map_predict calibration smoke actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5z.1 and Stage 4A-6.5aa were complete.
- Confirmed hard boundary for this task: offline calibration/formula replay
  only; no Isaac startup, no new RGB/depth capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no two-frame runtime, no
  rollout/open-ended loop, no training/RL/PPO/BC/IL, no checkpoint or existing
  observed_state/prediction NPZ modification, no prediction writeback, no
  prediction traversability/collision/ray blocking, no target/ground-truth
  planning/scoring, no external source modification/build, no Pareto gate or
  runtime planner, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_synthetic_map_predict_calibration_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_synthetic_map_predict_calibration_smoke.py`.
- The runner loaded only saved Stage 4A-6.5aa artifacts:
  `observed_state_synthetic_frame000.npy`,
  `oracle_global_prediction_layer.npz`,
  `map_predict/global_prediction_layer.npz`, `pose_000.json`,
  `camera_info.json`, `scene_metadata.json`,
  `synthetic_sc_validation_summary.json`, and saved raw mini-RRT trees.
- The runner replayed seeds `0..4`, source OCC/FREE bases, confidence
  thresholds `0.05,0.1,0.2,0.4,0.6,0.8`, occ/free threshold diagnostics,
  over-cost, decoupled minmax lambda `16,24,32,48`, and decoupled-log
  lambda `32,48`.
- Ran `py_compile` for:
  `run_synthetic_map_predict_calibration_smoke.py`,
  `test_synthetic_map_predict_calibration_smoke.py`,
  `offline_mini_rrt_tree.py`, and `sim_prediction_layer.py`.
- Ran the calibration smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ab_synthetic_calibration_smoke.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ab_synthetic_calibration_smoke`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ab_synthetic_calibration_smoke_test.log`.

Stage 4A-6.5ab conclusion:

- Validation passed.
- Completed `96` configs and `485` decision rows over `5` seeds.
- Measured-only selected measured-frontier in `5/5` seeds.
- Best candidate:
  `map_predict|source_occ_free|decoupled_minmax_lambda48|tau0p1|occ0p5|free0p5`.
- Best candidate result:
  hidden-room `5/5`, Oracle/map_predict agreement `1.0`, low-cost artifact
  fraction `0.0`, mean selected SC gain `5018.4`, mean hidden-region count
  `4527.0`, and median margin `21.285778495568792`.
- Decoupled lambda `32` was insufficiently stable for map_predict source
  OCC+FREE minmax (`3/5` at tau `0.1`, `2/5` at tau `0.4`, `2/5` at tau
  `0.8`), while lambda `48` was stable (`5/5`) at tau `0.1`, `0.4`, and
  `0.8`.
- Source OCC+FREE over-cost remained stable across tested confidence and
  occ/free thresholds for map_predict and Oracle, but is still marked
  useful-but-risky because SC remains inside the cost denominator.
- Across all calibration rows, `3/485` low-cost artifact flags appeared in
  weaker diagnostic configs; the recommended config had `0.0`.
- Required summary files and plots exist, including calibration manifest,
  formula definitions, per-seed decisions, config summary, threshold/lambda
  sensitivity, Oracle/map agreement, low-cost diagnosis, hidden-region signal,
  best candidates, final summary, recommendation, and 9 PNG plots.
- Recommended next small task: saved-frame one-step formula smoke only.
- Still not next:
  runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, external source
  modification/build, Pareto gate implementation, or runtime planner
  implementation.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no two-frame runtime, no
  rollout/open-ended loop, no training/RL/PPO/BC/IL, no
  checkpoint/observed_state/prediction NPZ modification, no prediction
  writeback, no prediction traversability/collision/ray blocking, no
  target/ground-truth scoring, no external source modification/build, and no
  coverage-improvement claim.

Stage 4A-6.5ac saved-frame lambda48 formula smoke actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5aa and Stage 4A-6.5ab were complete.
- Confirmed hard boundary for this task: saved-frame one-step formula smoke
  only; no Isaac startup, no new capture, no map_predict rerun, no SSCNet
  inference, no selected action execution, no two-frame runtime, no rollout,
  no open-ended loop, no training/RL/PPO/BC/IL, no checkpoint or existing
  observed_state/prediction NPZ modification, no prediction writeback, no
  prediction traversability/collision/ray blocking, no target/ground-truth
  planning/scoring, no external source modification/build, no Pareto gate or
  runtime planner, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_saved_frame_lambda48_formula_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_saved_frame_lambda48_formula_smoke.py`.
- The runner loaded only saved Stage 4A-6.5aa/6.5ab artifacts:
  `observed_state_synthetic_frame000.npy`,
  `oracle_global_prediction_layer.npz`,
  `map_predict/global_prediction_layer.npz`, `pose_000.json`,
  `camera_info.json`, `scene_metadata.json`,
  6.5aa per-seed/branch CSVs, 6.5ab `best_config_candidates.*`, and saved
  raw mini-RRT trees.
- The runner replayed seeds `0..4` for modes:
  `measured_only`, `oracle_lambda48`, `map_predict_lambda48`,
  `map_predict_lambda32`, `oracle_over_cost`, and `map_predict_over_cost`.
- The lambda48 implementation explicitly logged:
  `base_exp_value`, `normalized_sc`, `sc_bonus`, and `final_value`, using
  `gain_exp / cost + 48 * minmax(source_occ_free)`.
- Ran `py_compile` for:
  `run_saved_frame_lambda48_formula_smoke.py`,
  `test_saved_frame_lambda48_formula_smoke.py`, `offline_mini_rrt_tree.py`,
  and `sim_prediction_layer.py`.
- Ran the formula smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ac_saved_frame_lambda48_formula_smoke.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ac_saved_frame_lambda48_formula_smoke`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ac_saved_frame_lambda48_formula_smoke_test.log`.

Stage 4A-6.5ac conclusion:

- Validation passed.
- Completed `30` decision rows over `5` seeds and `6` modes.
- measured_only selected measured-frontier in `5/5` seeds.
- Oracle lambda48 selected hidden-room in `5/5` seeds.
- map_predict lambda48 selected hidden-room in `5/5` seeds.
- map_predict/Oracle lambda48 agreement was `1.0`.
- map_predict lambda48 low-cost artifact fraction was `0.0`.
- map_predict lambda48 median margin was `21.285778495568792`, matching the
  Stage 4A-6.5ab best-candidate median margin.
- map_predict lambda32 remained less stable (`3/5` hidden-room), consistent
  with Stage 4A-6.5ab.
- Required files and plots exist, including loaded input manifest, formula
  definition, per-seed decisions, per-seed value components, branch direction
  classification, lambda48 reproduction summary, Oracle-vs-map lambda48
  summary, low-cost artifact diagnosis, comparison to Stage 4A-6.5ab,
  prediction safety/hash reports, final summary, recommendation, and 7 PNG
  plots.
- Recommended next small task:
  saved-frame formula smoke on one real `medium_three_rooms` frame only.
- Still not next:
  runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, external source
  modification/build, Pareto gate implementation, or runtime planner
  implementation.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no two-frame runtime, no
  rollout/open-ended loop, no training/RL/PPO/BC/IL, no
  checkpoint/observed_state/prediction NPZ modification, no prediction
  writeback, no prediction traversability/collision/ray blocking, no
  target/ground-truth scoring, no external source modification/build, no
  leakage, and no coverage-improvement claim.

Stage 4A-6.5ad real-frame lambda48 formula smoke actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5aa, 6.5ab, and 6.5ac were complete.
- Confirmed hard boundary for this task: one saved real Frame2 offline formula
  smoke only; no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no two-frame runtime, no
  rollout, no open-ended loop, no training/RL/PPO/BC/IL, no checkpoint or
  existing observed_state/prediction NPZ modification, no prediction writeback,
  no prediction traversability/collision/ray blocking, no target/ground-truth
  planning/scoring, no external source modification/build, no Pareto gate or
  runtime planner, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_real_frame_lambda48_formula_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_real_frame_lambda48_formula_smoke.py`.
- The runner loaded saved Stage 4A-6.5p Frame2 inputs:
  `observed_state_frame002.npy`,
  `frame002_prediction/global_prediction_layer.npz`, `frame002_pose.json`,
  and `frame002_camera_info.json`.
- The runner also loaded saved Stage 4A-6.5y raw mini-RRT trees for seeds
  `0..9` to stay aligned with the Stage 4A-6.5p/6.5z Frame2 reference root
  and branch IDs. The first attempted rebuild from `frame002_pose.json`
  correctly stayed offline but did not reproduce the reference root, so the
  final validated path uses the saved raw-tree artifacts.
- Replayed modes:
  `measured_only`, `map_predict_lambda32`, `map_predict_lambda48`,
  `source_occ_free_over_cost`, `raw_hybrid_over_cost`, and
  `source_occ_free_no_cost`.
- The lambda48 implementation explicitly logged:
  `base_exp_value`, `source_occ_free`, `normalized_sc`, `sc_bonus`, and
  `final_value`, using
  `gain_exp / cost + 48 * minmax(source_occ_free)`.
- Ran `py_compile` for:
  `run_real_frame_lambda48_formula_smoke.py`,
  `test_real_frame_lambda48_formula_smoke.py`, `offline_mini_rrt_tree.py`,
  and `sim_prediction_layer.py`.
- Ran the formula smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ad_real_frame_lambda48_formula_smoke.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ad_real_frame_lambda48_formula_smoke`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ad_real_frame_lambda48_formula_smoke_test.log`.

Stage 4A-6.5ad conclusion:

- Validation passed.
- Completed `60` decision rows over `10` seeds and `6` modes.
- measured_only reproduced the Stage 4A-6.5p Frame2 measured reference for
  seed0: selected child `n0001`, best descendant `n0112`.
- map_predict lambda48 selected same-as-measured in `6/10` seeds and distinct
  non-measured branches in `4/10`; healthy non-measured fraction `0.4`.
- map_predict lambda48 avoided the prior low-cost SC basin:
  spatial prior basin fraction `0.0`, low-cost artifact fraction `0.0`.
- map_predict lambda48 seed0 kept selected child `n0001` and selected best
  descendant `n0243`, so it did not return to `n0127 -> n0162`.
- map_predict lambda32 matched lambda48 at branch-class level in this replay:
  same-as-measured `6/10`, distinct non-measured `4/10`, prior basin `0.0`.
- over-cost diagnostics reproduced the old risk shape: seed0 returned to
  `n0127`, primary spatial-prior classification appeared in `3/10` rows, and
  the spatial prior basin flag fraction was `0.5`.
- Required files and plots exist, including loaded input manifest, formula
  definition, reference branches, per-seed decisions, per-seed value
  components, branch classification, lambda48 behavior summary, low-cost
  artifact diagnosis, comparison to Stage 4A-6.5z/z.1, prediction safety/hash
  reports, final summary, recommendation, and 8 PNG plots.
- Recommended next small task:
  saved-frame formula smoke on another real medium frame only.
- Still not next:
  runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, external source
  modification/build, Pareto gate implementation, or runtime planner
  implementation.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no two-frame runtime, no
  rollout/open-ended loop, no training/RL/PPO/BC/IL, no
  checkpoint/observed_state/prediction NPZ modification, no prediction
  writeback, no prediction traversability/collision/ray blocking, no
  target/ground-truth scoring, no external source modification/build, no
  leakage, and no coverage-improvement claim.

Stage 4A-6.5ae real Frame1 lambda48 formula smoke actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5ab, 6.5ac, 6.5ad, and 6.5p context.
- Confirmed hard boundary for this task: another saved real
  `medium_three_rooms` frame only; no Isaac startup, no new capture, no
  map_predict rerun, no SSCNet inference, no selected action execution, no
  two-frame runtime, no rollout, no open-ended loop, no training/RL/PPO/BC/IL,
  no checkpoint or existing observed_state/prediction NPZ modification, no
  prediction writeback, no prediction traversability/collision/ray blocking,
  no target/ground-truth planning/scoring, no external source modification/
  build, no Pareto gate or runtime planner, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_real_frame_lambda48_formula_smoke_another.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_real_frame_lambda48_formula_smoke_another.py`.
- Selected the preferred Stage 4A-6.5p Frame1 saved inputs:
  `observed_state_frame001.npy`,
  `frame001_prediction/global_prediction_layer.npz`, `frame001_pose.json`,
  and `frame001_camera_info.json`. The selected-frame report records no
  fallback and no use of the Stage 4A-6.5ad Frame2 frame as the evaluated
  frame.
- Rebuilt one-frame mini-RRT decisions offline for seeds `0..9`, instead of
  reading the Stage 4A-6.5y Frame2 raw-tree cache.
- Replayed modes:
  `measured_only`, `map_predict_lambda32`, `map_predict_lambda48`,
  `source_occ_free_over_cost`, `raw_hybrid_over_cost`, and
  `source_occ_free_no_cost`.
- The lambda48 implementation explicitly logged:
  `base_exp_value`, `source_occ_free`, `normalized_sc`, `sc_bonus`, and
  `final_value`, using
  `gain_exp / cost + 48 * minmax(source_occ_free)`.
- Ran `py_compile` for:
  `run_real_frame_lambda48_formula_smoke_another.py` and
  `test_real_frame_lambda48_formula_smoke_another.py`, logged at
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ae_py_compile.log`.
- Ran the formula smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ae_real_frame1_lambda48_formula_smoke.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ae_real_frame1_lambda48_formula_smoke`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ae_real_frame1_lambda48_formula_smoke_test.log`.

Stage 4A-6.5ae conclusion:

- Validation passed.
- Completed `60` decision rows over `10` seeds and `6` modes.
- measured_only reproduced the Stage 4A-6.5p Frame1 measured reference for
  seed0: selected child `n0001`, best descendant `n0249`.
- map_predict lambda48 selected same-as-measured in `8/10` seeds and distinct
  non-measured branches in `2/10`; healthy non-measured fraction `0.2`.
- map_predict lambda48 avoided the historical prior low-cost SC basin:
  prior basin fraction `0.0`, low-cost artifact fraction `0.0`.
- map_predict lambda48 seed0 kept selected child `n0001` and best descendant
  `n0249`.
- map_predict lambda32 matched lambda48 at branch-class level in this replay:
  same-as-measured `8/10`, distinct non-measured `2/10`, prior basin `0.0`.
- over-cost diagnostics on Frame1 selected same-as-measured in `5/10` and
  distinct non-measured in `5/10`, with prior basin fraction `0.0` and
  low-cost artifact fraction `0.0`.
- Required files and plots exist, including selected-frame report, loaded
  input manifest, formula definition, reference branches, per-seed decisions,
  per-seed value components, branch classification, lambda48 behavior summary,
  low-cost artifact diagnosis, comparison to Stage 4A-6.5z/z.1, prediction
  safety/hash reports, final summary, recommendation, and 8 PNG plots.
- Still not next:
  runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, external source
  modification/build, Pareto gate implementation, or runtime planner
  implementation.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no two-frame runtime, no
  rollout/open-ended loop, no training/RL/PPO/BC/IL, no
  checkpoint/observed_state/prediction NPZ modification, no prediction
  writeback, no prediction traversability/collision/ray blocking, no
  target/ground-truth scoring, no external source modification/build, no
  leakage, and no coverage-improvement claim.

Stage 4A-6.5af offline saved-frame lambda48 consolidation actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5ab, 6.5ac, 6.5ad, and 6.5ae were
  complete.
- Confirmed hard boundary for this task: offline saved-frame-only
  consolidation/design review; no Isaac startup, no new capture, no
  map_predict rerun, no SSCNet inference, no selected action execution, no
  two-frame runtime, no rollout, no open-ended loop, no training/RL/PPO/BC/IL,
  no checkpoint or existing observed_state/prediction NPZ modification, no
  prediction writeback, no prediction traversability/collision/ray blocking,
  no target/ground-truth planning/scoring, no external source modification/
  build, no Pareto gate or runtime planner, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/consolidate_lambda48_saved_frame_review.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_lambda48_saved_frame_consolidation.py`.
- The consolidation runner loaded saved Stage 4A-6.5ab / 6.5ac / 6.5ad /
  6.5ae summaries, per-seed decisions, branch classification tables,
  low-cost diagnostics, safety/hash reports, and optional 6.5p/z/z.1 context
  paths only.
- It wrote:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65af_lambda48_saved_frame_consolidation`.
- Ran `py_compile` for:
  `consolidate_lambda48_saved_frame_review.py` and
  `test_lambda48_saved_frame_consolidation.py`.
- Ran the consolidation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65af_lambda48_saved_frame_consolidation.log`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65af_lambda48_saved_frame_consolidation_test.log`.

Stage 4A-6.5af conclusion:

- Validation passed.
- Generated all required CSV/JSON/MD outputs and 8 PNG plots:
  loaded inputs manifest, missing fields report, unified config table,
  lambda48 cross-frame summary, real-frame aggregate, lambda32-vs-lambda48
  comparison, over-cost diagnostic comparison, low-cost artifact cross-frame
  table, readiness matrix, design review findings, final summary, and
  recommended next faithful step.
- Synthetic lambda48 evidence remains strong:
  hidden-room `5/5`, Oracle/map_predict agreement `1.0`, low-cost artifact
  fraction `0.0`, median margin `21.285778495568792`.
- Real aggregate lambda48 across Frame2 and Frame1:
  total seed-frame rows `20`, same-as-measured `14/20`, distinct and healthy
  non-measured `6/20`, historical prior basin `0/20`, low-cost artifact
  `0/20`, real median margin `25.005253421860232`, and seed-level branch
  class consistency `6/10` across overlapping seed labels.
- Lambda32 matched lambda48 at real branch-class level (`20/20`) and selected
  the same immediate child in `20/20`, but exact best-descendant match was
  `13/20`; synthetic evidence still favors lambda48 because lambda32 was only
  `3/5` hidden-room.
- Over-cost remains diagnostic-only. Frame2 reproduced the old risk shape
  with prior basin fraction `0.5`; Frame1 was more aggressive without
  low-cost artifacts, but that does not justify over-cost runtime.
- Recommended next faithful small task:
  offline saved-frame-only multi-frame lambda48 replay over all available
  saved real `medium_three_rooms` frames.
- Still not next:
  runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL training,
  prediction writeback, observed_map prediction fusion, target/ground-truth
  scoring, checkpoint changes, coverage-improvement claims, external source
  modification/build, Pareto gate implementation, or runtime planner
  implementation.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no two-frame runtime, no
  rollout/open-ended loop, no training/RL/PPO/BC/IL, no
  checkpoint/observed_state/prediction NPZ modification, no prediction
  writeback, no prediction traversability/collision/ray blocking, no
  target/ground-truth scoring, no external source modification/build, no
  leakage, no runtime planner, and no coverage-improvement claim.

Stage 4A-6.5ag offline multi-frame lambda48 replay actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5ab through 4A-6.5af were complete.
- Confirmed hard boundary for this task: offline saved-frame-only replay over
  existing saved real `medium_three_rooms` artifacts; no Isaac startup, no new
  capture, no map_predict rerun, no SSCNet inference, no selected action
  execution, no two-frame runtime, no rollout/open-ended loop, no
  training/RL/PPO/BC/IL, no checkpoint or existing observed_state/prediction
  NPZ modification, no prediction writeback, no prediction traversability/
  collision/ray blocking, no target/ground-truth or future-observed
  planning/scoring, no external source modification/build, no Pareto gate or
  runtime planner, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_multi_frame_lambda48_replay.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_multi_frame_lambda48_replay.py`.
- The runner discovered existing candidates under
  `/home/ubuntu22/sc_explorer_ws/outputs`, deduplicated by observed_state
  hash, pose/camera hash, and root pose, and kept canonical Stage 4A-6.5p
  / rollout representatives.
- To use the available CPU more effectively, the replay was split into
  frame+seed tasks and run with `32` worker processes over `70` tasks.
- Ran `py_compile`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ag_py_compile.log`.
- Ran replay:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ag_multi_frame_lambda48_replay.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ag_multi_frame_lambda48_replay`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ag_multi_frame_lambda48_replay_test.log`.

Stage 4A-6.5ag conclusion:

- Validation passed.
- Required outputs exist: loaded context manifest, frame discovery inventory,
  duplicate and skipped-frame reports, selected-frame manifest, formula
  definition, root alignment report, per-frame seed/mode decisions,
  value components, branch classifications, lambda48 multi-frame summary,
  lambda32-vs-lambda48 comparison, over-cost diagnostic, low-cost artifact
  report, prediction safety/hash reports, final summary, recommendation, and
  10 PNG plots.
- Frame discovery found `20` candidate rows, `17` valid candidates, and `7`
  unique real medium frames after deduplication; duplicates recorded `10`
  rows.
- Replay completed `420` decision rows over `7` frames, seeds `0..9`, and
  `6` modes.
- map_predict lambda48 aggregate over `70` seed-frame rows:
  same-as-measured `33/70`, distinct non-measured `35/70`, local jitter
  `2/70`, healthy non-measured `35/70`, historical prior basin `0/70`,
  low-cost artifact `0/70`, median margin `18.93872168517339`.
- Per-frame healthy non-measured fractions:
  Frame1 `0.2`, Frame2 `0.4`, rollout step000 `0.3`, step001 `0.2`,
  step002 `0.8`, step003 `0.7`, and step004 `0.9`.
- Lambda32 vs lambda48:
  branch-class agreement `62/70`, selected-child agreement `61/70`, and
  best-descendant agreement `41/70`; lambda32 remains diagnostic because
  synthetic calibration still favors lambda48.
- Over-cost remains diagnostic-only:
  source_occ_free_over_cost had historical prior-basin fraction `24/70`
  despite low-cost artifact `0/70`.
- Recommended next faithful small task:
  multi-scene/start saved-frame replay if available, or staged one-frame
  runtime-smoke design review only. Do not recommend rollout directly.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no two-frame runtime, no
  rollout/open-ended loop, no training/RL/PPO/BC/IL, no checkpoint or existing
  observed_state/prediction NPZ modification, no prediction writeback, no
  prediction traversability/collision/ray blocking, no target/ground-truth or
  future-observed planning/scoring, no external source modification/build, no
  leakage, no runtime planner, and no coverage-improvement claim.

Stage 4A-6.5ah hardware-aware saved-frame discovery / design review actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5af and Stage 4A-6.5ag were complete.
- Confirmed hard boundary for this task: search existing saved artifacts only;
  no Isaac startup, no new capture, no map_predict rerun, no SSCNet
  inference, no selected action execution, no two-frame runtime, no rollout,
  no open-ended loop, no training/RL/PPO/BC/IL, no checkpoint or existing
  observed_state/prediction NPZ modification, no prediction writeback, no
  prediction traversability/collision/ray blocking, no target/ground-truth or
  future-observed planning/scoring, no external source modification/build, no
  Pareto dominance gate or runtime planner implementation, and no
  coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65ah_multiscene_or_design_review.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ah_multiscene_or_design_review.py`.
- Added `--max_workers` to the Stage 4A-6.5ah runner with default `32`.
  The run used `actual_max_workers=min(32, os.cpu_count() or 1)=32`.
- Set process-pool BLAS/OMP inner-thread environment variables to `1`
  (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`) and recorded them in
  `hardware_utilization_report.*`.
- Ran `py_compile`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ah_py_compile.log`.
- Ran the Stage 4A-6.5ah runner with `--max_workers 32`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ah_multiscene_or_design_review.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ah_multiscene_or_runtime_design_review`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ah_multiscene_or_design_review_test.log`.

Stage 4A-6.5ah conclusion:

- Validation passed.
- Broad saved-artifact discovery found `217` candidate rows:
  `7` already in Stage 4A-6.5ag, `55` duplicates or same-frame prediction
  variants, `127` incomplete missing prediction, `26` incomplete missing
  pose/camera, `2` synthetic/forbidden, and `0` new complete frames.
- No additional complete saved real frames were available beyond Stage
  4A-6.5ag's seven selected frame identities, so Stage 4A-6.5ah did not run
  offline replay and wrote staged one-frame runtime-smoke design review only.
- Design review outputs specify a future safe runtime smoke with one Isaac
  startup, one frame, one map_predict call, one source-protected tree
  decision, lambda48 only, measured-only shadow, optional lambda32 shadow, no
  selected action execution, no second frame, no rollout, no coverage claim,
  no prediction writeback, no prediction traversability/collision/ray
  blocking, and hard stop after decision output.
- Hardware utilization was logged:
  `os_cpu_count=32`, requested `--max_workers 32`,
  `actual_max_workers=32`, `parallel_backend=ProcessPoolExecutor`,
  `task_count=62`, process workers used, wall time recorded, and all
  process-worker BLAS/OMP inner-thread variables set to `1`.
- Required outputs exist:
  loaded context manifest, hardware utilization report, additional frame
  discovery inventory, duplicate report, new complete frame manifest, skipped
  candidate report, runtime smoke design review, runtime smoke safety
  checklist, future Stage 4A-6.5ai command sketch, final summary, and
  recommended next faithful step.
- Current next small task if runtime is desired:
  `Stage 4A-6.5ai staged one-frame lambda48 runtime smoke, no action
  execution`.
- Alternative next small task:
  collect additional saved frames in a controlled capture-only stage, still no
  rollout.
- Hardware utilization policy for future offline replay/analysis stages:
  use maximum available CPU parallelism by default. On this workstation,
  CPU-bound offline stages should request `--max_workers 32` and use
  `actual_max_workers=min(32, os.cpu_count() or 1)`. Process-pool workers
  should set BLAS/OMP inner threads to `1` to avoid oversubscription;
  single-process numeric runs may set BLAS/OMP/torch threads to `32`. Every
  stage should log requested/actual workers, `os.cpu_count()`, parallel
  backend, task count, wall time, worker/process/thread mode, and thread
  environment variables.
- Safety passed: no Isaac startup, no new capture, no map_predict rerun, no
  SSCNet inference, no selected action execution, no two-frame runtime, no
  rollout/open-ended loop, no training/RL/PPO/BC/IL, no checkpoint or existing
  observed_state/prediction NPZ modification, no prediction writeback, no
  prediction traversability/collision/ray blocking, no target/ground-truth or
  future-observed planning/scoring, no external source modification/build, no
  leakage, no Pareto gate, no runtime planner, and no coverage-improvement
  claim.

Stage 4A-6.5ai one-frame lambda48 runtime smoke actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5ag / 4A-6.5ah plus the hardware policy
  were present.
- Confirmed hard boundary for this task: exactly one Isaac startup, exactly
  one RGB/depth frame, exactly one measured-only observed_state update,
  exactly one map_predict call, exactly one primary lambda48 source-protected
  tree decision, measured-only shadow, lambda32 shadow, no selected action
  execution, no second frame, no two-frame runtime, no rollout/open-ended
  loop, no training/RL/PPO/BC/IL, no checkpoint or existing observed_state /
  prediction NPZ modification, no prediction writeback/fusion, no prediction
  traversability/collision/ray blocking/candidate sampling/edge validity, no
  target/ground-truth/future-observed planning/scoring, no external source
  modification/build, no Pareto gate/runtime planner, no over-cost runtime
  primary, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65ai_one_frame_lambda48_runtime_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ai_one_frame_lambda48_runtime_smoke.py`.
- Ran `py_compile`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ai_py_compile.log`.
- Ran the one-frame runtime smoke with `--max_workers 32`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ai_one_frame_lambda48_runtime_smoke.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ai_one_frame_lambda48_runtime_smoke`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ai_one_frame_lambda48_runtime_smoke_test.log`.

Stage 4A-6.5ai conclusion:

- Validation passed.
- Required outputs exist: loaded context manifest, hardware report, runtime
  capture summary, observed_state update summary, map_predict summary, formula
  definition, source-protection checklist, measured shadow decision, lambda48
  primary decision, lambda32 shadow decision, tree comparison, branch
  classification, low-cost diagnosis, prediction safety report, hash checks,
  no-action report, final summary, recommendation, capture RGB/depth,
  observed_state, map_predict global prediction layer, and 6 PNG plots.
- Runtime setup was one Isaac startup, deterministic `medium_three_rooms`
  scene seed `0`, canonical start pose `[-4.65, -4.65, 1.2]`, yaw
  `0.38710316317995463`, exactly one RGB/depth frame, exactly one
  measured-only observed_state update, exactly one map_predict call using
  `code_consistent_v1`, and no selected action execution, second frame,
  two-frame runtime, or rollout.
- Observed_state result: shape `(120,120,30)`, observed_ratio
  `0.0425462962962963`.
- map_predict result: prediction shape aligned to observed_state, valid
  predictions `57382`, predicted unmeasured OCC+FREE `40328`
  (`9494` occupied, `30834` free), checkpoint unchanged.
- Formula executed: `gain_exp / cost + 48 * minmax(source_occ_free)` with tau
  `0.1`, occ/free thresholds `0.5/0.5`; over-cost remained diagnostic-only
  history and was not run as runtime primary.
- Measured-only shadow selected `n0013 -> n0159`.
- Lambda48 primary selected `n0001 -> n0228`, branch classification
  `distinct_nonmeasured_branch`, healthy non-measured `true`, low-cost
  artifact `false`, historical prior basin `false`.
- Lambda32 shadow also selected `n0001 -> n0228`.
- Hardware utilization was logged:
  `os_cpu_count=32`, requested/actual `--max_workers 32/32`,
  `parallel_backend=single_process_runtime_stage_no_process_pool`,
  OMP `1`, OPENBLAS `32`, MKL `1`, NUMEXPR `1`, VECLIB `1`, GPU
  `NVIDIA GeForce RTX 5080`, total wall time `29.64396972299801s`.
- Safety passed: no action execution, no second frame, no rollout/open-ended
  loop, no training/RL/PPO/BC/IL, checkpoint unchanged, no existing
  observed_state or prediction NPZ modified, prediction stayed read-only and
  was not written/fused into observed_state, prediction was not used for
  traversability/collision/ray blocking/candidate sampling/edge validity, no
  target/ground-truth/future-observed planning/scoring, no external source
  modification/build, no leakage, no Pareto gate/runtime planner, no over-cost
  runtime promotion, and no coverage-improvement claim.
- Recommended next small task:
  `Stage 4A-6.5aj staged two-frame one-action lambda48 runtime smoke design
  review only`, or controlled capture-only additional saved-frame collection.
  Do not recommend rollout directly.

Stage 4A-6.5aj staged two-frame one-action lambda48 runtime smoke design
review actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, and confirmed Stage 4A-6.5ag / 4A-6.5ah / 4A-6.5ai plus the
  hardware policy were present.
- Confirmed hard boundary for this task: design review only; no Isaac startup,
  no RGB/depth capture, no map_predict call, no SSCNet inference, no selected
  action execution, no two-frame runtime, no rollout/open-ended loop, no
  training/RL/PPO/BC/IL, no checkpoint or existing observed_state/prediction
  NPZ modification, no prediction writeback/fusion, no prediction
  traversability/collision/ray blocking/candidate sampling/edge validity, no
  target/ground-truth/future-observed planning/scoring, no external source
  modification/build, no Pareto gate/runtime planner, no over-cost runtime
  primary, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/design_stage4a65aj_two_frame_one_action_runtime_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65aj_two_frame_one_action_design_review.py`.
- Ran `py_compile`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aj_py_compile.log`.
- Ran the Stage 4A-6.5aj design generator with `--max_workers 32`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aj_two_frame_one_action_design_review.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65aj_two_frame_one_action_runtime_design_review`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65aj_two_frame_one_action_design_review_test.log`.

Stage 4A-6.5aj conclusion:

- Validation passed.
- Required outputs exist: loaded context manifest, Stage 4A-6.5ai result
  review, future runtime smoke design, future two-frame sequence spec, future
  lambda48 formula spec, future source-protection profile, future pre-action
  safety gates, future Frame 2 stop conditions, future required outputs,
  future test requirements, future Stage 4A-6.5ak command sketch, DO NOT RUN
  statement, hardware policy, prediction safety design, risk register, rollout
  blocker statement, final summary, recommendation, missing-field report, and
  three design PNG diagrams.
- Stage 4A-6.5ai review result:
  clean one-frame no-action runtime, lambda48 selected `n0001 -> n0228`,
  branch `distinct_nonmeasured_branch`, low-cost artifact `false`,
  historical prior basin `false`, and rollout_ready `false`.
- Future Stage 4A-6.5ak design:
  exactly two frames, exactly two measured-only observed_state updates,
  exactly two map_predict calls, exactly one Frame 1 lambda48-selected action
  if safety gates pass, no second action, no third frame, and no rollout.
- Future lambda48 primary formula:
  `gain_exp / cost + 48 * minmax(source_occ_free)`; SC stays outside the cost
  denominator, and over-cost remains prohibited for runtime primary.
- Future safety gates require:
  no low-cost artifact, no historical prior basin unless action is blocked,
  prediction read-only, no prediction traversability/collision/ray
  blocking/candidate sampling/edge validity, no target/ground-truth/
  future-observed planning or scoring, finite in-bounds selected action, and
  action_execution_count `0` before Frame 1 action.
- Hardware policy for the future command includes `--max_workers 32`; helper
  process-pool workers should keep BLAS/OMP inner threads at `1`.
- Safety passed: no Isaac startup, no capture, no map_predict, no SSCNet
  inference, no action execution, no two-frame runtime, no rollout/open-ended
  loop, no training/RL/PPO/BC/IL, no checkpoint or existing observed_state /
  prediction NPZ modification, no prediction writeback/fusion, no prediction
  traversability/collision/ray blocking/candidate sampling/edge validity, no
  target/ground-truth/future-observed planning/scoring, no external source
  modification/build, no leakage, no Pareto gate/runtime planner, no over-cost
  runtime promotion, and no coverage-improvement claim.
- Recommended next small task:
  `Stage 4A-6.5ak staged two-frame one-action lambda48 runtime smoke execution`
  only if explicitly requested by user. Do not recommend rollout directly.

Stage 4A-6.5ak staged two-frame one-action lambda48 runtime smoke actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, plus the user-provided execution spec, and confirmed
  Stage 4A-6.5ag / 4A-6.5ah / 4A-6.5ai / 4A-6.5aj were present.
- Confirmed hard boundary: exactly one Isaac app startup per runner run,
  exactly two frames if Frame 1 gates pass, exactly two map_predict calls if
  the action executes, exactly one Frame 1 lambda48-selected action, no second
  action, no third frame, no rollout/open-ended loop, no training/RL/PPO/BC/IL,
  no checkpoint change, no existing observed_state or prediction NPZ edit, no
  prediction writeback/fusion, no prediction traversability/collision/ray
  blocking/candidate sampling/edge validity, no target/ground-truth/
  future-observed scoring, no external source build/edit, no over-cost runtime
  primary, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke.py`.
- Ran `py_compile` for the new runner/test plus `offline_mini_rrt_tree.py`,
  `sim_prediction_layer.py`, `isaac_map_predictor.py`, and
  `isaac_sscnet_preprocess.py`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ak_py_compile.log`.
- Ran the Stage 4A-6.5ak runtime smoke with `--max_workers 32`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ak_two_frame_one_action_lambda48_runtime_smoke.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ak_two_frame_one_action_lambda48_runtime_smoke_test.log`.

Stage 4A-6.5ak conclusion:

- Validation passed.
- Runtime setup:
  one Isaac startup, `medium_three_rooms` seed `0`, start pose
  `[-4.65, -4.65, 1.2]`, yaw `0.38710316317995463`, exactly `2` frames,
  exactly `2` measured-only observed_state updates, exactly `2` map_predict
  calls, exactly `1` selected action execution, no second action, no third
  frame, and no rollout.
- Frame 1 result:
  measured-only shadow selected `n0013 -> n0159`; lambda48 primary selected
  `n0001 -> n0228`; lambda32 shadow selected `n0001 -> n0228`.
  Lambda48 was `distinct_nonmeasured_branch`, healthy non-measured `true`,
  low-cost artifact `false`, historical prior basin `false`; all pre-action
  hard gates passed.
- Action execution:
  exactly one lambda48-selected action to pose `[-4.15, -4.55, 1.2]`, yaw
  `1.7681918866447788`.
- Frame 2 result:
  measured-only shadow selected `n0014 -> n0108`; lambda48 diagnostic selected
  `n0002 -> n0158`; lambda32 shadow selected `n0002 -> n0158`.
  Lambda48 was `distinct_nonmeasured_branch`, healthy non-measured `true`,
  low-cost artifact `false`, historical prior basin `false`.
- Hardware:
  `os_cpu_count=32`, requested/actual max_workers `32/32`,
  OMP/OPENBLAS/MKL/NUMEXPR/VECLIB `1/1/1/1/1`, GPU
  `NVIDIA GeForce RTX 5080`, total wall time `57.02001969000048s`.
- Safety passed:
  checkpoint unchanged, no existing observed_state modified, generated
  prediction NPZs unchanged after creation, prediction read-only and
  information-gain-only, no prediction writeback/fusion, no prediction
  traversability/collision/ray blocking/candidate sampling/edge validity, no
  target/ground-truth/future-observed scoring, no training/RL/PPO/BC/IL, no
  external source modification/build, no over-cost runtime primary, no leakage,
  and no coverage-improvement claim.
- Recommended next small task:
  `Stage 4A-6.5al post-action/two-frame diagnosis and repeat-safety review
  only`. Do not recommend rollout directly.

Stage 4A-6.5al post-action/two-frame diagnosis and repeat-safety review
actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, plus the user-provided Stage 4A-6.5al spec.
- Confirmed hard boundary: diagnosis/review only; no Isaac startup, no
  RGB/depth capture, no map_predict call, no SSCNet inference, no selected
  action execution, no two-frame runtime execution, no rollout/open-ended
  loop, no training/RL/PPO/BC/IL, no checkpoint change, no existing
  observed_state or prediction NPZ edit, no prediction writeback/fusion, no
  prediction traversability/collision/ray blocking/candidate sampling/edge
  validity, no target/ground-truth/future-observed planning/scoring, no
  external source modification/build, no over-cost runtime primary, and no
  coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_stage4a65al_post_action_two_frame.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65al_post_action_two_frame.py`.
- Ran `py_compile`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65al_py_compile.log`.
- Ran the offline Stage 4A-6.5al diagnosis with `--max_workers 32`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65al_post_action_two_frame_diagnosis.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65al_post_action_two_frame_diagnosis`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65al_post_action_two_frame_diagnosis_test.log`.

Stage 4A-6.5al conclusion:

- Validation passed.
- Stage 4A-6.5ak output sequence was reverified:
  exactly `2` frames, exactly `2` measured-only observed_state updates,
  exactly `2` map_predict calls, exactly `1` selected action execution,
  no second action, no third frame, and no rollout.
- Action pose consistency passed:
  executed pose `[-4.15, -4.55, 1.2]`, yaw `1.7681918866447788`,
  matched Frame 2 pose with position/yaw error `0.0`.
- Observed_state delta was sane:
  observed_ratio `0.0425462962962963 -> 0.06938657407407407`,
  delta `0.026840277777777775`, newly observed `11595`,
  unknown->free `10360`, unknown->occupied `1235`,
  minor free->occupied refinements `83`, occupied->free `0`, invalid labels
  `0`.
- map_predict remained stable post-action:
  Frame 1 valid/OCC+FREE `57382 / 40328`, Frame 2 valid/OCC+FREE
  `47814 / 30133`, density ratio `0.7471979765919461`, no density
  explosion/collapse, both `code_consistent_v1`.
- Tree/branch diagnosis:
  Frame 1 lambda48 selected `n0001 -> n0228`; Frame 2 lambda48 selected
  `n0002 -> n0158`; both were `distinct_nonmeasured_branch`, healthy
  non-measured `true`, low-cost artifact `false`, historical prior basin
  `false`. Lambda32 matched lambda48 on both frames.
- Hash audit passed:
  checkpoint, Stage 4A-6.5ak observed_state files, and Stage 4A-6.5ak
  prediction NPZ files were unchanged.
- Hardware:
  `os_cpu_count=32`, requested/actual max_workers `32/32`,
  OMP/OPENBLAS/MKL/NUMEXPR/VECLIB `1/1/1/1/1`, GPU from Stage 4A-6.5ak
  `NVIDIA GeForce RTX 5080`, analysis task count `76`.
- Safety passed:
  no Isaac startup/capture/map_predict/SSCNet inference/action/two-frame
  runtime/rollout in Stage 4A-6.5al; no training/RL/PPO/BC/IL; no checkpoint
  or input observed_state/prediction NPZ modification; no prediction
  writeback/fusion; no prediction traversability/collision/ray
  blocking/candidate sampling/edge validity; no target/ground-truth/
  future-observed planning/scoring; no external source modification/build; no
  over-cost runtime promotion; no leakage; and no coverage-improvement claim.
- Recommended next small task:
  `Stage 4A-6.5am bounded repeat-safety smoke design/execution only`, either
  same scene/start with different `tree_seed` or same scene seed with
  alternate start, exactly two frames/one action. Do not recommend rollout
  directly.

Stage 4A-6.5al revalidation / Stage 4A-6.5am design update:

- Re-ran `py_compile` for
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_stage4a65al_post_action_two_frame.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65al_post_action_two_frame.py`.
- Re-ran the offline diagnosis with saved Stage 4A-6.5ak outputs plus
  Stage 4A-6.5ag/ai/aj prior context, `--max_workers 32 --save_viz`.
- Re-ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65al_post_action_two_frame_diagnosis_test.log`.
- Validation passed again: required outputs, hardware, sequence, diagnosis,
  input hashes, forbidden-artifact checks, and readiness/safety checks all
  passed.
- The re-run remained offline analysis only: no Isaac startup, no RGB/depth
  capture, no map_predict call, no SSCNet inference, no action execution, no
  two-frame runtime execution, no rollout/open-ended loop, no training, and no
  checkpoint/observed_state/prediction NPZ modification.
- Added Stage 4A-6.5am bounded repeat-safety smoke design artifacts in the
  Stage 4A-6.5al output directory:
  `stage4a65am_bounded_repeat_safety_smoke_design.json` and
  `stage4a65am_bounded_repeat_safety_smoke_design.md`.
- Added a topdown visualization manifest and copied already-saved Stage
  4A-6.5ak reference PNGs into the Stage 4A-6.5al output directory for
  Frame1/Frame2 observed maps, prediction overlays, branch/action/path views,
  and per-frame value components. These are references only, not new capture
  or runtime output.

Stage 4A-6.5am bounded repeat-safety smoke actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, plus the user-provided Stage 4A-6.5am execution spec.
- Confirmed Stage 4A-6.5ag/ai/aj/ak/al context and the current next small
  task: bounded repeat-safety smoke with same scene/start as Stage 4A-6.5ak
  and mini-RRT `tree_seed=1`.
- Preserved the runtime safety boundary: exactly one Isaac startup, at most
  two frames, at most one selected action, no second action, no third frame,
  no rollout/open-ended loop, no training/RL/PPO/BC/IL, no checkpoint change,
  no existing observed_state or prediction NPZ modification, no prediction
  writeback/fusion, no prediction traversability/collision/ray
  blocking/candidate sampling/edge validity, no target/ground-truth/
  future-observed planning/scoring, no external source build, no over-cost
  runtime primary, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65am_bounded_repeat_safety_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65am_bounded_repeat_safety_smoke.py`.
- Ran `py_compile` for the new runner/test plus `offline_mini_rrt_tree.py`,
  `sim_prediction_layer.py`, `isaac_map_predictor.py`, and
  `isaac_sscnet_preprocess.py`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65am_py_compile.log`.
- Ran the Stage 4A-6.5am bounded repeat-safety smoke with
  `--tree_seed 1 --reference_tree_seed 0 --max_workers 32 --save_viz`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65am_bounded_repeat_safety_smoke_tree_seed1.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65am_bounded_repeat_safety_smoke_tree_seed1`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65am_bounded_repeat_safety_smoke_tree_seed1_test.log`.

Stage 4A-6.5am conclusion:

- Validation passed.
- Runtime sequence:
  exactly `1` Isaac startup, exactly `2` frames, exactly `2` measured-only
  observed_state updates, exactly `2` map_predict calls, exactly `1`
  selected action execution, no second action, no third frame, and no rollout.
- Repeat variant:
  same `medium_three_rooms` scene seed `0`, same start pose
  `[-4.65, -4.65, 1.2]`, yaw `0.38710316317995463`, reference tree_seed
  `0`, current tree_seed `1`; only tree_seed was intentionally changed.
- Frame 1:
  measured-only shadow `n0001 -> n0157`, lambda48 primary `n0001 -> n0157`,
  lambda32 shadow `n0001 -> n0157`; lambda48 classification
  `same_as_measured`, low-cost artifact `false`, historical prior basin
  `false`; pre-action safety gates all passed.
- Executed exactly one action to `[-4.15, -4.75, 1.2]`, yaw
  `2.1587989303424653`.
- Frame 2:
  measured-only shadow `n0003 -> n0255`, lambda48 diagnostic
  `n0001 -> n0214`, lambda32 shadow `n0003 -> n0179`; lambda48
  classification `distinct_nonmeasured_branch`, low-cost artifact `false`,
  historical prior basin `false`.
- Observed_state delta:
  observed_ratio `0.0425462962962963 -> 0.057699074074074076`, delta
  `0.015152777777777777`, newly observed `6546`, unknown->free `5893`,
  unknown->occupied `653`, occupied->free `0`, invalid labels `0`.
- map_predict stability:
  Frame 1 valid/OCC+FREE `57382 / 40328`, Frame 2 valid/OCC+FREE
  `37258 / 27254`, density ratio `0.6758083713548899`, no
  explosion/collapse, both `code_consistent_v1`.
- Repeat comparison against Stage 4A-6.5ak:
  Frame 1 selected delta `0.2m`, Frame 2 selected delta
  `1.0816653826391969m`, action pose delta `0.20000000000000018m`.
  Repeat outcome classified as `divergent_but_healthy`.
- Safety passed:
  checkpoint unchanged; Stage 4A-6.5ak/6.5al reference observed_state and
  prediction NPZ inputs unchanged; generated prediction NPZs unchanged after
  creation; prediction read-only and information-gain-only; no prediction
  writeback/fusion; no prediction traversability/collision/ray
  blocking/candidate sampling/edge validity; no target/ground-truth/
  future-observed planning/scoring; no training/RL/PPO/BC/IL; no external
  source build; no over-cost runtime primary; no leakage; and no
  coverage-improvement claim.
- Recommended next small task:
  another bounded repeat review, likely alternate start or `tree_seed=2`
  design, still exactly two frames/one action. Do not recommend rollout
  directly.

Stage 4A-6.5an repeat-comparison review and next bounded-repeat design
actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, plus the user-provided Stage 4A-6.5an review/design spec.
- Preserved the offline-only boundary: no Isaac startup, no RGB/depth capture,
  no map_predict call, no SSCNet inference, no selected action execution, no
  two-frame runtime execution, no rollout/open-ended loop, no training/RL/PPO/
  BC/IL, no checkpoint change, no observed_state or prediction NPZ
  modification, no prediction writeback/fusion, no prediction traversability/
  collision/ray blocking/candidate sampling/edge validity, no target/
  ground-truth/future-observed planning/scoring, no external source build, no
  over-cost runtime primary, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/review_stage4a65an_repeat_comparison.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65an_repeat_comparison.py`.
- Ran `py_compile`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65an_py_compile.log`.
- Ran the Stage 4A-6.5an offline review with `--max_workers 32 --save_viz`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65an_repeat_comparison_and_next_design.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65an_repeat_comparison_and_next_design`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65an_repeat_comparison_and_next_design_test.log`.

Stage 4A-6.5an conclusion:

- Validation passed.
- Inputs loaded:
  Stage 4A-6.5ak, Stage 4A-6.5al, Stage 4A-6.5am, optional Stage 4A-6.5ag,
  and the three project context files.
- Sequence/safety reverified:
  Stage 4A-6.5ak tree_seed `0` and Stage 4A-6.5am tree_seed `1` each had
  exactly `2` frames, exactly `2` map_predict calls, exactly `1` selected
  action, no second action, no third frame, and no rollout.
- Frame 1 comparison:
  tree_seed `0` lambda48 selected `n0001 -> n0228`
  (`distinct_nonmeasured_branch`); tree_seed `1` lambda48 selected
  `n0001 -> n0157` (`same_as_measured`). Selected-child delta was `0.2m`;
  best-descendant delta was `1.2529964086141674m`.
- Frame 2 comparison:
  tree_seed `0` lambda48 selected `n0002 -> n0158`; tree_seed `1` lambda48
  selected `n0001 -> n0214`. Both were `distinct_nonmeasured_branch`.
  Selected-child delta was `1.0816653826391969m`; best-descendant delta was
  `4.548626166217664m`.
- Action/observed_state comparison:
  action pose delta was `0.20000000000000018m`. Tree_seed `0` observed_state
  delta was `0.026840277777777775` with `11595` newly observed; tree_seed `1`
  observed_state delta was `0.015152777777777777` with `6546` newly observed.
  The lower tree_seed `1` delta is plausible from the different one-action
  pose and not a label-safety regression.
- map_predict comparison:
  Frame 1 valid/OCC+FREE matched exactly at `57382 / 40328`. Frame 2 was
  `47814 / 30133` for tree_seed `0` and `37258 / 27254` for tree_seed `1`;
  density ratios were `0.7471979765919461` and `0.6758083713548899`. Both
  remained `code_consistent_v1`, read-only, and free of density
  explosion/collapse.
- Branch health:
  no low-cost artifact and no historical prior basin in either run/frame.
  Lambda32 matched lambda48 on both tree_seed `0` frames and tree_seed `1`
  Frame 1; tree_seed `1` Frame 2 lambda32 stayed measured-like while lambda48
  selected the distinct nonmeasured branch.
- Repeat outcome:
  `divergent_but_healthy`; this is tree_seed sensitivity without safety
  regression. The current evidence is still not enough for rollout.
- Future Stage 4A-6.5ao design:
  same scene/start bounded repeat-safety smoke with `tree_seed=2`, exactly two
  frames if safety gates pass, exactly two map_predict calls if action
  executes, exactly one selected action, no second action, no third frame, no
  rollout, formula `gain_exp / cost + 48 * minmax(source_occ_free)`,
  measured-only shadow, lambda32 shadow, prediction read-only/
  information-gain-only, and `--max_workers 32`.
- Safety passed:
  checkpoint unchanged; Stage 4A-6.5ak/6.5am observed_state and prediction
  NPZ hashes unchanged; no new 6.5an capture artifacts, observed_state `.npy`,
  map_predict NPZ, frame003, action002, transitions, rollout plots, or episode
  manifest were produced; no prediction writeback/fusion or motion-safety use;
  no leakage and no coverage-improvement claim.

Stage 4A-6.5ap seed0/1/2 repeat review and alternate-start design actions:

- Re-read project context from `CURRENT_STATE.md`, `CODEX_LOG.md`, and
  `TODO.md`, plus the user-provided Stage 4A-6.5ap review/design spec.
- Preserved the offline-only boundary: no Isaac startup, no RGB/depth capture,
  no map_predict call, no SSCNet inference, no selected action execution, no
  two-frame runtime execution, no rollout/open-ended loop, no training/RL/
  GDPO/PPO/BC/IL, no checkpoint change, no observed_state or prediction NPZ
  modification, no prediction writeback/fusion, no prediction traversability/
  collision/ray blocking/candidate sampling/edge validity, no target/
  ground-truth/future-observed planning/scoring, no external source build, no
  over-cost runtime primary, and no coverage-improvement claim.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/review_stage4a65ap_seed012_alternate_start_design.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65ap_seed012_alternate_start_design.py`.
- Ran `py_compile` for the new review/test scripts.
- Ran the Stage 4A-6.5ap offline review with `--max_workers 32 --save_viz`:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ap_seed012_repeat_review_alternate_start_design.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65ap_seed012_repeat_review_alternate_start_design`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65ap_seed012_repeat_review_alternate_start_design_test.log`.

Stage 4A-6.5ap conclusion:

- Validation passed: 65 required JSON/CSV/MD files and 10 required PNG plots
  exist, input hashes are unchanged, forbidden runtime/RL/rollout artifacts are
  absent, and the future command sketch is marked DO NOT RUN in 6.5ap.
- Inputs loaded:
  Stage 4A-6.5ak tree_seed `0`, Stage 4A-6.5am tree_seed `1`, Stage
  4A-6.5ao tree_seed `2`, supporting Stage 4A-6.5al/6.5an/6.5ag context,
  scene metadata, and the three project context files.
- Sequence/safety reverified:
  seed0/1/2 each had exactly `2` frames, exactly `2` map_predict calls,
  exactly `1` selected action, no second action, no third frame, and no
  rollout.
- Frame 1 lambda48 comparison:
  seed0 `n0001 -> n0228` (`distinct_nonmeasured_branch`), seed1
  `n0001 -> n0157` (`same_as_measured`), seed2 `n0001 -> n0248`
  (`same_as_measured`).
- Frame 2 lambda48 comparison:
  seed0 `n0002 -> n0158`, seed1 `n0001 -> n0214`, seed2
  `n0003 -> n0227`; all are `distinct_nonmeasured_branch`.
- Action pose differences were plausible tree_seed variation:
  seed0-vs-seed1 `0.20000000000000018m`, seed0-vs-seed2
  `0.22360679774997896m`, and seed1-vs-seed2 `0.4123105625617663m`.
- observed_state deltas remained positive and measured-only:
  seed0 `0.026840277777777775` / `11595`, seed1
  `0.015152777777777777` / `6546`, seed2
  `0.013023148148148148` / `5626`.
- map_predict stability remained clean:
  Frame 1 valid/OCC+FREE matched `57382 / 40328`; Frame 2 was seed0
  `47814 / 30133`, seed1 `37258 / 27254`, seed2 `32890 / 24936`, with no
  explosion/collapse and all `code_consistent_v1`.
- Branch health:
  no low-cost artifact and no historical prior basin in any seed/frame.
  Lambda32 matched lambda48 on seed0 both frames and seed2 both frames; seed1
  Frame 2 lambda32 stayed measured-like while lambda48 selected the distinct
  nonmeasured branch.
- Repeat outcome:
  `seed_sensitive_but_clean`. Seed2 is spatially consistent with seed1 and
  closer than seed1 to seed0 on Frame 2 selected child, so tree_seed
  sensitivity is reduced but not eliminated. Current evidence is still not
  rollout-ready.
- Future Stage 4A-6.5aq design:
  alternate-start bounded repeat-safety smoke at `start_corridor`, pose
  `[0.0, -4.45, 1.2]`, yaw `1.5707963267948966`, scene seed `0`, tree_seed
  `0` first, exactly two frames/one action, no second action, no third frame,
  no rollout, formula `gain_exp / cost + 48 * minmax(source_occ_free)`,
  measured-only shadow, lambda32 shadow, prediction read-only/
  information-gain-only, and `--max_workers 32`.
- Long-term GDPO-style RL remains future direction only. No RL/GDPO/PPO/BC/IL,
  replay buffer, policy checkpoint, or rollout data was produced in 6.5ap.

Stage 4A-6.5av start_room_b bounded smoke actions:

- Read project context plus Stage 4A-6.5au design artifacts, Stage 4A-6.5at
  review, Stage 4A-6.5aq/as start_corridor references, canonical-start
  references, and start_room_b metadata.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/run_stage4a65av_start_room_b_bounded_smoke.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a65av_start_room_b_bounded_smoke.py`.
- Ran py_compile:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65av_py_compile.log`.
- Ran the real bounded runtime smoke:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65av_start_room_b_bounded_smoke.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65av_start_room_b_bounded_smoke`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a65av_start_room_b_bounded_smoke_test.log`.

Stage 4A-6.5av conclusion:

- Validation passed with `all_passed: true`.
- Runtime matched Stage 4A-6.5au start_room_b design and metadata:
  `medium_three_rooms`, scene seed `0`, pose `[2.75, -2.55, 1.2]`, yaw
  `2.7052603405912112`, tree_seed `0`, formula
  `gain_exp / cost + 48 * minmax(source_occ_free)`.
- Sequence stayed bounded:
  exactly one Isaac startup, exactly two frames, exactly two map_predict calls,
  exactly one selected action, no second action, no third frame, no rollout,
  no formal expert sampling, no open-ended loop, no expert dataset, and no
  training/RL/GDPO/PPO/BC/IL.
- Frame 1 lambda48 was `same_as_measured`: measured/lambda48/lambda32 all
  selected `n0001 -> n0053`; low-cost artifact and historical prior basin were
  both `false`. The single action executed to
  `[2.25, -2.4499999999999997, 1.2]`, yaw `0.588002603547567`.
- Frame 2 lambda48 was `local_jitter`: measured-only selected
  `n0167 -> n0167`, lambda48/lambda32 selected `n0002 -> n0200`; low-cost
  artifact and historical prior basin were both `false`.
- observed_state/map_predict remained sane:
  observed_ratio delta `0.04165740740740741`, newly observed `17996`;
  Frame 1 valid/OCC+FREE `60060 / 53080`, Frame 2 `52286 / 33383`, density
  ratio `0.6289186134137151`, both `code_consistent_v1`.
- Prediction remained read-only and information-gain-only with no
  writeback/fusion, no traversability/collision/ray blocking, no candidate
  sampling or edge-validity use, and no target/ground-truth/future-observed
  scoring.
- Outcome:
  `spatially_consistent_healthy_start_room_b`. This is clean bounded evidence
  but still not rollout-ready and not formal expert sampling-ready.
- Next recommended gate:
  Stage 4A-6.6 `larger_complex_scene_v1` construction/validation, followed by
  Stage 4A-6.6a scene complexity audit before any formal expert sampling
  pilot. Long-term GDPO remains future direction only.

Stage 4A-6.6b GUI / visual inspection setup actions:

- Read the project context files plus the Stage 4A-6.6 validation bundle and
  Stage 4A-6.6a audit bundle.
- Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/inspect_stage4a66b_gui_visual_environment.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66b_gui_visual_environment.py`.
- Ran py_compile:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66b_py_compile.log`.
- Ran the Stage 4A-6.6b GUI/visual inspection setup:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66b_gui_visual_inspection.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection`.
- Ran validation:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66b_gui_visual_inspection_test.log`.

Stage 4A-6.6b conclusion:

- Validation passed with `all_passed: true`.
- GUI capability:
  DISPLAY was `:1`, WAYLAND_DISPLAY was `wayland-0`, XAUTHORITY was
  `/run/user/1000/.mutter-Xwaylandauth.IRQ0P3`, and the GPU was an NVIDIA
  GeForce RTX 5080.
- GUI attempt:
  one bounded GUI-mode IsaacSim probe was attempted. It did not establish a
  confirmed visible user inspection GUI (`gui_attempt_status: failed`), with
  DRI3 presentation warnings in the probe log. User visibility remains
  unconfirmed.
- Fallback visual package:
  Isaac headless rendering succeeded for 24 inspection views. The package has
  24/24 nonblank RGB views, 24/24 finite-positive depth views, topdown labeled
  maps, warning-region maps for `corridor_east_spur` / `room_j` / spur rooms,
  closeups, an MP4 flythrough, and an HTML index:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection/visual_inspection_index.html`.
- Manual review:
  `human_visual_inspection_done=false` and
  `user_needs_to_review_visuals=true`; the checklist is
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection/human_visual_review_checklist.md`.
- Negative scope remained clean:
  no expert sampling, no expert dataset, no selected action execution, no
  rollout, no open-ended loop, no map_predict, no SSCNet inference, no
  prediction NPZ, no replay buffer, no checkpoint creation/modification, no
  observed_state modification, no target/ground-truth/future observed scoring,
  no external source build, and no RL/GDPO/PPO/BC/IL. Stage 4A-6.7 was not
  executed.
- Next faithful step:
  user visual review of the HTML/images/MP4/checklist. If approved, proceed to
  Stage 4A-6.7 bounded formal expert sampling pilot, measured-only first. If
  rejected, run Stage 4A-6.6c scene visual revision / scene editing. Full
  expert dataset collection remains blocked; long-term GDPO remains future
  direction only.

Stage 4A-6.6c home_like_scene_v1 actions:

- Replaced the active large validation scene with `home_like_scene_v1` while
  preserving historical `minimal` and `medium_three_rooms` builders.
- Confirmed the old `build_larger_complex_scene_v1` path in
  `sim_explorer/scene_factory.py` is disabled and raises a Stage 4A-6.6c
  guidance error. The new scene builder provides a home-like layout with
  living room, kitchen, dining room, main/guest bedrooms, study, bathroom,
  laundry/storage, entry room, halls/corridors, doors/openings, loops, narrow
  passages, occluders, furniture, and non-cuboid primitives.
- Updated
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/build_stage4a66c_home_like_scene_v1.py`
  to actively remove old larger-scene output directories if present and to
  write a hardware utilization report.
- Old larger-scene output directories were absent at execution time:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66_larger_complex_scene_v1_validation`,
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66a_scene_complexity_audit`,
  and
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection`.
- Ran py_compile:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_py_compile.log`.
- First Isaac command attempts failed before rendering because `TERM=dumb`
  made `isaaclab.sh` fail at `tabs`, then because base conda Python lacked
  `isaaclab`. A later attempt hit `GLXBadFBConfig` with display variables set.
  Final successful run explicitly activated `env_isaaclab`, unset
  `DISPLAY/WAYLAND_DISPLAY/XAUTHORITY`, used `TERM=xterm`, and launched Isaac
  headless rendering with `--device cuda:0 --max_workers 32`.
- Successful Isaac log:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_home_like_scene_v1_validation.log`.
- Wrote outputs:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation`.

Stage 4A-6.6c conclusion (superseded by Stage 4A-6.6c-build-v2 below):

- Scene counts meet the requested thresholds: 9 rooms, 4 corridors/halls,
  19 openings, 69 wall cuboids, 83 total obstacles, 68 cuboid obstacles,
  15 non-cuboid primitives, 10 materials/colors, 10 start variants,
  18 validation poses, and 30 inspection poses.
- Isaac headless load validation passed with exactly one successful startup.
  The log and hardware report record NVIDIA GeForce RTX 5080 rendering on
  `cuda:0` and 32 requested/actual workers.
- Fixed captures passed: 18/18 validation RGB views nonblank, 18/18
  validation depth views finite-positive, 30/30 inspection RGB views nonblank,
  and 30/30 inspection depth views finite-positive.
- Measured-only observed_state integration passed: final shape
  `(240, 240, 30)`, labels limited to `-1/0/1`, invalid labels `0`,
  observed_ratio `0.19225752314814815`, observed_count `332221`,
  free_count `311183`, occupied_count `21038`, and unknown_count `1395779`.
- Visual package includes JSON/MD/CSV inventories, topdown maps, closeup maps,
  RGB/depth PNGs, HTML index, MP4 flythrough, and human visual review
  checklist:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation/visual_inspection_index.html`,
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation/home_like_scene_v1_flythrough.mp4`,
  and
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation/human_visual_review_checklist.md`.
- Local artifact audit passed: 378 files, MP4 nonzero, key PNGs nonblank,
  observed_state labels valid, and no forbidden rollout/action/prediction/
  replay/checkpoint artifacts were created. The only files containing words
  like expert/rollout/map_predict are explicit `no_*_report` safety reports.
- Negative scope remained clean:
  no rollout, no expert sampling, no selected action execution, no map_predict,
  no SSCNet inference, no prediction NPZ, no replay buffer, no policy
  checkpoint, no checkpoint modification, and no RL/GDPO/PPO/BC/IL.
- Manual review gate remains closed:
  `human_visual_inspection_done=false`,
  `formal_expert_sampling_ready=false`, and next step is Stage 4A-6.6d review
  + human visual confirmation. Stage 4A-6.7 was not executed.

Stage 4A-6.6c old-scene cleanup actions:

- User explicitly rejected `larger_complex_scene_v1` visual effect.
- Targeted only these old larger-scene output directories for deletion:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66_larger_complex_scene_v1_validation`,
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66a_scene_complexity_audit`,
  and
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection`.
- All three directories were already absent when cleanup began; `rm -rf` was
  still invoked on the exact user-specified paths, and post-cleanup
  verification confirmed they remain absent.
- Left unrelated outputs untouched, including checkpoint files,
  `medium_three_rooms` historical results, map_predict diagnostic history,
  SSCNet training/inference outputs, and other Stage 4A-6.5 artifacts.
- Confirmed `build_larger_complex_scene_v1` in
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py` is a disabled
  stub that raises `RuntimeError` and points callers to
  `build_home_like_scene_v1`.
- Wrote cleanup manifests:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation/old_scene_cleanup_manifest.json`
  and
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation/old_scene_cleanup_manifest.md`.
- Updated `CURRENT_STATE.md` and `TODO.md` to mark
  `larger_complex_scene_v1 rejected` and to block Stage 4A-6.7 from using the
  old larger scene or its old output bundles.
- No rollout, expert sampling, selected action execution, map_predict,
  SSCNet inference, checkpoint modification, replay buffer creation, RL, GDPO,
  BC, or IL was run during cleanup.

Stage 4A-6.6c-build-v2 asset-based home_like_scene_v1 actions:

- Superseded the earlier procedural/cuboid-heavy 6.6c `home_like_scene_v1`
  build. The old `larger_complex_scene_v1` remains rejected/disabled, and
  procedural composite furniture is not the main furniture solution.
- Found no sufficient preexisting local USD/mesh furniture set. Downloaded
  the Kenney Furniture Kit from `https://kenney.nl/assets/furniture-kit` /
  `https://kenney.nl/media/pages/assets/furniture-kit/440e0608a4-1677580847/kenney_furniture-kit.zip`.
  The package license is `Creative Commons Zero, CC0 1.0`; zip sha256 is
  `e67652d0932cee41683f74711c03d3e192a2af9979ef8e6b237711f5482d46b0`.
- Updated:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py` and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/build_stage4a66c_home_like_scene_v1.py`.
  The scene now records `furniture_assets`, `furniture_inventory`,
  `primitive_inventory`, `material_color_inventory`, `topology_graph`,
  local/downloaded asset manifests, and a conversion/import validation report.
- Converted 88 distinct Kenney GLB furniture assets to USD inside IsaacSim and
  spawned 108 mesh furniture instances. Conversion/import report:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation/conversion_import_validation_report.json`.
- After visual sanity check, added a glTF Y-up to Isaac Z-up orientation
  correction and default mesh spawn scale `2.0`; the final RGB grids show
  recognizable sofas, tables, beds, kitchen fixtures, bathroom fixtures, and
  storage furniture rather than tiny floor-level meshes.
- Final successful Isaac run used
  `/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python` with IsaacLab
  source paths in `PYTHONPATH`, `DISPLAY=` and `HEADLESS=1 ENABLE_CAMERAS=1`.
  Earlier attempts failed before capture because base conda Python lacked
  `isaaclab`, and one display-backed attempt hit `GLXBadFBConfig`.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation`.

Stage 4A-6.6c-build-v2 conclusion:

- Requested thresholds passed: 9 semantic rooms, 108 furniture mesh objects,
  108 non-cuboid/composite mesh asset instances, 14 material/color entries,
  10 start variants, 22 validation poses, and 38 inspection poses.
- Isaac fixed capture passed: 22/22 validation RGB nonblank, 22/22 validation
  depth finite-positive, 38/38 inspection RGB nonblank, and 38/38 inspection
  depth finite-positive.
- `observed_state_final.npy` passed: shape `(240, 240, 30)`, labels contain
  `UNKNOWN/FREE/OCCUPIED` (`-1/0/1`), invalid labels `0`, observed_ratio
  `0.21396354166666667`.
- Visual inspection package passed artifact checks and includes:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation/visual_inspection_index.html`,
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation/home_like_scene_v1_flythrough.mp4`,
  and
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation/human_visual_review_checklist.md`.
- Negative scope stayed clean:
  no expert sampling, no rollout, no selected action execution, no
  map_predict, no SSCNet inference, no prediction NPZ, no replay buffer, no
  checkpoint creation/modification, and no RL/GDPO/PPO/BC/IL.
- Next required step:
  Stage 4A-6.6d home-like scene audit + human visual review. Keep
  `human_visual_inspection_done=false` and
  `formal_expert_sampling_ready=false` until the user approves the visual
  package.

Stage 4A-6.6c-build-v2 deletion actions:

- User requested deleting the just-generated scene after viewing the actual
  IsaacSim top-down render.
- Deleted the full generated output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_home_like_scene_v1_validation`.
  This removed the visual package, RGB/depth captures,
  `observed_state_final.npy`, downloaded/converted Kenney assets, manifests,
  HTML, MP4, and IsaacSim top-down render.
- Deleted the temporary top-down renderer script:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/render_stage4a66c_home_like_sim_topdown.py`.
- Confirmed both paths are absent after deletion.
- Left unrelated outputs, checkpoints, historical logs, and other scene
  builders untouched. `larger_complex_scene_v1` remains rejected/disabled.
- No rollout, expert sampling, selected action execution, map_predict,
  SSCNet inference, prediction NPZ, checkpoint modification, replay buffer,
  RL, GDPO, PPO, BC, or IL was run during deletion.
- Current gate:
  no `home_like_scene_v1` visual package exists on disk, so Stage 4A-6.6d
  audit and any later expert sampling remain blocked until a replacement scene
  package is built and approved.

Stage 4A-6.6c-usd-import user USD staging and blocked validation:

- Used `/home/ubuntu22/sc_explorer_ws/building_scene.usd` as the user-provided
  current project candidate environment for `home_like_scene_v1`.
- Staged it without modification to
  `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment/home_like_scene_v1.usd`.
  Source/staged sha256:
  `11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b`.
- Updated `/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py` so the
  active `build_home_like_scene_v1` delegates to a staged-USD loader. The
  loader spawns `/World/HomeLikeSceneV1` from the staged USD when requested and
  reports metadata without generating a procedural fallback. The old
  `larger_complex_scene_v1` remains rejected/disabled as a `RuntimeError`
  stub, and the old 6.6 / 6.6a / 6.6b output directories remain absent.
- Added
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/import_stage4a66c_usd_home_like_scene.py`
  for staging, offline USD reports, start/pose proposals, a single Isaac
  validation attempt, capture/observed-state packaging on success, and blocker
  reports on failure.
- Added/updated
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66c_usd_home_like_scene.py`.
  It keeps the success validation path and now also supports
  `--allow_blocked_isaac_load` to verify a faithful blocked package without
  pretending capture succeeded.
- Offline USD inspection succeeded: 758 prims, 99 references, 67 unresolved/
  missing dependencies, Z-up, `metersPerUnit=1.0`, 1 mesh prim, 2 materials,
  0 texture references, and semantic-name guesses for home/interior categories
  including sofa, table, bed, bathroom, cabinet/shelf, chair, door/window,
  hallway/corridor/stair/elevator, room, floor, and wall.
- The formal Isaac headless validation was started exactly once. It failed
  while loading/resolving the staged USD with `LLVM ERROR: out of memory`; the
  Kit log showed unresolved remote Omniverse HTTPS dependencies. No second
  Isaac validation attempt was made.
- Because the Isaac load was blocked, no validation RGB/depth captures,
  inspection RGB/depth captures, measured `observed_state_final.npy`, or
  `usd_scene_flythrough.mp4` were created. The blocked output package contains
  offline USD reports, start/pose manifests, topdown PNGs, closeups, blocker
  reports, a blocked `visual_inspection_index.html`, and closed manual-review
  gates.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_home_like_scene_validation`.
- Logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_usd_home_like_scene_validation.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_usd_home_like_scene_py_compile.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_usd_home_like_scene_validation_test.log`.
- Blocked-mode validation now reports `all_passed: true` for the evidence
  package: staging/hash match, old-scene disablement, offline reports, blocker
  state, closed gates, and forbidden-output absence. This is not a successful
  scene-load/capture validation.
- Negative scope stayed clean:
  no procedural scene generation, no asset download, no rollout, no expert
  sampling, no selected action execution, no map_predict, no SSCNet inference,
  no prediction NPZ, no replay buffer, no checkpoint creation/modification,
  and no RL/GDPO/PPO/BC/IL.
- Current gates:
  `human_visual_inspection_done=false`,
  `user_needs_to_review_visuals=true`,
  `formal_expert_sampling_ready=false`,
  `full_expert_dataset_ready=false`, and `stage4a67_executed=false`.
- Next faithful step:
  provide a self-contained/local dependency package or lighter fully local USD,
  then rerun Stage 4A-6.6c-usd-import. Stage 4A-6.6d and Stage 4A-6.7 remain
  blocked until a successful Isaac load, RGB/depth capture, observed-state
  generation, and human visual review.

Stage 4A-6.6c-usd-download-official-isaac-deps actions:

- User allowed exact official Isaac dependency downloads for the current USD
  dependency repair path only. Created:
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/download_stage4a66c_official_isaac_deps.py`
  and
  `/home/ubuntu22/sc_explorer_ws/sim_explorer/test_stage4a66c_official_isaac_deps.py`.
- Read and confirmed the previous dependency blocker context:
  99 remote reference occurrences, 67 unique official Isaac S3/USD assets,
  0 local candidates, no prior dependency localization, no prior Isaac retry
  from the dependency-fix pass, no validation RGB/depth, no
  `observed_state_final.npy`, and no MP4.
- Downloaded/localized only exact official Isaac dependencies from
  `dependency_package_request.md` and recursive dependencies discovered inside
  those USDs under the same official base
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/`.
  Final local package:
  `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment/dependencies`.
  Package contains 278 files: 67 USD, 23 MDL, 187 PNG, and 1 DDS, totaling
  511,952,260 bytes.
- Created localized package root:
  `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized`
  and localized USD:
  `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized/home_like_scene_v1.usd`.
  Patched root remote references via PXR/Sdf text export from official URLs
  to local `./dependencies/Assets/Isaac/4.5/Isaac/...` paths. Source USD and
  original staged USD remained unchanged with sha256
  `11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b`.
- Post-patch closure report:
  remote official refs remaining `0`, `omniverse://` refs remaining `0`,
  unsupported external deps `0`, and unresolved local deps `0`.
- Isaac retry gate was allowed by dependency closure and was attempted exactly
  once. The retry failed before capture because the localized USD has no
  `defaultPrim`; Kit reported unresolved reference prim path
  `@.../current_environment_localized/home_like_scene_v1.usd@<defaultPrim>`
  for `/World/HomeLikeSceneV1`. No validation RGB/depth,
  `observed_state_final.npy`, visual inspection HTML, MP4, or flythrough was
  produced or fabricated.
- Output directory:
  `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_download_official_isaac_deps`.
  Logs:
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_download_official_isaac_deps_py_compile.log`,
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_download_official_isaac_deps.log`,
  and
  `/home/ubuntu22/sc_explorer_ws/logs/stage4a66c_download_official_isaac_deps_test.log`.
- Validation reports `all_passed: true` for required outputs, context,
  previous blocker evidence, URL count, manifests, localized package, patch
  report, hash preservation, blocked retry evidence, and negative scope.
- Negative scope stayed clean:
  no rollout, no selected action, no expert sampling, no expert dataset,
  no map_predict, no SSCNet inference, no prediction NPZ, no replay buffer,
  no checkpoint creation/modification, and no RL/GDPO/PPO/BC/IL.
- Gates remain closed:
  `human_visual_inspection_done=false`,
  `formal_expert_sampling_ready=false`,
  `full_expert_dataset_ready=false`, `stage4a66d_executed=false`, and
  `stage4a67_executed=false`.
- Next faithful step:
  fix the localized USD load target by setting/using a concrete root prim
  (likely `World`) instead of `<defaultPrim>`, then run a new explicitly
  authorized validation attempt. Do not enter Stage 4A-6.6d or Stage 4A-6.7
  until successful RGB/depth capture, measured `observed_state_final.npy`, and
  human visual review exist.

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
