Project: SC-Explorer SSC network real-data dry-run
Updated: 2026-05-29

Current continuation status:

- User manually downloaded and extracted real NYU repackaged NPZ data. Do not re-download it.
- Real data roots:
  - /home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtrain_npz
  - /home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz
- File counts:
  - train: 794 .npz files
  - test: 654 .npz files
- ssc_exploration/ssc_network/config.py now points NYU train/val to those real roots.
- All Python commands in this continuation were run through env_isaaclab.
- CUDA is not visible inside the default command sandbox for torch, so model forward, train.py, and test.py were run with escalated execution while still using env_isaaclab.

Completed checks:

- Real NPZ field check: passed.
  - Required fields present: rgb, depth, tsdf_hr, tsdf_lr, target_lr, position.
  - Extra field target_hr is present and harmless for current dataloader.
  - target_lr sampled non-ignore labels are within 0..11.
  - target_lr 255 ignore_index is present in every sampled file.
  - Log: /home/ubuntu22/sc_explorer_ws/logs/real_npz_field_check.log
- Dataloader smoke test: passed.
  - train_batches=794, val_batches=654.
  - Log: /home/ubuntu22/sc_explorer_ws/logs/real_dataloader_smoke.log
- Model forward smoke test: passed.
  - Output shape: (1, 12, 60, 36, 60).
  - Output finite: True.
  - Log: /home/ubuntu22/sc_explorer_ws/logs/real_model_forward_smoke.log
- Real data 1 epoch dry-run: passed.
  - Command used --epochs=1, --batch_size=1, --workers=0.
  - No RL, Unreal, AirSim, or 50 epoch training was run.
  - Log: /home/ubuntu22/sc_explorer_ws/logs/ssc_network_real_dryrun.log
  - Checkpoints:
    - /home/ubuntu22/sc_explorer_ws/checkpoints/real_dryrun/cp_SSCNet_real_dryrun.pth.tar
    - /home/ubuntu22/sc_explorer_ws/checkpoints/real_dryrun/cpBest_SSCNet_real_dryrun.pth.tar
- test.py dry-run from generated checkpoint: passed.
  - Log: /home/ubuntu22/sc_explorer_ws/logs/ssc_network_real_test_dryrun.log
  - Result: p 54.2, r 85.0, IoU 48.6, pixel-acc 41.5239, mean IoU 11.2.

Important constraints for future continuation:

- Do not download the dataset again.
- Do not run RL, Unreal, AirSim, or full 50 epoch training unless explicitly requested.
- For SSC network Python commands, use env_isaaclab.
- For CUDA-dependent Python commands, default sandbox may fail with "No CUDA GPUs are available"; use the already approved escalated command pattern or request approval if needed.
