# SC-Explorer ssc_network Training Notes

## 0. Goal

Prepare and dry-run the official SC-Explorer `ssc_network` training flow for a PALNet-style 3D scene completion / `map_predict` network. This stage does not do RL, PPO, imitation learning, planner integration, Unreal, or AirSim.

## 1. Local Context Summary

Current project direction is SC-Explorer + RL for indoor robot active exploration. The current task is only to inspect, prepare, and run the SC-Explorer prediction network training flow. Prediction is a future `map_predict` signal and must not overwrite the measured map. A deterministic SC-Explorer-style expert may later be used as an imitation learning teacher, but RL should not start before the deterministic expert and prediction pipeline are understood.

## 2. Environment

Workspace:

```text
/home/ubuntu22/sc_explorer_ws
```

System checks:

```text
Ubuntu 22.04.5 LTS (jammy)
Linux ubuntu22 6.8.0-111-generic x86_64
ROS_DISTRO=humble
rosversion: command not found
conda 26.3.2
git version 2.34.1
NVIDIA GeForce RTX 5080, driver 580.159.03, reported CUDA runtime 13.0
nvcc: command not found
```

Created directories:

```text
/home/ubuntu22/sc_explorer_ws/logs
/home/ubuntu22/sc_explorer_ws/checkpoints
/home/ubuntu22/sc_explorer_ws/data
```

## 3. env_isaaclab Status

`env_isaaclab` exists and was used for all Python, pip, torch, and training commands.

```text
CONDA_DEFAULT_ENV=env_isaaclab
python: /home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python
pip: /home/ubuntu22/miniconda3/envs/env_isaaclab/bin/pip
Python: 3.11.15
pip: 26.1.1
torch: 2.7.0+cu128
torch cuda: 12.8
cuda available: True
cuda device: NVIDIA GeForce RTX 5080
numpy: 1.26.0
```

Dependency status after work:

```text
torch OK 2.7.0+cu128
torch_scatter OK 2.1.2+pt27cu128
numpy OK 1.26.0
scipy OK 1.15.3
h5py OK 3.16.0
cv2 OK 4.11.0
matplotlib OK 3.10.3
imageio OK 2.37.0
tqdm OK 4.67.3
torchvision OK 0.22.0+cu128
yaml OK 6.0.2
sklearn MISSING
skimage MISSING
```

`skimage` is not required by the official `ssc_network/requirements.txt`. `sklearn` is required by `utils/ssc_metrics.py`; network download failures prevented installation, so a local fallback was added for the small dry-run.

## 4. Repository Status

Repository:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration
```

Commit:

```text
3b1492e47a2e62363328cc931d9ef2342fb55891
```

Initial status after clone was clean on `main`. After dry-run preparation, the official clone has local compatibility/config changes:

```text
M ssc_network/config.py
M ssc_network/models/SSCNet.py
M ssc_network/models/__init__.py
M ssc_network/train.py
M ssc_network/utils/ssc_metrics.py
```

No main MapExRL project code was modified.

## 5. ssc_network Structure

Files:

```text
./CMakeLists.txt
./README.md
./config.py
./dataloaders/__init__.py
./dataloaders/dataloader.py
./models/SSCNet.py
./models/__init__.py
./package.xml
./pretrained_models/weights/PALNet.pth.tar
./requirements.txt
./src/ssc_network_node.py
./test.py
./test.sh
./train.py
./train.sh
./utils/projection_layer.py
./utils/seed.py
./utils/ssc_metrics.py
./utils/utils.py
./voxel_utils/README.md
./voxel_utils/makefile
./voxel_utils/scripts/configure.py
./voxel_utils/setup.py
./voxel_utils/voxel_util.cpp
./voxel_utils/voxel_util.cu
./voxel_utils/voxel_util_module.c
```

Directories:

```text
.
./dataloaders
./models
./pretrained_models
./pretrained_models/weights
./src
./utils
./voxel_utils
./voxel_utils/scripts
```

## 6. Official README Summary

The README says the scene completion network is adapted from PALNet. Requirements are PyTorch >= 1.4.0, `torch_scatter`, `imageio`, `scipy`, `scikit-learn`, and `tqdm`.

The README says data can be obtained from SSCNet raw data or repackaged data from Google Drive / BaiduYun. The repackaged `.npz` samples include:

```text
rgb
depth
tsdf_hr      (240, 144, 240)
tsdf_lr      (60, 36, 60)
target_hr    (240, 144, 240)
target_lr    (60, 36, 60)
position
```

The README says dataset roots are configured by editing `config.py`. Training is run by `bash train.sh`; testing is run by `bash test.sh`.

The README mentions `infer_ros.py --model palnet --resume trained_model.pth`, but the checked-out repository does not contain `infer_ros.py`. The available ROS inference entry is `src/ssc_network_node.py`.

## 7. Dependencies

Official `requirements.txt`:

```text
torch==1.4.0
torchvision==0.5.0
torch-scatter -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
imageio
scipy
scikit-learn
tqdm
```

High-risk dependency locks:

```text
torch==1.4.0
torchvision==0.5.0
torch-scatter wheel URL for torch 1.4.0 CPU
```

The full requirements file was not installed because it would downgrade/replace the existing IsaacLab PyTorch/CUDA stack. Instead, only `torch-scatter` was installed as a no-deps binary wheel matching torch 2.7.0 + CUDA 12.8:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python -m pip install --no-deps --only-binary=:all: torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

`scikit-learn` install was attempted with no-deps binary wheels, but failed repeatedly due network `IncompleteRead` errors.

## 8. Training Script

Training entry:

```text
ssc_network/train.py
ssc_network/train.sh
```

Main function:

```text
train.py: main() -> train()
```

Arguments from `train.py`:

```text
--dataset choices: nyu, nyucad, debug; default nyu
--model choices: sscnet; default sscnet
--epochs default 50
--lr default 0.01
--lr_adj_n default 100
--lr_adj_rate default 0.1
--batch_size default 4
--workers default 4
--resume checkpoint path
--checkpoint checkpoint output directory prefix
--model_name checkpoint name suffix
```

Dataset root is not an argument. It is specified in `config.py` via `Path.db_root_dir(dataset)`. Train/val split is a pair of directories returned by `config.py`, e.g. `{'train': ..., 'val': ...}`.

Checkpoint save paths are:

```text
args.checkpoint + 'cp_<model_name>.pth.tar'
args.checkpoint + 'cpBest_<model_name>.pth.tar'
```

Resume:

```text
--resume /path/to/checkpoint.pth.tar
```

Optimizer and loss:

```text
torch.optim.SGD(lr=args.lr, weight_decay=0.0005, momentum=0.9)
torch.nn.CrossEntropyLoss(weight=config.class_weights, ignore_index=255)
```

The training loss is semantic 12-class cross entropy over the voxel target. Occupancy metrics are computed in validation, but there is no separate occupancy loss in the code.

README/train.sh vs code:

```text
train.py default epochs = 50
train.py default lr = 0.01
train.py default lr_adj_n = 100
train.sh sets lr_adj_n = 10 and lr_adj_rate = 0.1
train.sh sets batch_size = 1 and epochs = 50
```

So the 50 epochs / lr 0.01 / every 10 epochs decay 0.1 behavior is present in `train.sh`, but `train.py` default `lr_adj_n` is 100.

Compatibility fixes made for dry-run:

```text
models/__init__.py: fixed invalid f-string in fallback print.
models/SSCNet.py: added import fallback for top-level `from models import make_model`.
models/SSCNet.py: allowed `x_rgb` keyword in forward because train.py passes x_rgb.
train.py: adjusted validation return unpacking from 6 to 7 values.
utils/ssc_metrics.py: added small fallback for accuracy/precision/recall when sklearn is unavailable.
config.py: set nyu train/val roots to /home/ubuntu22/sc_explorer_ws/data/NYUtrain_npz and NYUtest_npz.
```

## 9. Evaluation Script

Evaluation entry:

```text
ssc_network/test.py
ssc_network/test.sh
```

`test.py` arguments:

```text
--dataset choices: nyu, nyucad, debug; default nyu
--model choices: sscnet; default sscnet
--batch_size default 4
--workers default 4
--resume checkpoint path
```

Checkpoint loading:

```text
cp_states = torch.load(args.resume)
net.load_state_dict(cp_states['state_dict'], strict=True)
```

Test data is selected by `--dataset`, then `config.py` supplies the `val` directory. There is no direct test root argument.

Metrics printed:

```text
occupancy precision p
occupied recall r
occupancy IoU
pixel accuracy
semantic per-class SSC IoU
semantic mean IoU
occupancy calibration binary
occupancy calibration semantic
testing wall time
```

There is no explicit FPS metric.

## 10. Inference Script

README mentions:

```text
infer_ros.py --model palnet --resume trained_model.pth
```

But `infer_ros.py` is missing in the checked-out repository. The actual ROS inference file is:

```text
ssc_network/src/ssc_network_node.py
```

`src/ssc_network_node.py` supports:

```text
--input_topic_name default /airsim_drone/Depth_cam
--output_topic_name default /ssc
--world_frame default /odom
--model choices ddrnet, palnet, palnet_ours; default palnet
--resume checkpoint path
```

Input ROS message:

```text
sensor_msgs.msg.Image
```

Output ROS message:

```text
ssc_msgs.msg.SSCGrid
```

It subscribes to `input_topic_name` and publishes `SSCGrid` to `output_topic_name`.

The node uses `torch.cuda.is_available()` and falls back to CPU if CUDA is unavailable. It needs ROS Python packages (`rospy`, `sensor_msgs`, `tf`, `cv_bridge`) and `VoxelUtils` for TSDF and projection computation from depth images.

The callback computes `scores = Softmax(dim=0)(y_pred.squeeze())`, then encodes class id plus free-space confidence into `SSCGrid.data`. It does not publish the full class probability tensor; it publishes an encoded prediction grid. Output grid dimensions are taken from `preds.shape`, expected `(60, 36, 60)` after model output.

Important mismatch: `src/ssc_network_node.py` calls `make_model(self.args.model, num_classes=12)` with `palnet`, but `models/__init__.py` only recognizes `sscnet` and otherwise falls back to `SSCNet`.

## 11. Model Architecture

Model file:

```text
ssc_network/models/SSCNet.py
```

Model class:

```text
SSCNet(nn.Module)
```

Model registry:

```text
ssc_network/models/__init__.py
make_model(modelname, num_classes)
```

Only `sscnet` is explicitly supported. There is no separate `PALNet` class in the checked-out code.

Branches:

```text
Depth branch: yes, 2D Conv2d on a single-channel depth image.
2D-to-3D projection: yes, Project2Dto3D using torch_scatter.scatter_max.
3D TSDF branch: no active branch in current SSCNet forward. x_tsdf exists in the signature but is not used.
RGB branch: no active branch in current SSCNet forward. train.py passed x_rgb before compatibility fix, but SSCNet ignores it.
Flipped TSDF: data contains tsdf_hr/tsdf_lr, but current SSCNet forward does not consume TSDF.
```

The current network is a light depth-only 2D-to-3D projection + 3D CNN. It includes grouped 3D convolutions and reduced channel counts, which appears to be a lightweight modification relative to the stored `PALNet.pth.tar` weights.

## 12. Dataset / Dataloader

Dataloader file:

```text
ssc_network/dataloaders/dataloader.py
```

Dataset class:

```text
NYUDataset(torch.utils.data.Dataset)
```

Dataloader factory:

```text
ssc_network/dataloaders/__init__.py
make_data_loader(args)
```

The active path uses `.npz` files only:

```text
self.subfix = 'npz'
glob(root + '/*.npz')
```

For training (`istest=False`), each sample returns:

```text
rgb_tensor
depth_tensor
tsdf_hr
target_lr.T
position
filename
```

For validation/testing (`istest=True`), each sample returns:

```text
rgb_tensor
depth_tensor
tsdf_hr
target_lr.T
nonempty.T
position
filename
```

Fields read from `.npz`:

```text
rgb
depth
tsdf_hr
target_lr
position
tsdf_lr  # only for test/validation nonempty mask
```

The README also lists `target_hr`, but the active `.npz` dataloader path does not read it.

There is legacy code for SSCNet-style raw files:

```text
.png depth
rgb.png image
.bin RLE 3D labels
.npz tsdf
```

But that path is not active because `self.subfix = 'npz'`.

## 13. NYU Data Requirements

The code does not directly consume the official NYU Depth V2 labeled `.mat` file. It expects preprocessed/repackaged PALNet/SSCNet-style `.npz` files.

Required for the active `.npz` path:

```text
NYUtrain_npz/*.npz
NYUtest_npz/*.npz
```

Required fields:

```text
rgb
depth
tsdf_hr
tsdf_lr
target_lr
position
```

The README says the repackaged data can be downloaded from the linked Google Drive or BaiduYun folders. The raw data source is SSCNet.

## 14. 3D Annotation Requirements

Supervised 3D scene completion training requires 3D voxel labels. In this code, the supervised label used by loss is `target_lr`, a 12-class voxel grid with ignore index 255.

The official NYU Depth V2 2D labels alone are not sufficient for this dataloader. The needed 3D labels are in the PALNet/SSCNet-style repackaged `.npz` data or legacy SSCNet `.bin` RLE annotations.

Blocker for real supervised training:

```text
Cannot start supervised 3D scene completion training until required 3D voxel annotations / repackaged NPZ files are obtained.
```

## 15. Expected Data Directory Structure

Configured dry-run structure:

```text
/home/ubuntu22/sc_explorer_ws/data/
  NYUtrain_npz/
    sample_train_0000.npz
  NYUtest_npz/
    sample_test_0000.npz
```

Expected real structure after downloading real repackaged data:

```text
/path/to/NYUtrain_npz/*.npz
/path/to/NYUtest_npz/*.npz
```

Then edit `ssc_network/config.py`:

```python
return {'train': '/path/to/NYUtrain_npz',
        'val': '/path/to/NYUtest_npz'}
```

Data download plan:

| Data item | Required? | Size | Source | Used for | Download now? |
|---|---:|---:|---|---|---|
| NYU Depth V2 labeled dataset | Not directly for active dataloader | ~2.8GB | NYU Depth V2 official | Possibly source RGB-D/2D labels, but not enough for current training path | No, unless preprocessing path is chosen |
| NYU Depth V2 raw dataset | No for active dataloader | ~428GB | NYU Depth V2 raw | Raw RGB-D stream, not consumed by active `.npz` dataloader | No |
| 3D voxel ground truth annotations | Yes | Unknown here | SSCNet raw data or repackaged PALNet/SSCNet data linked in README | Supervised target voxel labels (`target_lr`) | Yes, but only from confirmed repackaged/annotation source |
| pretrained weights | Optional | 914KB in repo | `pretrained_models/weights/PALNet.pth.tar` | Pretrained inference/checkpoint load | Already present, but incompatible with current model |
| PALNet / SSCNet preprocessed `.npz` data | Yes for easiest training | Unknown here | README Google Drive / BaiduYun links | Dataloader input and labels | Yes, preferred over raw NYU |

## 16. Input / Output Shapes

From README/dataloader/model:

```text
depth input: expected tensor shape (B, 1, 480, 640) in dry-run
position mapping: expected tensor shape (B, 480, 640), flattened to projection indices
rgb: present in dataset, not used by current SSCNet forward
tsdf_hr: present in dataset, not used by current SSCNet forward
tsdf_lr: used only to compute validation nonempty mask
target_lr: (60, 36, 60)
```

Model output:

```text
logits shape: (B, 12, 60, 36, 60)
classes: 12
```

Dry-run random forward result:

```text
output shape: (1, 12, 60, 36, 60)
elapsed_sec: 0.6476
peak_mem_mib: 749.34
```

Voxel size:

```text
high-res voxel unit in dataloader: 0.02m
downsample = 4
low-res output voxel size: 0.08m
output resolution: 60 x 36 x 60
```

This is equivalent to the paper-style `60 x 60 x 36` grid under a different axis ordering.

## 17. Pretrained Weights

Found pretrained file:

```text
ssc_network/pretrained_models/weights/PALNet.pth.tar
size: 914KB
sha256: 973436395655de03025e6bfe917603be3439f8f2c8fd0521be473a7e9d22cc29
```

Checkpoint can be read with torch:

```text
type: dict
top-level keys: ['state_dict']
state_dict len: 68
first key: conv2d_depth.0.weight
first tensor shape: (6, 1, 3, 3)
```

## 18. Pretrained Inference Result

Checkpoint load into current `SSCNet` failed due architecture mismatch:

```text
size mismatch for pool1.weight:
checkpoint [8, 6, 7, 7, 7], current model [8, 6, 4, 4, 4]
size mismatch for conv2_1.2.weight:
checkpoint [8, 8, 3, 3, 3], current model [8, 4, 3, 3, 3]
size mismatch for conv3_1.2.weight:
checkpoint [8, 8, 3, 3, 3], current model [8, 4, 3, 3, 3]
size mismatch for conv3_3.2.weight:
checkpoint [32, 32, 3, 3, 3], current model [32, 8, 3, 3, 3]
...
```

Conclusion: the bundled `PALNet.pth.tar` is not directly compatible with the current lightweight `SSCNet.py` model definition. Pretrained inference with this checkpoint did not run. Untrained model forward did run and produced `(1, 12, 60, 36, 60)`.

## 19. Dry-run Training Result

Dry-run data:

```text
/home/ubuntu22/sc_explorer_ws/data/NYUtrain_npz/sample_train_0000.npz
/home/ubuntu22/sc_explorer_ws/data/NYUtest_npz/sample_test_0000.npz
```

These are synthetic NYU-style `.npz` files matching the dataloader fields. They are only for pipeline verification, not for model quality.

Dry-run command:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
echo "$CONDA_DEFAULT_ENV"
which python
mkdir -p /home/ubuntu22/sc_explorer_ws/checkpoints/dryrun
export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:$PYTHONPATH
python ./train.py --model=sscnet --dataset=nyu --epochs=1 --batch_size=1 --workers=0 --lr=0.01 --lr_adj_n=10 --lr_adj_rate=0.1 --checkpoint=/home/ubuntu22/sc_explorer_ws/checkpoints/dryrun/ --model_name=SSCNet_dryrun 2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/ssc_network_dryrun.log
```

Result:

```text
Training data:/home/ubuntu22/sc_explorer_ws/data/NYUtrain_npz
Dataset:1 files
Validate data:/home/ubuntu22/sc_explorer_ws/data/NYUtest_npz
Dataset:1 files
Training epochs:1
Initial Learning rate:0.01
Batch size:1
Number of workers:0
Training epoch 1/1: 1step, ~3.53 step/s
Validating: 1frame, ~6.34 frame/s
Validate: epoch 1, p 0.3, r 100.0, IoU 0.3
Training finished in: 0:00:00.575962
```

Checkpoint:

```text
/home/ubuntu22/sc_explorer_ws/checkpoints/dryrun/cp_SSCNet_dryrun.pth.tar
size: 436KB
```

Official test entry also ran on the dry-run checkpoint:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:$PYTHONPATH
python ./test.py --model=sscnet --dataset=nyu --batch_size=1 --workers=0 --resume=/home/ubuntu22/sc_explorer_ws/checkpoints/dryrun/cp_SSCNet_dryrun.pth.tar 2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/ssc_network_test_dryrun.log
```

Test result:

```text
Validate with TSDF: p 0.3, r 100.0, IoU 0.3
pixel-acc 0.0000, mean IoU 0.0
Testing finished in: 0:00:00.385151
```

## 20. Full Training Command Draft

Do not run this until real repackaged NYU/SSCNet-style `.npz` training and test data are downloaded and `config.py` points to the real directories.

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
echo "$CONDA_DEFAULT_ENV"
which python
mkdir -p /home/ubuntu22/sc_explorer_ws/checkpoints/full_train
mkdir -p /home/ubuntu22/sc_explorer_ws/logs/full_train
export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:$PYTHONPATH
cd /home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network
python ./train.py \
  --model=sscnet \
  --dataset=nyu \
  --epochs=50 \
  --batch_size=1 \
  --workers=1 \
  --lr=0.01 \
  --lr_adj_n=10 \
  --lr_adj_rate=0.1 \
  --checkpoint=/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/ \
  --model_name=SSCNet_NYU_full \
  2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/full_train/SSCNet_NYU_full_train.log
```

## 21. Errors and Fixes

Error: README references `infer_ros.py`, but file is missing.

Fix/status: use `src/ssc_network_node.py` as the actual ROS inference entry. Full ROS inference was not run.

Error: `requirements.txt` pins old torch/torchvision and torch-scatter CPU wheel for torch 1.4.0.

Fix/status: did not install full requirements. Installed only compatible `torch-scatter` wheel into `env_isaaclab` with no deps.

Error: `models/__init__.py` had invalid f-string `f"Unknown model '{}', ..."` under Python 3.11.

Fix: changed it to include `modelname`.

Error: `models/SSCNet.py` used a relative import that failed when `train.py` imports `models` as a top-level package.

Fix: added fallback import from `utils.projection_layer`.

Error: `SSCNet.forward()` did not accept `x_rgb`, but `train.py` and validation call it with `x_rgb=...` for `sscnet`.

Fix: added optional `x_rgb=None` keyword to forward. Current model ignores RGB.

Error: `train.py` expected 6 return values from `validate_on_dataset`, but the function returns 7.

Fix: ignored the seventh calibration return in training.

Error: `scikit-learn` failed to install due network `IncompleteRead` errors; `utils/ssc_metrics.py` imports sklearn.

Fix: added small local fallback implementations for `accuracy_score` and binary `precision_recall_fscore_support` so dry-run can proceed. Real training should install scikit-learn when network is stable.

Error: bundled `PALNet.pth.tar` does not match current model shapes.

Fix/status: no architecture rewrite was done. Pretrained load remains blocked for this checkpoint/model pair.

## 22. env_isaaclab Changes Made

Installed:

```text
torch-scatter 2.1.2+pt27cu128
```

Command:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python -m pip install --no-deps --only-binary=:all: torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

Attempted but not installed:

```text
scikit-learn
```

Commands attempted:

```bash
python -m pip install --no-deps --only-binary=:all: scikit-learn
python -m pip install --no-deps --only-binary=:all: --retries 5 --timeout 120 scikit-learn
python -m pip install --no-deps --only-binary=:all: --retries 5 --timeout 120 -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn
```

All failed due incomplete network reads. No torch, torchvision, CUDA, IsaacLab, numpy, or scipy packages were upgraded/downgraded.

## 23. Next Steps

1. Download the real repackaged PALNet/SSCNet-style NYU `.npz` data from the README Google Drive or BaiduYun link. Do not download NYU raw 428GB.
2. Replace the synthetic dry-run data directories with real `NYUtrain_npz` and `NYUtest_npz`, or update `config.py` to real paths.
3. Install `scikit-learn` into `env_isaaclab` when network is stable, then remove or ignore the fallback if desired.
4. Decide how to handle pretrained mismatch:
   - obtain a checkpoint matching current lightweight `SSCNet.py`, or
   - restore the original PALNet architecture matching `PALNet.pth.tar`, or
   - train current lightweight `SSCNet` from scratch.
5. Run a real-data dry-run with 1 epoch / small batch.
6. Only after real-data dry-run passes, start full 50 epoch training.

## 24. Real NPZ Data Download

Current compatibility changes were preserved before data work:

```text
/home/ubuntu22/sc_explorer_ws/logs/ssc_network_compatibility_changes.patch
```

README repackaged data links:

```text
Google Drive:
https://drive.google.com/drive/folders/15vFzZQL2eLu6AKSAcCbIyaA9n1cQi3PO?usp=sharing

BaiduYun:
https://pan.baidu.com/s/1mtdAEdHYTwS4j8QjptISBg
Access code: lpmk
```

The README describes these as repackaged PALNet/SSCNet-style `.npz` samples with:

```text
rgb
depth
tsdf_hr
tsdf_lr
target_hr
target_lr
position
```

Google Drive folder contents observed with `gdown --folder`:

```text
NYUCADtest_npz.zip
NYUCADtrain_npz.zip
NYUtest_npz.zip
NYUtrain_npz.zip
```

For this stage, only `NYUtrain_npz.zip` and `NYUtest_npz.zip` should be downloaded. Do not download NYU raw 428GB, official NYU labeled `.mat`, or NYUCAD unless explicitly needed later.

Disk check before download:

```text
/dev/nvme0n1p7: 484G total, 168G used, 297G available
/home/ubuntu22/sc_explorer_ws/data: 124K
```

`gdown` is installed in `env_isaaclab`:

```text
gdown 6.0.0 at /home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages
```

Resume update on 2026-05-29 after stream disconnect:

```text
Current working directory: /home/ubuntu22/sc_explorer_ws
Current working directory git status: not a git repository
Official SC-Explorer repository: /home/ubuntu22/sc_explorer_ws/ssc_exploration
Official repository HEAD: 3b1492e47a2e62363328cc931d9ef2342fb55891
Official repository status: local compatibility changes still present in ssc_network/config.py,
models/SSCNet.py, models/__init__.py, train.py, and utils/ssc_metrics.py.
```

Current changes were saved again:

```text
/home/ubuntu22/sc_explorer_ws/logs/ssc_network_current_changes.patch
/home/ubuntu22/sc_explorer_ws/logs/ssc_network_git_status.txt
```

Automatic Google Drive download status:

```text
Target directory:
/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz

Completed real .npz files found:
0

Completed NYUtrain_npz directory:
missing

Completed NYUtest_npz directory:
missing

Partial files/directories found:
/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz.zip.parts
/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz.zipkmakv_gk.part
/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/SSC_Dataset
```

`NYUtest_npz.zip` was attempted with `gdown` and then with a local range-download helper script. The Google Drive folder listing was reachable and confirmed these file IDs:

```text
NYUtest_npz.zip:
1qXI-GPWAGnnsDUHuSzxeTudK8RySqat0

NYUtrain_npz.zip:
1N1LgTA2ANXLXkeH2slsC8U7HROKfy-3P

NYUCADtest_npz.zip:
1ygk9_Aw_8YeKLelC058G0JGQUg-pwyXi

NYUCADtrain_npz.zip:
1WG-5UP5StIPSgxz8J4vfRRg5XGirP9F7
```

The automatic download did not complete. The latest log ends with:

```text
RuntimeError: part 46 failed after 20 attempts: curl: (35) error:0A000126:SSL routines::unexpected eof while reading
```

Log:

```text
/home/ubuntu22/sc_explorer_ws/logs/download_NYUtest_npz.log
```

Decision for this resume: stop automatic download attempts. Manual download is needed unless the network/proxy situation changes.

Manual download instructions:

```text
Download from README Google Drive folder:
https://drive.google.com/drive/folders/15vFzZQL2eLu6AKSAcCbIyaA9n1cQi3PO?usp=sharing

or README BaiduYun:
https://pan.baidu.com/s/1mtdAEdHYTwS4j8QjptISBg
Access code: lpmk

Required files for this stage only:
NYUtrain_npz.zip
NYUtest_npz.zip

Place them under:
/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/

Unzip/organize to:
/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtrain_npz/*.npz
/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz/*.npz
```

Do not download NYU raw 428GB, official NYU labeled `.mat`, or NYUCAD for this stage.

## Resume After Stream Disconnect

The previous run was interrupted during real data download, not during code execution, field checks, dataloader smoke tests, model forward, or training.

Observed status on resume:

```text
Current working directory: /home/ubuntu22/sc_explorer_ws
Current working directory is not a git repository.
Official SC-Explorer git repository is valid at:
/home/ubuntu22/sc_explorer_ws/ssc_exploration

Synthetic dry-run training: completed previously.
Synthetic dry-run test.py: completed previously.
Real .npz data: not downloaded completely.
Real NPZ field check: not run.
Real dataloader smoke test: not run.
Real model forward smoke test: not run.
Real data dry-run training: not run.
```

Current blocker:

```text
The README repackaged NYU .npz data has not been fully downloaded.
Automatic command-line Google Drive download repeatedly fails with SSL EOF errors.
Manual download of NYUtrain_npz.zip and NYUtest_npz.zip is needed.
```

Next continuation point:

```text
After manual download and unzip, continue from real NPZ field check.
```

## 25. Real NPZ Field Check

Not run on resume because no completed real `.npz` files were present under:

```text
/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtrain_npz
/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz
```

Only synthetic dry-run `.npz` files exist under:

```text
/home/ubuntu22/sc_explorer_ws/data/NYUtrain_npz/sample_train_0000.npz
/home/ubuntu22/sc_explorer_ws/data/NYUtest_npz/sample_test_0000.npz
```

## 26. Real Dataloader Smoke Test

Not run. Blocked by missing completed real `.npz` data.

## 27. Real Model Forward Smoke Test

Not run. Blocked by missing completed real `.npz` data.

## 28. Real Data Dry-run Training

Not run. Blocked by missing completed real `.npz` data and therefore no field-check/dataloader/forward prerequisites.

## 29. Real NPZ Field Check After Manual Download

Status date: 2026-05-29.

The user manually downloaded and extracted the real repackaged NYU `.npz` data. No data download commands were run in this continuation.

Real data directories:

```text
/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtrain_npz
/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz
```

File counts:

```text
train: 794 .npz files
test: 654 .npz files
```

Field check command was run in `env_isaaclab` and logged to:

```text
/home/ubuntu22/sc_explorer_ws/logs/real_npz_field_check.log
```

For the first 3 train and first 3 test `.npz` files:

```text
Required fields present: rgb, depth, tsdf_hr, tsdf_lr, target_lr, position
Extra field present: target_hr

rgb:       (3, 480, 640), float32, normalized image values around [-2.1179, 2.64]
depth:     (1, 480, 640), float64
tsdf_hr:   (240, 144, 240), float32, min about -0.9167, max 1.0
tsdf_lr:   (60, 36, 60), float32, min about -0.001, max 1.0
target_lr: (60, 36, 60), uint8, min 0, max 255
position:  (480, 640), int32
```

`target_lr` class check:

```text
Non-ignore labels in sampled files are within 0..11.
255 ignore_index is present in every sampled file.
FIELD_CHECK_OK=True
```

## 30. Real Data Config Update

Updated:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/config.py
```

`Path.db_root_dir('nyu')` now points to:

```text
train: /home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtrain_npz
val:   /home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz
```

## 31. Real Dataloader Smoke Test

Command was run in `env_isaaclab` and logged to:

```text
/home/ubuntu22/sc_explorer_ws/logs/real_dataloader_smoke.log
```

Result:

```text
train_batches=794
val_batches=654
train tuple length: 6
val tuple length: 7
DATALOADER_SMOKE_OK=True
```

Observed batch tensor shapes:

```text
train.rgb:         (1, 3, 480, 640), float32
train.depth:       (1, 1, 480, 640), float64
train.tsdf_hr:     (1, 240, 144, 240), float32
train.target_lr_T: (1, 60, 36, 60), uint8
train.position:    (1, 480, 640), int32

val.rgb:           (1, 3, 480, 640), float32
val.depth:         (1, 1, 480, 640), float64
val.tsdf_hr:       (1, 240, 144, 240), float32
val.target_lr_T:   (1, 60, 36, 60), uint8
val.nonempty_T:    (1, 60, 36, 60), float32
val.position:      (1, 480, 640), int32
```

## 32. Real Model Forward Smoke Test

First attempt inside the command sandbox could not access CUDA:

```text
torch.cuda.is_available() = False
RuntimeError: No CUDA GPUs are available
```

The same command was rerun outside the sandbox, still in `env_isaaclab`, and logged to:

```text
/home/ubuntu22/sc_explorer_ws/logs/real_model_forward_smoke.log
```

Successful result:

```text
torch=2.7.0+cu128
cuda_available=True
cuda_device_count=1
output_shape=(1, 12, 60, 36, 60)
output_dtype=torch.float32
output_finite=True
cuda_max_memory_allocated_mb=751.5
MODEL_FORWARD_SMOKE_OK=True
```

## 33. Real Data 1 Epoch Dry-run Training

No RL, Unreal, AirSim, or 50 epoch training was run.

Command was run in `env_isaaclab` with CUDA access:

```text
python ./train.py --model=sscnet --dataset=nyu --epochs=1 --batch_size=1 --workers=0 --lr=0.01 --lr_adj_n=10 --lr_adj_rate=0.1 --checkpoint=/home/ubuntu22/sc_explorer_ws/checkpoints/real_dryrun/ --model_name=SSCNet_real_dryrun
```

Log:

```text
/home/ubuntu22/sc_explorer_ws/logs/ssc_network_real_dryrun.log
```

Result:

```text
Training data: /home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtrain_npz
Validate data: /home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz
Training epochs: 1
Batch size: 1
Number of workers: 0
Training finished in: 0:03:01.581893
```

Validation after the dry-run epoch:

```text
Validate: epoch 1, p 54.2, r 85.0, IoU 48.6
pixel-acc 41.5239, mean IoU 11.2
SSC IoU: [33.2 0. 92.3 14.8 0. 0. 0. 0. 0. 0. 16.6 0.]
```

Generated checkpoints:

```text
/home/ubuntu22/sc_explorer_ws/checkpoints/real_dryrun/cp_SSCNet_real_dryrun.pth.tar
/home/ubuntu22/sc_explorer_ws/checkpoints/real_dryrun/cpBest_SSCNet_real_dryrun.pth.tar
```

## 34. Real `test.py` Dry-run

Because checkpoint generation succeeded, `test.py` was run in `env_isaaclab` with CUDA access:

```text
python ./test.py --model=sscnet --dataset=nyu --batch_size=1 --workers=0 --resume=/home/ubuntu22/sc_explorer_ws/checkpoints/real_dryrun/cp_SSCNet_real_dryrun.pth.tar
```

Log:

```text
/home/ubuntu22/sc_explorer_ws/logs/ssc_network_real_test_dryrun.log
```

Result:

```text
Validate with TSDF: p 54.2, r 85.0, IoU 48.6
pixel-acc 41.5239, mean IoU 11.2
SSC IoU: [33.22678 0. 92.3152 14.784882 0. 0. 0. 0. 0. 0. 16.620502 0.]
Occupancy calibration binary: 0.016 free, 0.226 occ.
Testing finished in: 0:01:41.666661
```

The repeated log line `Number of classes > 12: 0.` comes from the existing metric code and was not a failure.

## 31. Sanity 5 Epoch Status Check

Checked on 2026-05-29.

Training log:

```text
/home/ubuntu22/sc_explorer_ws/logs/sanity_5epoch/SSCNet_sanity5_train.log
```

Last log state:

```text
Epoch 4 validation completed and saved a new best checkpoint.
Epoch 5/5 started, but the log ends during training at step 182.
No epoch 5 validation line is present.
No "Training finished" line is present.
No Traceback, Error, Killed, or interrupted line was found.
```

Validation metrics found in the training log:

```text
epoch 1: p 54.8, r 82.6, IoU 48.2, pixel-acc 42.2025, mean IoU 10.9
epoch 2: p 53.0, r 95.0, IoU 51.4, pixel-acc 40.0017, mean IoU 14.7
epoch 3: p 55.2, r 92.7, IoU 52.7, pixel-acc 45.4076, mean IoU 14.8
epoch 4: p 58.3, r 90.7, IoU 54.6, pixel-acc 50.2463, mean IoU 17.1
```

Best checkpoint was saved at epoch 4:

```text
Yeah! Got better mIoU 17.13340932672674% in epoch 4. State saved
```

Checkpoints:

```text
latest: /home/ubuntu22/sc_explorer_ws/checkpoints/sanity_5epoch/cp_SSCNet_sanity5.pth.tar
  mtime: 2026-05-29 16:39:00.146388552 +0800
  size: 445630 bytes

best: /home/ubuntu22/sc_explorer_ws/checkpoints/sanity_5epoch/cpBest_SSCNet_sanity5.pth.tar
  mtime: 2026-05-29 16:39:00.148388354 +0800
  size: 445994 bytes
```

The best checkpoint is usable: `test.py` loaded it successfully in `env_isaaclab` with CUDA access and completed evaluation.

Best checkpoint test log:

```text
/home/ubuntu22/sc_explorer_ws/logs/sanity_5epoch/SSCNet_sanity5_best_test.log
```

Best checkpoint test.py result:

```text
Validate with TSDF: p 58.3, r 90.7, IoU 54.6
pixel-acc 50.2463, mean IoU 17.1
SSC IoU: [37.564804 0. 91.66317 25.872135 0. 0. 35.8701 0. 6.657828 0. 22.187113 6.2171583]
Occupancy calibration binary: 0.011 free, 0.363 occ.
Testing finished in: 0:01:42.088655
```

Resume status:

```text
Resume is not required to use the best checkpoint, because epoch 4 best checkpoint is present and test.py verified it.
Resume would only be needed if the exact 5/5 sanity run completion record is required.
```

Full training recommendation:

```text
Do not start full training solely from this interrupted sanity log. Use the verified best checkpoint for the next pipeline step first, or explicitly resume only if a completed epoch 5 record is required.
```

Local context path issue fixed on 2026-05-29: `.project_context` is now a directory, and the preserved context file is `.project_context/README.md`.

## 32. Full 50 Epoch Training Result

Checked on 2026-05-29 after the 50 epoch full training run. No further training, RL, Unreal, or AirSim commands were run for this check.

Training log:

```text
/home/ubuntu22/sc_explorer_ws/logs/full_train/SSCNet_NYU_full_train.log
```

Training status:

```text
Completed normally: yes
Training finished line: Training finished in: 2:15:09.683078
Traceback/Error/Killed found: no
Last 200 log lines checked: yes, they end at epoch 50 validation and Training finished.
```

Checkpoints:

```text
latest: /home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cp_SSCNet_NYU_full_train.pth.tar
  size: 446171 bytes
  mtime: 2026-05-29 19:15:36.881783007 +0800

best: /home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar
  size: 446407 bytes
  mtime: 2026-05-29 18:10:42.343013179 +0800
```

The training script saves `cpBest` by semantic mean IoU. The best checkpoint was saved at epoch 26:

```text
Validate:epoch 26, p 56.4, r 93.5, IoU 54.1
pixel-acc 50.1922, mean IoU 22.5
Yeah! Got better mIoU 22.535241733897816% in epoch 26. State saved
```

Final epoch validation:

```text
Validate:epoch 50, p 56.1, r 93.8, IoU 53.9
pixel-acc 49.7988, mean IoU 22.4
```

Validation metrics by epoch:

```text
epoch 01: p 54.4, r 84.8, IoU 48.7, pixel-acc 41.7348, mean IoU 11.1
epoch 02: p 53.2, r 94.5, IoU 51.4, pixel-acc 40.1687, mean IoU 13.6
epoch 03: p 54.5, r 93.1, IoU 52.2, pixel-acc 44.2866, mean IoU 14.9
epoch 04: p 60.2, r 89.2, IoU 55.6, pixel-acc 51.9506, mean IoU 17.0
epoch 05: p 50.3, r 97.7, IoU 49.7, pixel-acc 37.3378, mean IoU 18.0
epoch 06: p 56.8, r 92.5, IoU 54.1, pixel-acc 48.2753, mean IoU 18.7
epoch 07: p 58.8, r 91.0, IoU 55.2, pixel-acc 51.5571, mean IoU 19.3
epoch 08: p 54.4, r 95.0, IoU 52.7, pixel-acc 45.6141, mean IoU 18.4
epoch 09: p 51.8, r 96.9, IoU 51.0, pixel-acc 39.9179, mean IoU 15.4
epoch 10: p 53.5, r 95.6, IoU 52.2, pixel-acc 44.3682, mean IoU 20.3
epoch 11: p 55.0, r 94.5, IoU 53.2, pixel-acc 47.5872, mean IoU 21.6
epoch 12: p 56.1, r 93.7, IoU 53.9, pixel-acc 49.1514, mean IoU 22.0
epoch 13: p 56.1, r 93.7, IoU 53.9, pixel-acc 49.2573, mean IoU 22.0
epoch 14: p 55.8, r 94.0, IoU 53.7, pixel-acc 48.9812, mean IoU 21.8
epoch 15: p 56.0, r 93.9, IoU 53.8, pixel-acc 49.2448, mean IoU 21.9
epoch 16: p 54.4, r 95.1, IoU 52.8, pixel-acc 46.8967, mean IoU 21.8
epoch 17: p 55.6, r 94.3, IoU 53.6, pixel-acc 48.7619, mean IoU 21.9
epoch 18: p 55.7, r 94.1, IoU 53.7, pixel-acc 48.7812, mean IoU 21.3
epoch 19: p 56.5, r 93.5, IoU 54.1, pixel-acc 49.8919, mean IoU 21.5
epoch 20: p 54.2, r 95.3, IoU 52.7, pixel-acc 46.7788, mean IoU 21.5
epoch 21: p 55.9, r 93.9, IoU 53.8, pixel-acc 49.5015, mean IoU 22.3
epoch 22: p 56.4, r 93.5, IoU 54.1, pixel-acc 50.1022, mean IoU 22.4
epoch 23: p 56.3, r 93.6, IoU 54.0, pixel-acc 49.9643, mean IoU 22.3
epoch 24: p 55.7, r 94.1, IoU 53.7, pixel-acc 49.2254, mean IoU 22.2
epoch 25: p 56.0, r 93.9, IoU 53.8, pixel-acc 49.6003, mean IoU 22.3
epoch 26: p 56.4, r 93.5, IoU 54.1, pixel-acc 50.1922, mean IoU 22.5
epoch 27: p 56.1, r 93.7, IoU 53.9, pixel-acc 49.8161, mean IoU 22.3
epoch 28: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.6920, mean IoU 22.3
epoch 29: p 56.3, r 93.7, IoU 54.0, pixel-acc 50.0134, mean IoU 22.4
epoch 30: p 56.2, r 93.7, IoU 53.9, pixel-acc 49.8561, mean IoU 22.4
epoch 31: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.8309, mean IoU 22.4
epoch 32: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7944, mean IoU 22.4
epoch 33: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7828, mean IoU 22.4
epoch 34: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.8003, mean IoU 22.4
epoch 35: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7931, mean IoU 22.4
epoch 36: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7608, mean IoU 22.4
epoch 37: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7620, mean IoU 22.4
epoch 38: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7864, mean IoU 22.4
epoch 39: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.8045, mean IoU 22.4
epoch 40: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7989, mean IoU 22.4
epoch 41: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7975, mean IoU 22.4
epoch 42: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7982, mean IoU 22.4
epoch 43: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7985, mean IoU 22.4
epoch 44: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7986, mean IoU 22.4
epoch 45: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7984, mean IoU 22.4
epoch 46: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7968, mean IoU 22.4
epoch 47: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7994, mean IoU 22.4
epoch 48: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7977, mean IoU 22.4
epoch 49: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7976, mean IoU 22.4
epoch 50: p 56.1, r 93.8, IoU 53.9, pixel-acc 49.7988, mean IoU 22.4
```

Best checkpoint test command was run in `env_isaaclab`:

```text
/home/ubuntu22/sc_explorer_ws/logs/full_train/best_checkpoint_test.log
```

Best checkpoint test.py result:

```text
Validate with TSDF: p 56.4, r 93.5, IoU 54.1
pixel-acc 50.1922, mean IoU 22.5
SSC IoU: [33.339954 17.513067 92.019966 27.77389 0. 0.6074526 39.513786 30.215069 7.8830385 0. 23.526686 8.834691]
Occupancy calibration binary: 0.009 free, 0.350 occ.
Testing finished in: 0:01:42.312438
```

The best checkpoint test result matches the epoch 26 training validation metrics, with only formatting-level differences.

Epoch 40-50 trend:

```text
The run is flat from epoch 40 through 50: p 56.1, r 93.8, occupancy IoU 53.9, mean IoU 22.4, and pixel accuracy approximately 49.798.
There is no clear upward trend in mIoU or occupancy IoU after epoch 40.
```

Recommendation:

```text
Do not continue training now. Use the best checkpoint for planner / map_predict integration.
Occupancy IoU is usable for the current map_predict task, even though semantic mean IoU is modest, because the downstream path mainly needs occupancy/free/confidence.
Next step: wire the best checkpoint into planner / map_predict and run integration smoke tests only.
```

Checked `.project_context`: it is currently a directory, so the unresolved file-vs-directory path issue does not remain.

## 31. Stage 2A Inference Wrapper Design

Stage 2A scope checked on 2026-05-29:

```text
Goal: best checkpoint offline map_predict inference wrapper plus PredictionLayer.
Not in scope: planner, RL, PPO, imitation learning, Unreal, AirSim, retraining, observed_map writes.
Best checkpoint: /home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar
```

Existing SSCNet inference pattern:

```text
test.py builds the model with make_model(args.model, num_classes=12).cuda().
test.py loads checkpoint via torch.load(args.resume), then net.load_state_dict(cp_states['state_dict'], strict=True).
train.py saves checkpoints as {'state_dict': net.state_dict()} and selects cpBest by validation mean IoU.
```

Dataloader fields and shapes:

```text
NYUDataset npz fields: rgb, depth, tsdf_hr, tsdf_lr, target_lr, position.
Validation DataLoader batch shapes:
  rgb:      (1, 3, 480, 640), float32
  depth:    (1, 1, 480, 640), float64 from npz, cast to float for model
  tsdf_hr:  (1, 240, 144, 240), float32
  target:   (1, 60, 36, 60), uint8, from target_lr.T
  nonempty: (1, 60, 36, 60), float32, derived from tsdf_lr and target_lr
  position: (1, 480, 640), int32, cast to long for model
```

Model input/output:

```text
SSCNet.forward(x_depth, x_tsdf=None, p=None, x_rgb=None)
Current SSCNet uses x_depth and p/position.
x_rgb can be passed to match train.py/test.py, but the current forward ignores it.
x_tsdf / tsdf_hr are not used by current SSCNet.
Output logits shape: (B, 12, 60, 36, 60).
```

Free/occupied class convention:

```text
config.py colorMap labels class 0 as "empty, free space".
ssc_metrics.py treats predict > 0 and target > 0 as occupied for completion IoU.
offline_infer_npz.py therefore records free_class_id = 0 and uses occupied_prob = 1 - prob[class 0].
```

## 32. Offline Inference Script

Created:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/offline_infer_npz.py
```

Behavior:

```text
Loads a single NYU .npz sample.
Builds official make_model("sscnet", num_classes=12).
Loads cpBest checkpoint with strict=True.
Runs model.eval() under torch.no_grad().
Computes class_prob = softmax(logits, dim=1), pred_class, confidence, free_prob, occupied_prob.
Writes a standalone prediction .npz and does not write to observed_map.
```

Single-sample command:

```text
python ./offline_infer_npz.py \
  --checkpoint /home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar \
  --input_npz /home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz/NYU0670_0000_voxels.npz \
  --output_dir /home/ubuntu22/sc_explorer_ws/outputs/sscnet_inference \
  --save_probs \
  --print_stats
```

Single-sample result:

```text
input: /home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz/NYU0670_0000_voxels.npz
output: /home/ubuntu22/sc_explorer_ws/outputs/sscnet_inference/NYU0670_0000_voxels_prediction.npz
device: cuda
logits shape: (1, 12, 60, 36, 60)
pred_class shape: (60, 36, 60)
pred_class unique counts: 0:107822, 2:3257, 3:4290, 5:3, 6:88, 7:196, 8:79, 10:13001, 11:864
confidence min/max/mean: 0.167309/0.993769/0.793621
occupied_prob min/max/mean: 0.006231/0.998514/0.276600
free_prob min/max/mean: 0.001486/0.993769/0.723400
inference time: 0.113546s
GPU memory peak: 787975680 bytes
```

Saved arrays:

```text
pred_class: uint8, (60, 36, 60)
confidence: float32, (60, 36, 60)
occupied_prob: float32, (60, 36, 60)
free_prob: float32, (60, 36, 60)
class_prob: float16, (12, 60, 36, 60), saved because --save_probs was used
input_npz/checkpoint/free_class_id/free_class_assumption metadata
```

Log:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage2a_offline_infer_npz.log
```

## 33. PredictionLayer Wrapper

Created:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/prediction_layer.py
```

Implemented API:

```text
PredictionLayer.from_npz(path)
shape()
get_pred_class(index)
get_confidence(index)
get_occupied_prob(index)
get_free_prob(index)
is_predicted(index, tau=0.1)
is_predicted_occupied(index, tau=0.1)
is_predicted_free(index, tau=0.1)
get_prediction_gain(index, tau=0.1)
```

Semantics:

```text
index is a 3D voxel index tuple matching the prediction array order.
is_predicted: confidence[index] >= tau.
is_predicted_occupied: occupied_prob[index] >= 0.5 and confidence[index] >= tau.
is_predicted_free: free_prob[index] >= 0.5 and confidence[index] >= tau.
get_prediction_gain: returns confidence when confidence >= tau, else 0.0.
```

This class is read-only, has no ROS dependency, and does not modify observed_map.

## 34. Offline Prediction Smoke Test

Created:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_offline_prediction_layer.py
```

Smoke test command:

```text
python ./test_offline_prediction_layer.py
```

Smoke test result:

```text
input: /home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz/NYU0001_0000_voxels.npz
output: /home/ubuntu22/sc_explorer_ws/outputs/sscnet_inference_smoke/NYU0001_0000_voxels_prediction.npz
logits shape: (1, 12, 60, 36, 60)
pred_class shape: (60, 36, 60)
confidence min/max/mean: 0.163874/0.996051/0.793409
occupied_prob min/max/mean: 0.003949/0.999262/0.273300
free_prob min/max/mean: 0.000738/0.996051/0.726700
inference time: 0.115207s
confidence >= 0.1 voxels: 129600
occupied_prob >= 0.5 voxels: 26383
free_prob >= 0.5 voxels: 103217
PredictionLayer smoke test passed.
```

Basic assertions passed:

```text
pred_class/confidence/occupied_prob/free_prob shapes match.
confidence, occupied_prob, and free_prob are within [0, 1].
pred_class range is within 0..11.
At least one voxel has confidence >= 0.1.
```

Log:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage2a_prediction_layer_test.log
```

## 35. Batch5 Inference Test

Ran offline inference on the first 5 NYUtest .npz files only. This was not a full test-set batch run.

Output directory:

```text
/home/ubuntu22/sc_explorer_ws/outputs/sscnet_inference_batch5
```

Batch5 results:

```text
NYU0670_0000_voxels: inference 0.114093s, confidence mean 0.793621, occupied_prob mean 0.276600
NYU0706_0000_voxels: inference 0.113477s, confidence mean 0.811185, occupied_prob mean 0.237049
NYU0781_0000_voxels: inference 0.114511s, confidence mean 0.835477, occupied_prob mean 0.209923
NYU0787_0000_voxels: inference 0.115180s, confidence mean 0.775362, occupied_prob mean 0.310343
NYU0839_0000_voxels: inference 0.113260s, confidence mean 0.847539, occupied_prob mean 0.215823
```

Log:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage2a_batch5_inference.log
```

## 36. Next Step: Deterministic Expert Integration

Stage 2A is complete:

```text
The best SSCNet checkpoint can now be loaded strictly and used for standalone offline map_predict inference.
PredictionLayer can load the resulting prediction .npz and expose confidence/free/occupied queries.
Prediction output remains separate from observed_map.
```

Next recommended step:

```text
Stage 2B: deterministic SC-Explorer-style expert candidate scoring using PredictionLayer.
Still not RL, not PPO, not imitation learning, and not Unreal/AirSim.
```

## 37. Stage 2B Paper-Faithful Expert Candidate Scorer

Stage 2B strict scorer is complete. This version replaces the earlier
target-label mock observed-map prototype. The strict rule is:

```text
target_lr and target_hr are ground truth.
They must not be used for measured S, predicted P, candidate generation,
ray blocking, collision, path cost, gain, final score, or expert_action.
They may be used only for offline evaluation/debug after expert_action has
already been chosen.
```

Created files:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/sc_explorer_paper_expert.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/run_paper_expert_offline.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_paper_expert.py
```

The earlier prototype entry files were disabled to prevent accidental
ground-truth leakage:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/sc_explorer_expert.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/run_sc_explorer_expert_offline.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_sc_explorer_expert.py
```

Measured set S construction:

```text
Default measured_mode: tsdf_lr.
Allowed measured modes: tsdf_lr, position, union.

tsdf_lr mode:
  Uses only sensor-derived tsdf_lr.
  The local dataloader comments/construction indicate:
    0.001  = empty/free-like sensor-observed cell
    1.0    = surface cell
    -0.001 = occluded/unmeasured cell
  S is approximated as tsdf_lr >= 0.001.
  Measured occupied/surface blocking mask is tsdf_lr == 1.0.

position mode:
  Uses only the 2D-to-3D depth projection mapping position.
  High-res indices are converted from (240, 144, 240) to (60, 36, 60).

union mode:
  S = tsdf_lr-derived mask OR position-derived mask.
```

This is an offline NYU approximation of S. In real exploration, S must come
from the online sensor-integrated measured map.

Prediction set P:

```text
predicted_mask[v] = prediction_layer.confidence[v] >= tau
P[v] = predicted_mask[v] and not S[v]
```

target_lr is not read for P.

Candidate generation:

```text
Candidates are sampled from measured voxels in S.
Start voxel is the measured voxel nearest the median of S.
Candidate yaw is sampled deterministically from the configured RNG seed.
No target_lr labels are used to decide free/occupied/collision.
This is only a candidate scorer, not a full robot motion planner.
```

Ray-casting:

```text
Default raycast_mode: non_blocking.

non_blocking:
  Scene-completion prediction never blocks rays.
  Only sensor-derived measured_occupied_mask may block rays.

sc_blocking:
  Optional paper ablation mode.
  Predicted occupied voxels may block rays via
  PredictionLayer.is_predicted_occupied(v, tau).
```

Gain formulas:

```text
S = measured voxels from sensor-derived fields.
P = predicted-by-SC voxels not in S.

I_exp(v) = 0 if v in S else 1
I_sc(v) = 1 if v in P else 0
I_hybrid(v) = I_exp(v) + I_sc(v)
I_occ(v) = 1 if v in P and predicted occupied else 0
I_conf(v) = abs(0.5 - occupied_prob[v]) if v in P else 0

gain_* is the sum over visible voxels.
```

Cost and utility:

```text
voxel_size = 0.08 m
v_max = 1.0 m/s
yaw_rate = 90 deg/s

distance_m = voxel_distance * voxel_size
time_pos = distance_m / v_max
time_yaw = abs(delta_yaw) / yaw_rate
path_cost = time_pos + time_yaw

utility_exp = gain_exp / max(path_cost, eps)
utility_sc = gain_sc / max(path_cost, eps)
utility_hybrid = gain_hybrid / max(path_cost, eps)
utility_occ = gain_occ / max(path_cost, eps)
utility_conf = gain_conf / max(path_cost, eps)

Default gain_mode = hybrid
final_score = utility_hybrid
```

This is a per-candidate paper-faithful gain/cost scorer. Full SC-Explorer
RRT tree utility Eq. 12 and replanning are not implemented in Stage 2B.

Single sample command:

```text
python ./run_paper_expert_offline.py \
  --sample_npz /home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz/NYU0670_0000_voxels.npz \
  --prediction_npz /home/ubuntu22/sc_explorer_ws/outputs/sscnet_inference/NYU0670_0000_voxels_prediction.npz \
  --output_dir /home/ubuntu22/sc_explorer_ws/outputs/paper_expert \
  --num_candidates 64 \
  --top_n 16 \
  --tau 0.1 \
  --measured_mode tsdf_lr \
  --raycast_mode non_blocking \
  --gain_mode hybrid \
  --print_topn
```

Single sample result:

```text
sample: NYU0670_0000_voxels
measured_mode: tsdf_lr
raycast_mode: non_blocking
gain_mode: hybrid
S measured voxels: 92449
P predicted-unmeasured voxels: 37151
candidates: 64
expert_action: 0
best score: 872.624436
best gain_exp: 473.0
best gain_sc: 473.0
best gain_hybrid: 946.0
best gain_occ: 352.0
best gain_conf: 137.822484
best path_cost: 1.084086
```

Outputs:

```text
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert/paper_expert_decision_NYU0670_0000_voxels.npz
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert/paper_expert_decisions.jsonl
```

Saved npz fields:

```text
candidate_features
feature_names
candidate_positions
candidate_yaws
valid_mask
expert_action
expert_scores
top_candidate_ids
gain_mode
measured_mode
raycast_mode
sample_npz
prediction_npz
measured_note
strict_no_target_note
tree_limitation_note
all_candidate_count
```

Smoke test:

```text
python ./test_paper_expert.py
```

Result:

```text
PredictionLayer loaded.
Sample npz loaded.
Measured mask built from sensor fields without target_lr.
Formula tests passed for I_exp, I_sc, I_hybrid, I_occ, and I_conf.
P = predicted and not measured was verified.
non_blocking raycast does not use predicted occupied as blocking.
sc_blocking raycast can use predicted occupied as blocking.
Utilities are finite.
expert_action is valid.
Output npz/jsonl fields exist.
Paper expert smoke test passed.
```

Log:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage2b_paper_expert_test.log
```

Batch5 smoke test:

```text
Output dir: /home/ubuntu22/sc_explorer_ws/outputs/paper_expert_batch5
Log: /home/ubuntu22/sc_explorer_ws/logs/stage2b_paper_expert_batch5.log
```

Batch5 results:

```text
NYU0670_0000_voxels: score 872.624436, gain_exp 473.0, gain_sc 473.0, gain_hybrid 946.0, gain_occ 352.0, gain_conf 137.822484, path_cost 1.084086, expert_action 0
NYU0706_0000_voxels: score 808.842120, gain_exp 274.0, gain_sc 274.0, gain_hybrid 548.0, gain_occ 45.0, gain_conf 63.967435, path_cost 0.677512, expert_action 0
NYU0781_0000_voxels: score 631.484156, gain_exp 562.0, gain_sc 562.0, gain_hybrid 1124.0, gain_occ 384.0, gain_conf 172.891850, path_cost 1.779934, expert_action 0
NYU0787_0000_voxels: score 392.273564, gain_exp 534.0, gain_sc 534.0, gain_hybrid 1068.0, gain_occ 243.0, gain_conf 184.878110, path_cost 2.722590, expert_action 0
NYU0839_0000_voxels: score 421.431736, gain_exp 392.0, gain_sc 392.0, gain_hybrid 784.0, gain_occ 220.0, gain_conf 106.438742, path_cost 1.860325, expert_action 0
```

Current limitations:

```text
Offline NYU measured S is an approximation from tsdf_lr/position, not a real online measured map.
Candidate generation samples measured voxels; no real collision-free motion planning is implemented.
Cost is a simple position/yaw time approximation, not measured execution time.
Full SC-Explorer RRT tree utility Eq. 12 is not implemented yet.
No RL, PPO, imitation learning training, Unreal, AirSim, retraining, or robot execution was performed.
```

## 38. Stage 2C Paper Expert Dataset Generation

Stage 2C is complete. This stage converts strict Stage 2B paper-faithful
expert outputs into an imitation-learning-ready dataset format. It does not
train imitation learning and does not create a policy network.

Created files:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/generate_paper_expert_dataset.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/test_paper_expert_dataset.py
```

Dataset output root used for smoke:

```text
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/samples
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/predictions
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/logs
```

Dataset sample format:

```text
candidate_features: float32 [N, D]
feature_names: string [D]
candidate_positions: int32 [N, 3]
candidate_yaws: float32 [N]
valid_mask: bool [N]
expert_action: int64 scalar
expert_scores: float32 [N]
top_candidate_ids: int64 [N]
gain_mode: string
measured_mode: string
raycast_mode: string
sample_npz: string
prediction_npz: string
sample_id: string
strict_no_target_note: string
tree_limitation_note: string
```

The Stage 2C smoke dataset preserves the Stage 2B top-candidate format:

```text
N = top_n = 16
D = 15
num_candidates scored before top-n selection = 64
```

Feature names:

```text
gain_exp
gain_sc
gain_hybrid
gain_occ
gain_conf
path_cost
utility_exp
utility_sc
utility_hybrid
utility_occ
utility_conf
final_score
visible_count
measured_visible_count
predicted_unmeasured_visible_count
```

No target_lr-derived or target_hr-derived features are saved.

Manifest:

```text
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/manifest.jsonl
```

Each ok line includes:

```text
sample_id
sample_npz
prediction_npz
expert_npz
num_candidates
top_n
expert_action
best_score
best_gain_exp
best_gain_sc
best_gain_hybrid
best_gain_occ
best_gain_conf
best_path_cost
gain_mode
measured_mode
raycast_mode
status
```

Failed samples are written as manifest lines with status `failed` and an
error string. Samples are not silently skipped.

Metadata:

```text
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/metadata.json
```

Metadata includes:

```text
created_at
code_version / git_commit
checkpoint
prediction_tool_used
expert_scorer_file
feature_names
num_candidates
top_n
tau
measured_mode
raycast_mode
gain_mode
voxel_size
v_max
yaw_rate_deg
max_range
num_yaw
num_pitch
strict_no_target_lr: true
target_lr_usage: evaluation_only_not_used_for_scoring
tree_utility_eq12_implemented: false
note: per-candidate paper-faithful gain/cost expert, not full RRT tree planner
```

Smoke command:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:$PYTHONPATH
cd /home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network

python ./generate_paper_expert_dataset.py \
  --sample_dir /home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz \
  --prediction_dir /home/ubuntu22/sc_explorer_ws/outputs/sscnet_inference_batch5 \
  --output_dir /home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke \
  --max_samples 5 \
  --num_candidates 64 \
  --top_n 16 \
  --tau 0.1 \
  --measured_mode tsdf_lr \
  --raycast_mode non_blocking \
  --gain_mode hybrid \
  2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/stage2c_dataset_smoke.log
```

Smoke result:

```text
total samples: 5
ok: 5
failed: 0
manifest rows: 5
sample npz count: 5
combined smoke npz:
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/combined_smoke.npz
```

The smoke generator reused the existing Stage 2A batch5 predictions:

```text
/home/ubuntu22/sc_explorer_ws/outputs/sscnet_inference_batch5
```

The generator prioritizes samples that have matching predictions when
`--generate_missing_predictions` is not set. This lets the Stage 2C smoke run
reuse the existing batch5 prediction set while still recording selected
failures in the manifest instead of silently skipping them.

Test command:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:$PYTHONPATH
cd /home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network

python ./test_paper_expert_dataset.py \
  --dataset_dir /home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke \
  2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/stage2c_dataset_test.log
```

Test result:

```text
Stage 2C paper expert dataset validation passed.
manifest records: 5
ok samples: 5
failed samples: 0
first shape: N=16 D=15
forbidden target fields check: passed
```

Logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage2c_dataset_smoke.log
/home/ubuntu22/sc_explorer_ws/logs/stage2c_dataset_test.log
```

Current limitations:

```text
Offline NYU measured S is still an approximation from tsdf_lr/position.
There is still no online sensor-integrated measured map in this dataset.
Candidate generation still samples measured voxels and is not real robot motion planning.
Full SC-Explorer RRT tree utility Eq. 12 is not implemented.
No imitation-learning training was performed.
No RL, PPO, Unreal, AirSim, SSCNet retraining, or observed_map write was performed.
```

Next recommended step:

```text
Stage 3A: create an imitation-learning Dataset/DataLoader and behavior cloning
training script, but first run only a data-loading smoke test.
```

## 39. Stage 3A IL Dataset/DataLoader Smoke Test

Stage 3A is complete. This stage adds a small imitation-learning module that
can read Stage 2C paper expert dataset samples, batch them with PyTorch, compute
feature normalization statistics, and run a forward-only behavior-cloning
skeleton smoke check.

No imitation-learning training was performed. No optimizer step was performed.
No model was saved.

Created files:

```text
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/__init__.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/paper_expert_dataset.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/policy.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/train_bc.py
/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/il/test_dataset.py
```

Main Dataset/DataLoader APIs:

```text
PaperExpertDataset
collate_paper_expert_batch
compute_feature_stats
save_feature_stats
CandidateMLPPolicy
```

Dataset behavior:

```text
Reads only Stage 2C expert sample `.npz` files from manifest ok rows.
Does not open original `sample_npz`.
Does not open SSCNet `prediction_npz`.
Checks metadata strict_no_target_lr == true by default.
Checks each expert sample has no target_lr, target_hr, gt, or ground_truth fields.
Checks expert_action is in range and valid_mask[expert_action] is true.
Checks candidate_features and expert_scores are finite.
Optionally applies externally supplied feature mean/std normalization.
```

Collate behavior:

```text
Fixed-size Stage 2C smoke samples stack to [B, N, D].
Variable candidate counts are padded defensively to max_N.
Padded candidate_features are 0.
Padded valid_mask is false.
Padded expert_scores are -inf.
expert_action remains each sample's original index.
```

Policy skeleton:

```text
CandidateMLPPolicy
Input:
  candidate_features [B, N, D]
  valid_mask [B, N]
Output:
  logits [B, N]
Invalid candidates are masked to -1e9.
```

Dataset smoke command:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:$PYTHONPATH
cd /home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network

python -m il.test_dataset \
  --dataset_dir /home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke \
  2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/stage3a_il_dataset_test.log
```

Dataset smoke result:

```text
Stage 3A IL dataset smoke test passed.
dataset size: 5
first candidate_features shape: (16, 15)
batch candidate_features shape: (2, 16, 15)
feature_names length: 15
expert_action valid: True
feature_stats mean shape: (15,)
feature_stats std min: 1.053888
logits shape: (2, 16)
cross_entropy loss: 0.000060
optimizer step performed: no
forbidden target fields: none
```

BC dry-run command:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:$PYTHONPATH
cd /home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network

python -m il.train_bc \
  --dataset_dir /home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke \
  --batch_size 2 \
  --num_workers 0 \
  --dry_run \
  --feature_stats_out /home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/feature_stats.npz \
  2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/stage3a_bc_dry_run.log
```

BC dry-run result:

```text
Stage 3A behavior cloning skeleton
dataset size: 5
B,N,D: 2,16,15
expert_action: [0, 0]
logits shape: (2, 16)
loss: 0.165347
optimizer step performed: no
model saved: no
```

Feature stats output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke/feature_stats.npz
mean shape: (15,)
std shape: (15,)
feature_names shape: (15,)
std min: 1.053888
```

Logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage3a_il_dataset_test.log
/home/ubuntu22/sc_explorer_ws/logs/stage3a_bc_dry_run.log
```

Current limitations:

```text
The dataset still comes from the Stage 2C offline NYU approximation.
No online sensor-integrated measured map is included.
No full RRT tree Eq. 12 is included.
No behavior cloning training has been run.
No RL, PPO, Unreal, AirSim, SSCNet retraining, or observed_map write was performed.
```

Next recommended step:

```text
Stage 3B: run actual behavior cloning training on a larger generated expert
dataset. Recommended first action is generating a larger expert dataset before
training, rather than training on only the 5-sample smoke dataset.
```

## 40. Stage 4A-1 Isaac Depth Observation Smoke Test

Stage 4A-1 is complete. This stage intentionally switched away from static NYU
rollout work and validated the first simulator sensing loop:

```text
simulator depth -> measured-only observed voxel map
```

No SSCNet prediction, PredictionLayer, paper expert scorer, RL, PPO,
imitation-learning training, behavior-cloning training, target labels, or
ground-truth map was used.

Isaac / IsaacLab:

```text
env: env_isaaclab
python: /home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python
Isaac Lab path: /home/ubuntu22/IsaacLab
Isaac Lab commit: 090aed18163b2194d5551c7919f7539283677743
Isaac Lab package: 0.54.3
Isaac Sim path: /home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim
Isaac Sim package metadata: 5.1.0.0
Isaac Sim VERSION: 5.1.0-rc.19+release.26219.9c81211b.gl
GPU: NVIDIA GeForce RTX 5080, driver 580.159.03
```

Headless startup:

```text
Official empty scene launched with:
/home/ubuntu22/IsaacLab/isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

Official USD camera tutorial produced depth with:
python scripts/tutorials/04_sensors/run_usd_camera.py --headless --enable_cameras

Working headless camera environment required:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
__GLX_VENDOR_LIBRARY_NAME=nvidia
unset DISPLAY WAYLAND_DISPLAY XAUTHORITY GNOME_SETUP_DISPLAY
```

Official sensor output:

```text
depth key: distance_to_image_plane
official tutorial depth shape: torch.Size([2, 480, 640, 1])
```

Minimal depth scene:

```text
script: /home/ubuntu22/sc_explorer_ws/sim_explorer/minimal_depth_scene.py
output dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke
camera resolution: 160 x 120
camera max depth: 5m
poses:
  pose 0: center, yaw 0 deg
  pose 1: center, yaw 90 deg
  pose 2: shifted forward, yaw 0 deg
```

Depth outputs:

```text
depth_000.npy: shape (120, 160), dtype float32, min 1.3499999, max 3.9250004
depth_001.npy: shape (120, 160), dtype float32, min 1.6134452, max 3.9250004
depth_002.npy: shape (120, 160), dtype float32, min 0.3499999, max 2.9250004
camera_info.json
pose_000.json
pose_001.json
pose_002.json
scene_metadata.json
```

Observed voxel map:

```text
script: /home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py
voxel states: UNKNOWN=-1, FREE=0, OCCUPIED=1
voxel size: 0.1m
map bounds: x [-4, 4], y [-4, 4], z [0, 3]
observed map shape: (80, 80, 30)
unknown_count: 143335
free_count: 44435
occupied_count: 4230
observed_ratio: 0.2534635416666667
```

Outputs:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step0.npy
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step1.npy
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step2.npy
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_summary.json
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

Current limitations:

```text
The scene is a minimal synthetic indoor-like room, not the SC-Explorer planner.
The observed map uses simple per-pixel ray marching and a level yaw-only pose convention.
No semantic scene completion prediction has been connected.
No paper expert scoring has been connected.
No online exploration policy, planner, RL, or IL training has been run.
```

Next recommended step:

```text
Stage 4A-2: connect one simulator observed voxel map step to PredictionLayer
and the paper expert scorer, while keeping prediction separate from observed_map.
```

## 41. Stage 4A-2 Simulator Observed Map Expert Step

Stage 4A-2 is complete. It validates the one-step simulator path:

```text
Isaac measured-only observed map -> frontier/candidate generation
  -> observed-map raycast visibility -> paper-style gain scoring
  -> best next viewpoint selection
```

This stage deliberately uses `EmptyPredictionLayer` only. It does not run
SSCNet on Isaac depth, does not read NYU `target_lr` or `target_hr`, does not
use scene ground truth, and does not write prediction into `observed_state`.

New simulator files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py
```

Input:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step2.npy
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_summary.json
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/camera_info.json
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/pose_002.json
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/scene_metadata.json
```

Observed map stats:

```text
shape: (80, 80, 30)
unknown_count: 143335
free_count: 44435
occupied_count: 4230
observed_ratio: 0.2534635416666667
voxel_size: 0.1m
bounds: x [-4, 4], y [-4, 4], z [0, 3]
current pose world: [1.0, 0.0, 1.2]
current pose grid: [50, 40, 11]
current yaw: 0.0 rad
```

Expert stats:

```text
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
gain_occ: 0.0
gain_conf: 0.0
path_cost: 0.8217694333482268
grid position: [51, 38, 14]
world position: [1.15, -0.15, 1.45]
yaw: -0.7030942394487684 rad
```

Outputs:

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

Smoke test result:

```text
Stage 4A-2 simulator expert smoke test passed.
observed_state_modified: no
rl_or_optimizer_or_policy_training_run: no
```

Current limitations:

```text
EmptyPredictionLayer only.
No SSCNet on Isaac depth yet.
No full continuous rollout yet.
No A* or collision-checked path planner yet.
No RL, PPO, behavior cloning, imitation-learning training, or optimizer step.
```

Next recommended step:

```text
Stage 4A-3: turn this one-step expert into a loop:
move camera to best candidate, capture new depth, update observed_map, run expert again.
Keep prediction empty until Isaac-depth-to-SSCNet preprocessing is solved.
```

## 42. Stage 4A-3 Empty-Prediction Expert Rollout

Stage 4A-3 is complete. This stage is still simulator/expert infrastructure,
not learning. It performs a deterministic multi-step rollout in Isaac:

```text
current camera pose -> RGB/depth capture -> measured-only observed_map update
  -> EmptyPredictionLayer expert scoring -> best candidate
  -> planar teleport camera motion -> repeat
```

New / updated files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py
```

Rollout output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000
```

Run log:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a3_rollout_empty_pred.log
```

Configuration:

```text
scene: minimal indoor-like Isaac room
prediction_mode: empty
prediction_layer: EmptyPredictionLayer
gain_mode: hybrid
motion_mode: planar
camera_height: 1.2m
max_steps: 10
num_candidates: 64
top_n: 16
voxel_size: 0.1m
pixel_stride: 2
map bounds: x/y [-4,4], z [0,3]
```

Result:

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
best_score min/mean/max: 29.41531522194122 / 105.48766454499457 / 190.10038228379815
gain_exp min/mean/max: 37.0 / 63.8 / 89.0
gain_sc min/mean/max: 0.0 / 0.0 / 0.0
path_cost min/mean/max: 0.2 / 0.8726943498167937 / 2.6516798957102083
```

Saved artifacts:

```text
step_000.npz ... step_009.npz
transitions.jsonl
observed_state_step000.npy ... observed_state_step009.npy
observed_state_final.npy
episode_summary.json
camera_info.json
pose_000.json ... pose_009.json
rgb_000.png ... rgb_009.png
depth_000.npy ... depth_009.npy
rollout_topdown_path.png
observed_ratio_curve.png
frontier_count_curve.png
step_topdown_000.png ... step_topdown_009.png
rollout_index.html
viz_summary.json
```

Smoke test:

```text
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a3_rollout_test.log
synthetic_transition_serialization: ok
real_episode_validation: ok, 10 steps
observed_ratio_non_decreasing: yes
gain_sc_empty_prediction: zero
prediction_writes_observed_map: no
rl_optimizer_bc_training_run: no
```

Boundary notes:

```text
No RL, PPO, behavior-cloning training, imitation-learning training, policy
optimization, optimizer step, or model save was run.
No SSCNet inference on Isaac depth was run.
No PredictionLayer / SSCNet map_predict was connected.
No NYU target_lr or target_hr was used.
No scene ground truth or simulator ground truth was used.
Prediction was not written into observed_state.
Observed_map remains measured-only from Isaac depth.
Motion is teleport planar camera movement, not physical collision-checked path
execution.
No A* planner and no full SC-Explorer RRT tree planner is implemented yet.
```

Next recommended step:

```text
Stage 4A-4: generate multiple rollout episodes with randomized scripted rooms
or start poses, still using EmptyPredictionLayer, to create a sequential expert
dataset.

Alternative Stage 4A-3.5: add A* over observed FREE space before scaling to
multi-episode generation.
```

## 43. Stage 4A-3.2 Medium-Complexity Scripted Scene

Stage 4A-3.2 is complete. This stage deliberately stays outside training and
prediction integration: it only adds a more complex scripted Isaac indoor
scene, captures fixed RGB/depth views, fuses measured-only depth into an
observed voxel map, renders visualizations, and runs one optional
EmptyPredictionLayer one-step expert smoke.

New / updated simulator files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/medium_complex_depth_scene.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/render_medium_complex_scene_views.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_medium_complex_scene_metadata.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py
```

Scene:

```text
bounds: x/y [-6,6], z [0,3]
floor: 12m x 12m
wall height: 2.2m
rooms: 3
corridors: 1
openings: 3
walls: 13
obstacles: 13 cuboid boxes
camera poses: 5
main camera: 160 x 120, max depth 8m
overview camera: 640 x 480
```

Depth / observed-map output:

```text
output dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_smoke
depth files: depth_000.npy ... depth_004.npy
rgb files: rgb_000.png ... rgb_004.png
observed_state files: observed_state_step0.npy ... observed_state_step4.npy
final observed map: observed_state_final.npy
observed map shape: (120, 120, 30)
unknown/free/occupied: 339813 / 86064 / 6123
observed_ratio: 0.21339583333333334
```

Visualization output:

```text
output dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_viz
scene_overview_rgb.png
scene_overview_depth_color.png
scene_layout_topdown.png
camera_rgb_grid.png
camera_depth_grid.png
observed_topdown_compare.png
free_occupied_voxels_3d_final.png
slices_final.png
```

Optional one-step expert smoke on the final fixed-pose observed map:

```text
output dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_smoke
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

Boundary notes:

```text
No RL, PPO, behavior-cloning training, imitation-learning training, optimizer
step, policy training, model save, SSCNet retraining, or SSCNet inference on
Isaac depth was run.
No PredictionLayer / SSCNet map_predict was connected.
No prediction was written into observed_map.
No NYU target_lr/target_hr, scene ground truth, or simulator ground truth was
used for exploration.
Observed_map remains measured-only from Isaac depth.
```

Stage 4A-3.2 handoff, now completed by Stage 4A-3.5:

```text
Stage 4A-3.5: add A* over observed FREE.
Then Stage 4A-4: run multi-step rollout on the medium scene.
```

## 44. Stage 4A-3.5 A* Observed-Free Path Cost

Stage 4A-3.5 is complete. It adds an observed-free A* path-cost mode to the
simulator expert while keeping `EmptyPredictionLayer` as the only prediction
layer. No SSCNet prediction is connected to Isaac depth in this stage.

Files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/astar_planner.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_astar_planner.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_astar.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
```

Planner rule:

```text
2D traversability comes only from observed_state.
FREE in the robot-height band is required.
OCCUPIED in the band is blocked and inflated by robot radius.
UNKNOWN is not traversable.
No scene metadata, target labels, ground truth, or prediction output is used
for traversability.
```

One-step medium A* expert result:

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
Euclidean comparison: prior medium one-step selected the same candidate with
path_cost 2.051412 and score 53.62160777611031, but did not reject unreachable
candidates.
```

Medium A* rollout result:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred
episode: medium_three_rooms_astar_empty_pred_000
episode dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred/episodes/medium_three_rooms_astar_empty_pred_000
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a35_medium_rollout_astar_empty_pred.log
steps_completed: 5
done_reason: no_valid_candidate
observed_ratio: 0.0 -> 0.04308796296296296
final unknown/free/occupied: 413386 / 15863 / 2751
final pose: [0.6500000000000004, 0.6500000000000004, 1.2]
average reachable candidates: 18.4
average best path_cost: 0.9421159585855353
main blocker: at expert step 5 all 64 sampled candidates were unreachable
under conservative observed-free A* traversability.
```

Validation:

```text
py_compile: passed
test_astar_planner.py: passed
test_sim_expert_astar.py: passed
Euclidean one-step regression test: passed
Euclidean rollout regression test: passed
observed_ratio non-decreasing: yes
gain_sc with EmptyPredictionLayer: 0
prediction writes observed_map: no
target/ground-truth fields: none
RL/optimizer/BC/IL training: no
```

Boundary notes:

```text
No RL/PPO.
No behavior cloning or imitation-learning training.
No SSCNet inference on Isaac depth.
No PredictionLayer / map_predict connection.
No prediction writes into observed_map.
No target_lr/target_hr.
No scene or simulator ground truth for exploration.
A* is only used for path-cost scoring.
Motion still teleports; there is no physical path execution.
No full SC-Explorer RRT tree planner is implemented.
```

Next recommended step:

```text
Stage 4A-4: run multi-step EmptyPredictionLayer rollouts on multiple medium
scene seeds/start poses using A* cost, after addressing the no-valid-candidate
failure mode in conservative observed-free traversability/candidate sampling.
Then Stage 4A-5: carefully begin adding map_predict / prediction gain.
```

## 45. Stage 4A-3.6 Reachability-Aware A* Candidate Sampling

Stage 4A-3.6 is complete. It keeps the simulator expert paper-expert-first and
EmptyPredictionLayer-only, while changing A* candidate generation to sample
from the current observed-FREE reachable component before scoring.

Files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/astar_planner.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_reachable_candidate_sampling.py
```

Reachability rules:

```text
Traversability still comes only from observed_state.
UNKNOWN remains non-traversable.
No scene metadata, simulator ground truth, NYU target labels, or prediction
output is used for reachability.
candidate_sampling_mode=auto resolves to reachable_frontier for A* and
frontier for Euclidean mode.
If current xy is not traversable, --snap_start_to_traversable can snap to the
nearest observed traversable cell within --max_snap_radius_cells.
The current/snap start cell is excluded from candidates when alternatives
exist, preventing trivial same-pose selections.
```

One-step medium reachable A* result:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_astar_reachable_smoke
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a36_medium_expert_step_astar_reachable.log
traversable/blocked/unknown 2D cells: 4316 / 1907 / 8177
reachable component count: 1196
reachable frontier-adjacent count: 1196
candidate source: reachable_frontier
reachable/unreachable candidates: 64 / 0
top_n: 16
best score: 88.24634362636618
best gain_exp: 66.0
best gain_sc: 0.0
best path_cost: 0.7479063413600806
best A* path length: 0.28284271247461906m
best grid/world: [58, 82, 11] / [-0.15, 2.25, 1.15]
```

Medium reachable A* rollout result:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_reachable_empty_pred
episode: medium_three_rooms_astar_reachable_empty_pred_000
episode dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_reachable_empty_pred/episodes/medium_three_rooms_astar_reachable_empty_pred_000
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

Boundaries:

```text
No RL/PPO.
No behavior cloning or imitation-learning training.
No SSCNet inference on Isaac depth.
No PredictionLayer / map_predict connection.
No prediction writes into observed_map.
No target_lr/target_hr.
No scene or simulator ground truth for exploration.
No physical path execution.
No full SC-Explorer RRT tree planner.
```

Next recommended stage:

```text
Stage 4A-4: run multiple medium-scene EmptyPredictionLayer A* rollout
episodes with different seeds/start poses.
```

## 46. Stage 4A-4 Multi-Episode Empty-Prediction A* Rollout Dataset

Stage 4A-4 is complete. It generated a deterministic multi-episode sequential
expert rollout dataset in Isaac using EmptyPredictionLayer, measured-only
observed maps, reachable-frontier candidate sampling, and observed-free A*
path-cost scoring.

Files added:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_batch.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_rollout_dataset.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_rollout_dataset_batch.py
```

Rollout runner updates:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
```

Dataset:

```text
root: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar
scene_variant: medium_three_rooms
scene_seeds: 0, 1, 2
start_variants: start_room_a, start_corridor, start_room_b
intended episodes: 9
ok episodes: 9
failed episodes: 0
total transitions: 90
max_steps per episode: 10
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

Outputs:

```text
manifest.jsonl
dataset_summary.json
dataset_summary.md
rollout_dataset_index.html
aggregate_observed_ratio_curve.png
aggregate_observed_ratio_end_bar.png
aggregate_steps_completed_bar.png
aggregate_steps_hist.png
aggregate_done_reasons.png
aggregate_reachable_candidates_curve.png
aggregate_no_valid_candidate_stats.png
```

Logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_batch_empty_pred_astar.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_dataset_summary.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_dataset_test.log
```

Validation:

```text
py_compile: passed
test_reachable_candidate_sampling.py: passed
test_astar_planner.py: passed
test_sim_expert_rollout.py regression: passed
test_rollout_dataset_batch.py: passed
observed_ratio non-decreasing: yes
gain_sc with EmptyPredictionLayer: zero
prediction writes observed_map: no
target/ground-truth fields: none
RL/optimizer/BC/IL training: no
UNKNOWN traversability shortcut: no
Euclidean fallback: no
```

Stage 4A-4 did not connect SSCNet, PredictionLayer, or map_predict. No
SSCNet inference on Isaac depth was run. No RL, PPO, behavior cloning training,
imitation-learning training, optimizer step, target label use, ground-truth
scoring, prediction writeback, physical path execution, or full RRT tree
planner was performed.

Next recommended stage:

```text
Stage 4A-5: connect map_predict / PredictionLayer as a read-only Isaac
prediction layer, beginning with a single-frame preprocessing and
shape-alignment smoke test.
```

## 47. Stage 4A-5 Isaac Single-Frame map_predict Alignment Smoke

Stage 4A-5 is complete. One Isaac depth frame from the Stage 4A-4 dataset was
converted into SSCNet input, passed through the best SSCNet checkpoint, and
aligned into a simulator-native global prediction layer with the same shape as
`observed_state`.

Added:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_sscnet_preprocess.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_prediction_layer.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_map_predict_single.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_isaac_prediction_alignment.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_map_predict_single.py
```

Input:

```text
dataset: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar
episode: medium_three_rooms_seed0_start_room_a_empty_astar
step: 0
depth: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar/depth_000.npy
pose: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar/pose_000.json
observed_state: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar/observed_state_step000.npy
checkpoint: /home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar
```

SSCNet position convention check:

```text
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a5_position_convention_check.log
NYU position samples: shape (480,640), dtype int32, all indices in [0,8294400)
ProjectionLayer: scatter flat index into (240,144,240), then permute to
  network axes (z_forward,y_up,x_right)
Isaac preprocessing: provisional smoke-only local volume x_right [-2.4,2.4],
  y_up [-1.44,1.44], z_forward [0,4.8], high-res voxel 0.02m
```

Run result:

```text
output dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke
depth input shape: (480,640)
position shape: (480,640)
valid position pixels: 166888
logits shape: (1,12,60,36,60)
local prediction shape: (60,36,60)
global prediction shape: (120,120,30)
local confidence min/mean/max: 0.160744 / 0.722959 / 0.997107
local occupied_prob min/mean/max: 0.002893 / 0.348689 / 0.997254
global valid prediction voxels: 56602
global predicted occupied voxels: 15664
predicted_unmeasured voxels: 39400
inference time: 0.1617s
```

Outputs:

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
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a5_isaac_map_predict_single_test.log
observed_state hash unchanged: yes
SimPredictionLayer read-only API: ok
no target_lr/target_hr/ground_truth fields in prediction artifacts: yes
RL/optimizer/BC/IL training: no
SSCNet checkpoint modified/trained: no
expert/rollout used prediction: no
prediction writeback: no
prediction used for collision/traversability/A*: no
```

Boundary decision: Stage 4A-5 introduces `map_predict` only as a read-only
prediction layer. It does not modify measured `observed_state`, and it must
not affect traversability, collision, A*, rollout, or expert decisions until a
separate one-step scoring stage validates the integration.

## 48. Stage 4A-5.1 One-Step SC-Aware Expert Scoring

Stage 4A-5.1 is complete. The simulator-native Stage 4A-5
`SimPredictionLayer` is now used by the one-step simulator expert scorer to
compute SC-aware information gain:

```text
S = observed_state != UNKNOWN
P = prediction_layer.is_predicted(v, tau) and not S[v]
I_exp(v) = 0 if v in S else 1
I_sc(v) = 1 if v in P else 0
I_hybrid(v) = I_exp(v) + I_sc(v)
I_occ(v) = 1 if v in P and predicted occupied else 0
I_conf(v) = abs(0.5 - occupied_prob[v]) if v in P else 0
```

Prediction remains read-only and information-gain-only. It is not used for
candidate sampling, traversability, A*, collision checking, observed map
updates, or ray blocking.

Updated/added:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_with_prediction.py
```

Inputs:

```text
observed_state: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar/observed_state_step000.npy
pose: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar/pose_000.json
prediction: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke/global_prediction_layer.npz
tau: 0.1
```

Runs:

```text
empty baseline:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/empty_baseline
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_empty_baseline.log

SC prediction:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/sc_prediction
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_sc_prediction.log

test:
/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_with_prediction_test.log
```

Results:

```text
empty best id: 11
empty best score: 331.3448560321166
empty best gain_exp/gain_sc/gain_hybrid: 55.0 / 0.0 / 55.0
empty best path_cost: 0.1659902032541859

SC best id: 11
SC best score: 662.6897120642332
SC best gain_exp/gain_sc/gain_hybrid: 55.0 / 55.0 / 110.0
SC best gain_occ: 13.0
SC best gain_conf: 19.406008422374725
SC best path_cost: 0.1659902032541859
candidates with gain_sc > 0: 64 / 64
max/mean gain_sc: 174.0 / 71.59375
total predicted_unmeasured visible count: 4582
best candidate changed: false
score delta: 331.3448560321166
gain_hybrid delta: 55.0
top-N overlap: 16 / 16
```

Outputs:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/comparison_summary.json
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/comparison_summary.md
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/empty_vs_prediction_best_candidate.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/gain_comparison_bar.png
```

Validation:

```text
py_compile: passed
test_sim_expert_with_prediction.py: passed
observed_state hash unchanged: yes
empty mode gain_sc == 0: yes
prediction mode gain_sc nonzero: yes
gain_hybrid identity: yes
prediction used for traversability/collision/A*: no
prediction blocks rays: no
prediction writeback: no
target_lr/target_hr/ground_truth leakage: no
RL/optimizer/BC/IL training: no
rollout run: no
```

## 49. Stage 4A-6 Short Multi-Step SC-Aware Rollout

Stage 4A-6 is complete. The best SSCNet checkpoint is now loaded once at
rollout startup and reused for a short 5-step dynamic simulator rollout. Each
step runs Isaac depth preprocessing, SSCNet inference, local-to-global
prediction alignment, and read-only `SimPredictionLayer` expert scoring.

Added/updated:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_map_predictor.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/compare_sc_pred_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_sc_aware_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
```

Run:

```text
episode: medium_three_rooms_seed0_start_room_a_sc_pred_dynamic_000
scene_variant: medium_three_rooms
scene_seed: 0
start_variant: start_room_a
max_steps: 5
prediction_mode: sim_dynamic
gain_mode: hybrid
path_cost_mode: astar
candidate_sampling_mode: reachable_frontier
motion_mode: planar
checkpoint: /home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a6_sc_pred_dynamic_rollout.log
```

Result:

```text
steps_completed: 5
done_reason: max_steps
observed_ratio: 0.000000 -> 0.05899768518518519
final counts unknown/free/occupied: 406513 / 21226 / 4261
average gain_exp: 49.6
average gain_sc: 49.4
average gain_hybrid: 99.0
average gain_occ: 8.8
average gain_conf: 16.96283725500107
average best_score: 441.9845465468916
candidates_with_gain_sc_positive min/mean/max: 63 / 63.6 / 64
```

map_predict performance:

```text
model_loaded_once: true
average preprocess_time: 0.05369079960000818 s
average inference_time: 0.020522295199771178 s
average alignment_time: 0.03251961960013432 s
average map_predict total time: 0.14326694260016665 s
average expert_time: 1.026360238399866 s
total_wall_time: 19.86559214800036 s
GPU memory peak: 794354176 bytes
GPU: NVIDIA RTX 5080
CPU/RAM context: AMD Ryzen 9 9950X3D, 32 threads, 32GB RAM
```

Comparison against matching Stage 4A-4 empty baseline:

```text
baseline: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar
comparison output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke/comparison_to_empty_baseline
compared_steps: 5
empty final observed_ratio: 0.06896296296296296
SC final observed_ratio: 0.05899768518518519
SC-empty observed_ratio delta: -0.009965277777777774
changed selected actions: 5
mean score delta SC-empty: 233.79287700349096
mean gain_exp delta SC-empty: -5.199999999999996
mean SC gain_sc: 49.4
mean path_cost delta SC-empty: -0.09316736073162746
```

The SC-aware rollout is lower than the measured-only empty baseline on
observed_ratio at the compared 5-step horizon. This is a correctness milestone,
not evidence of improved exploration performance.

Validation:

```text
py_compile: passed
test_sim_sc_aware_rollout.py: passed
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a6_sc_aware_rollout_test.log
observed_state hash unchanged by prediction: yes
gain_sc nonzero: yes
gain_hybrid identity: yes
prediction used for traversability/collision/A*: no
prediction blocks rays: no
prediction writeback: no
checkpoint modified: no
target_lr/target_hr/ground_truth leakage: no
RL/PPO/BC/IL/optimizer/SSCNet training: no
```

Stage 4A-6 keeps prediction information-gain-only. It does not write into the
measured simulator map and does not affect A* traversability, collision
checking, candidate reachability, or ray blocking.

Per-step selected SC gains were nonzero on every completed step:

```text
step 0: gain_sc 53, gain_hybrid 106, predicted_unmeasured 39375
step 1: gain_sc 52, gain_hybrid 105, predicted_unmeasured 37537
step 2: gain_sc 48, gain_hybrid 96, predicted_unmeasured 33943
step 3: gain_sc 53, gain_hybrid 106, predicted_unmeasured 33149
step 4: gain_sc 41, gain_hybrid 82, predicted_unmeasured 31587
```

No SSCNet training or checkpoint modification occurred. The checkpoint stat and
sha256 before/after the rollout match:

```text
sha256: 003b49eb784f6381d8085c6057e0d3535328899c97dd8adf580f9840a240d5d8
size: 446407 bytes
mtime_ns: 1780049442343013179
```

Fixed issue:

```text
initial failure: step 0 prediction visualization could not find observed_state_source
fix: write observed_state_source/depth_source/pose_source/camera_info_source into prediction_alignment_summary.json
final status: fixed; dynamic rollout, comparison, and validation test passed
```

Next recommendation is Stage 4A-6.1 analysis/ablation/tuning before any longer
rollout: static prediction ablation, gain_sc weighting, tau sweep, optional cap
on prediction gain, overlay/action inspection, and then a 10-step comparison
against the measured-only baseline. Do not jump to RL or IL training.

## 50. Stage 4A-6.1 SC-Aware Rollout Analysis and Ablation

Stage 4A-6.1 is complete. The stage did not train SSCNet and did not modify the
checkpoint. It only analyzed the existing 5-step SC-aware rollout and ran five
small 5-step scoring ablations with the same read-only `SimPredictionLayer`
boundary.

Implemented:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/analyze_sc_rollout_behavior.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sc_pred_ablation_sweep.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_sc_pred_ablation.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sc_pred_ablation.py
```

Updated simulator expert scoring with optional ablation-only controls:

```text
sc_gain_weight: default 1.0
sc_gain_cap: default None
score_gain_mode: hybrid_raw | hybrid_weighted, default hybrid_raw
```

Raw paper-style gains remain logged:

```text
gain_exp
gain_sc
gain_hybrid = gain_exp + gain_sc
gain_occ
gain_conf
```

New weighted/capped fields are additional diagnostics:

```text
weighted_gain_sc
gain_hybrid_weighted
utility_hybrid_weighted
sc_gain_weight
sc_gain_cap_value
```

Existing Stage 4A-6 analysis:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/existing_sc_vs_empty
empty final observed_ratio at 5 steps: 0.06896296296296296
original SC final observed_ratio: 0.05899768518518519
SC-empty delta: -0.009965277777777774
first SC lag step: 1
changed selected actions: 5 / 5
mean gain_exp empty / SC: 54.8 / 49.6
mean gain_sc SC: 49.4
mean path_cost empty / SC: 0.36998367643136965 / 0.2768163156997422
mean best_score empty / SC: 208.19166954340062 / 441.9845465468916
```

Ablations completed:

```text
dynamic_w025_tau01
dynamic_w05_tau01
dynamic_w1_tau03
dynamic_w1_tau01_cap50
static_step0_weight_1p0_tau_0p1
```

All five ablations completed 5 steps and ended at the same measured coverage as
the original SC rollout:

```text
final observed_ratio: 0.05899768518518519
delta vs empty baseline: -0.009965277777777774
delta vs original SC: 0.0
changed actions vs empty: 5
same selected actions as original SC: 5 / 5 for every ablation
```

Performance:

```text
dynamic avg map_predict inference per ablation: 0.0207 s to 0.0301 s
static step0 avg map_predict inference: 0.0 s
dynamic GPU memory peak per ablation: 794296320 bytes
checkpoint modified: no
```

Qualitative / summary outputs:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_ablation/summary
/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/qualitative_inspection
```

Validation:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a61_existing_sc_analysis.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a61_ablation_sweep.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a61_ablation_summary.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a61_ablation_test.log

py_compile: passed
test_sc_pred_ablation.py: passed
prediction read-only: yes
prediction information-gain-only: yes
prediction writeback/traversability/collision/A*/ray blocking: no
target/ground-truth leakage: no
RL/PPO/BC/IL/optimizer/training: no
SSCNet checkpoint modified: no
```

## 53. Stage 4A-6.4 Calibrated / Confidence-Gated I_sc

Stage 4A-6.4 implemented and validated selective prediction gain for the
simulator expert. The goal was to stop treating every predicted-unmeasured
voxel as `+1` for action scoring, while preserving raw `gain_sc` as a logged
diagnostic. The stage did not train SSCNet, did not modify the checkpoint, did
not run RL/PPO/BC/IL, did not scale rollouts, did not use prediction for A* or
collision, and did not write prediction into `observed_state`.

Key scripts:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/calibrate_prediction_gain.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sc_gain_gating_ablation.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_sc_gain_gating.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sc_gain_gating.py
```

Updated scoring/logging:

```text
raw gain_sc: logged as the original predicted-unmeasured count
effective_gain_sc: formula-gated SC contribution
weighted_gain_sc: sc_gain_weight * min(effective_gain_sc, sc_gain_cap)
score_gain_mode=hybrid_weighted: gain_exp + weighted_gain_sc
```

Calibration output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/calibration
samples: 11175
occupied_prob weighted bin correlation: 0.8699543518514645
confidence weighted bin correlation: 0.893222674245022
recommended occ/conf thresholds: 0.9 / 0.9
calibrated_occupied usable: true
```

Future observed maps were used only for post-hoc reliability-table estimation,
not runtime planning or expert scoring.

5-step ablation output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/ablation
completed configs: occupied_only_occ07, occupied_only_occ08,
  occupied_margin_occ06_w05, confidence_weighted_conf05_cap30
failed configs: []
```

5-step result:

```text
empty baseline observed_ratio: 0.06896296296296296
fixed raw SC observed_ratio: 0.05899768518518519
all gated configs observed_ratio: 0.05899768518518519
changed actions vs fixed raw SC: 0/5 for all completed configs
```

Selectivity:

```text
mean raw gain_sc: 49.4
mean effective_gain_sc occupied_only_occ07: 4.2
mean effective_gain_sc occupied_only_occ08: 3.2
mean effective_gain_sc occupied_margin_occ06_w05: 1.7860426306724548
mean effective_gain_sc confidence_weighted_conf05_cap30: 36.19095666408539
```

Summary and validation:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/summary
/home/ubuntu22/sc_explorer_ws/logs/stage4a64_gain_gating_summary.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a64_py_compile.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a64_gain_gating_test.log

py_compile: passed
test_sc_gain_gating.py: passed
prediction read-only: yes
prediction information-gain-only: yes
prediction writeback/traversability/collision/A*/ray blocking: no
future observations used for planning/scoring: no
target/ground-truth leakage: no
RL/PPO/BC/IL/optimizer/training: no
SSCNet checkpoint modified: no
```

Conclusion:

```text
The gated gain formulas work and reduce effective SC gain, but action ranking
does not change in this seed/start. The remaining blocker is not raw +1 gain
alone; it is that prediction gain is still rank-insensitive relative to
measured-frontier gain and path cost. Next inspect candidate-level
score/rank decomposition and qualitative spatial placement of selected vs
rejected candidates under gated formulas. Do not jump to RL/IL or longer
rollout scaling.
```

Conclusion:

```text
The underperformance is most likely due to prediction scoring quality/calibration
rather than rollout plumbing. gain_sc is dense across nearly all reachable
candidates, and the expert remains drawn to nearby low-path-cost local actions.
Weight, tau, and cap changed utility values but did not change the selected
actions in this small sweep. The next stage should inspect Isaac-to-SSCNet
preprocessing, global alignment, confidence calibration, and NYU-to-Isaac
domain shift before longer SC-aware rollout scaling. Do not jump to RL/IL.
```

## 51. Stage 4A-6.2 map_predict Preprocessing / Alignment / Calibration Diagnostics

Stage 4A-6.2 completed offline diagnostics for Isaac map_predict
preprocessing, projection alignment, confidence calibration, and domain shift.
The stage did not train SSCNet, did not modify the checkpoint, did not run RL
or IL, did not use future measurements for planning, and did not write
prediction into `observed_state`.

New diagnostic scripts:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_isaac_sscnet_preprocess.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_prediction_global_alignment.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/evaluate_prediction_against_future_observed.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_alignment_variant_sweep.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_map_predict_diagnostics.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_map_predict_diagnostics.py
```

Diagnostics root:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics
```

Main findings:

```text
Isaac mean depth: 2.532350206375122 m
NYU mean depth: 2.8481276988983155 m
Isaac valid position ratio: 0.565763671875
NYU position nonzero proxy ratio: 0.74495458984375
mean valid prediction in-front ratio: 0.9977230302200312
mean inside global bounds ratio: 0.8669629629629629
best diagnostic alignment variant: xz_swap_variant
default alignment rank: 7
Brier improvement vs default: 0.0735458940774611
tau=0.1 predicted_unmeasured mean: 35118.2
tau=0.1 later measured fraction: 0.059004217437215026
tau=0.1 occupied precision: 0.25632042463242544
tau=0.1 occupied Brier: 0.2786559495144023
ECE-like occupied calibration: 0.3405436085907938
gain_exp/gain_sc correlation: 0.9647202023737985
final_score vs inverse path_cost correlation: 0.9713818732156227
```

Interpretation:

```text
The direct frustum sanity check does not show a gross behind-camera yaw error,
but the diagnostic variant sweep found that an x/z swap-style projection fits
future sensor measurements better than current_default. This is enough to make
alignment convention the primary suspected issue before more rollout tuning.
Independently, the prediction is too dense and poorly calibrated on Isaac at
tau=0.1: only about 5.9% of predicted-unmeasured voxels are later measured in
the 5-step horizon, tau does not reduce density enough until it discards most
coverage, and gain_sc remains highly correlated with gain_exp.
```

Final recommendation:

```text
Stage 4A-6.3 should fix/reconcile the local prediction to global projection
convention and rerun Stage 4A-5/5.1/6 smoke. Keep prediction read-only and
information-gain-only. Do not jump to RL/IL. If alignment is fixed but dense
calibration remains, implement calibrated/capped confidence-based I_sc before
any longer rollout scaling.
```

Validation logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a62_preprocess_stats.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a62_global_alignment.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a62_future_observed_eval.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a62_alignment_variant_sweep.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a62_diagnostic_summary.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a62_diagnostics_test.log
```

Validation result:

```text
py_compile: passed
test_map_predict_diagnostics.py: passed
observed_state modified: no
prediction writeback/traversability/collision/A*/ray blocking: no
future observations marked post-hoc evaluation only: yes
checkpoint modified: no
diagnostics can run without Isaac startup: yes
log scan blockers: none found
```

## 52. Stage 4A-6.3 SSCNet Alignment Convention Fix

Stage 4A-6.3 fixed/reconciled the SSCNet output-axis convention used by Isaac
`map_predict`. It did not train SSCNet, did not modify the checkpoint, did not
run RL/PPO/BC/IL, and did not use future observations for planning.

Key code audit result:

```text
Project2Dto3D scatter: flat -> view(W,H,D) -> permute(D,H,W)
Python raw dataloader flatten: np.ravel_multi_index((x,y,z),(240,144,240))
C++/ROS projection flatten: z*(240*144)+y*240+x
repackaged npz branch: loads precomputed position
dataloader target_lr.T: reverses stored axes before loss flattening
```

The new named conventions are:

```text
current_default_v0:
  input position flatten (x_right,y_up,z_forward)
  output axes (z_forward,y_up,x_right)

xz_swap_diagnostic:
  reprojects existing local predictions as output axes
  (x_right,y_up,z_forward); diagnostic only

code_consistent_v1:
  input position flatten (z_forward,y_up,x_right), matching voxel_util.cpp
  output axes (x_right,y_up,z_forward)
```

Convention validation:

```text
axis audit: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a63_alignment_fix/axis_convention_audit
convention eval: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a63_alignment_fix/convention_eval
best diagnostic convention: xz_swap_diagnostic
recommended fixed convention: code_consistent_v1
current_default_v0 occupied Brier: 0.2786559495144023
code_consistent_v1 occupied Brier: 0.20511005543694122
Brier improvement: 0.0735458940774611
current_default_v0 ECE-like: 0.3405436085907937
code_consistent_v1 ECE-like: 0.22427722861569463
```

Fixed smoke results:

```text
single-frame output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke_alignment_fixed
one-step expert output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_alignment_fixed_smoke
5-step rollout output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_alignment_fixed_smoke

single-frame global_valid_prediction_count: 56602
single-frame predicted_unmeasured_count: 39400
one-step best gain_exp/gain_sc/gain_hybrid: 55.0 / 55.0 / 110.0
5-step observed_ratio: 0.0 -> 0.05899768518518519
empty baseline 5-step observed_ratio: 0.06896296296296296
original SC 5-step observed_ratio: 0.05899768518518519
changed actions vs original SC: 0
```

Interpretation:

```text
The x/z diagnostic improvement is now explained by the C++/ROS projection
index convention rather than adopted blindly. Future Isaac map_predict runs
should use code_consistent_v1. The fixed alignment improves post-hoc prediction
diagnostics but does not improve the 5-step SC-aware rollout; gain_sc remains
dense and largely duplicates gain_exp, with path cost still dominating action
selection.
```

Validation:

```text
py_compile: passed
test_alignment_convention_fix.py: passed
prediction read-only: yes
prediction information-gain-only: yes
prediction writeback/traversability/collision/A*/ray blocking: no
future observations used only for post-hoc evaluation: yes
target/ground-truth leakage: no
RL/PPO/BC/IL/optimizer/training: no
SSCNet checkpoint modified: no
```

## 54. Stage 4A-6.5a Candidate Rank Sensitivity Diagnosis

Stage 4A-6.5a used existing rollout/ablation outputs only to diagnose why
calibrated/gated `I_sc` did not alter action ranking. It did not train SSCNet,
modify checkpoints, change expert scoring, launch Isaac, or run new rollouts.

```text
script: /home/ubuntu22/sc_explorer_ws/sim_explorer/analyze_candidate_rank_sensitivity_small.py
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65a_rank_sensitivity
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a65a_rank_sensitivity.log
```

Rank diagnosis:

```text
gated selected candidate ids/positions identical across steps 0..4: yes
top-1 stable vs fixed raw SC: yes
final_score best explained by inverse path_cost: Pearson 0.8919154707376216
final_score vs effective_gain_sc Pearson: 0.03806071813182923
selected low-path-cost rank mean: 1.0333333333333334
selected gain_exp rank mean: 14.4
candidate_ids missing per candidate: reported, not fatal
next small task if path-cost dominance is the focus: offline counterfactual
score analysis
```

## 55. Stage 4A-6.5b Offline Counterfactual Score Analysis

Stage 4A-6.5b analyzed alternate score formulas offline using only the
Stage 4A-6.5a candidate rank table and summaries. It did not train SSCNet,
modify checkpoints, rerun map_predict, change expert runtime behavior, launch
Isaac, or run new rollouts.

```text
script: /home/ubuntu22/sc_explorer_ws/sim_explorer/offline_score_counterfactuals_small.py
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65b_counterfactual_scores
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a65b_counterfactual_scores.log
```

Counterfactual result:

```text
formula variants executed: 94
configs: 6
steps: 0..4
candidate rows: 480
exp_only_no_cost changed: 30/30 groups
exp/raw/effective over-cost formulas changed: 0 groups
alpha=0.5 vs alpha=1 changed: 80 grouped sweeps
sc_only changed: 10/20 executable groups
SC-specific lambda threshold: min 0.1, median 0.5
later smoke candidate, if any: decoupled_sc_lambda0p5 one-step only
Still not RL/PPO/BC/IL/training.
```
