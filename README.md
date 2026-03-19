# Airbotix RL Training Setup Guide

**Isaac Lab + rl_training Installation & Configuration**

A complete, reproducible guide for setting up reinforcement learning training environments for DeepRobotics robots (Lite3/M20) using Isaac Lab.

---

## Prerequisites

- **OS**: Ubuntu 22.04 (tested)
- **GPU**: NVIDIA CUDA-capable GPU (recommended: RTX 3090 or higher)
- **RAM**: Minimum 16GB (32GB recommended for 4096 parallel environments)
- **Disk Space**: 50GB+ available

---

## Version Pinning

To ensure reproducibility and avoid compatibility issues, use the exact commits specified below. **Version mismatches are a common source of errors.**

| Component   | Version/Commit                           | Status |
|-------------|------------------------------------------|--------|
| Ubuntu      | 22.04                                    | ✓      |
| Python      | 3.11                                     | ✓      |
| Isaac Sim   | 5.1.0 (pip)                              | ✓      |
| PyTorch     | 2.7.0 + CUDA 12.8                        | ✓      |
| IsaacLab    | `0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20` | ✓      |
| rl_training | `1e9e10dd5ae7715edd1f29913a54ab976d0c23ff` | ✓      |

---

## Installation Steps

### 1. Create Python Environment

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
pip install --upgrade pip
```

### 2. Install Isaac Sim

```bash
pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
```

**First Launch**: The initial run will download required extensions (~10 minutes). Accept the NVIDIA EULA when prompted.

Verify installation:
```bash
isaacsim
```

### 3. Install PyTorch with CUDA 12.8

```bash
pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install Isaac Lab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Checkout the verified commit
git checkout 0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20

# Install Isaac Lab
./isaaclab.sh --install
```

Verify installation:
```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

Expected result: A black simulation window opens without errors.

### 5. Install rl_training

```bash
cd ~
git clone --recurse-submodules https://github.com/DeepRoboticsLab/rl_training.git
cd rl_training

# Checkout the verified commit
git checkout 1e9e10dd5ae7715edd1f29913a54ab976d0c23ff

# Install in development mode
python -m pip install -e source/rl_training
```

---

## Verification

### Check Available Environments

```bash
python scripts/tools/list_envs.py
```

**Expected output**:
- `Rough-Deeprobotics-Lite3-v0`
- `Rough-Deeprobotics-M20-v0`

### Run Sanity Test

Before training, verify the complete pipeline:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --num_envs=2
```

**Verification checklist**:
- ✓ Robot loads without errors
- ✓ Simulation runs without crashing
- ✓ No "tuple/dict" observation format errors
- ✓ Clean shutdown when window is closed

---

## Training

### Start Training

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --num_envs=4096 \
  --headless
```

**Key parameters**:
- `--num_envs=4096`: Optimal for most GPUs; adjust based on available VRAM
- `--headless`: Disables rendering for faster training (recommended)

### Resume Training

To continue from a checkpoint:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --num_envs=4096 \
  --resume \
  --load_run=<run_folder> \
  --checkpoint=model_9999.pt \
  --headless
```

Replace `<run_folder>` with the experiment directory name.

---

## Inference & Evaluation

### Run Trained Policy

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --num_envs=10
```

### Enable Keyboard Control

Add the `--keyboard` flag for manual control:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --num_envs=10 \
  --keyboard
```

**Keyboard Mapping**:
| Action    | Key |
|-----------|-----|
| Forward   | ↑   |
| Backward  | ↓   |
| Strafe    | ←/→ |
| Rotate    | Z/X |

### Record Video

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --num_envs=10 \
  --video \
  --video_length=200
```

**Requirements**: `ffmpeg` must be installed
```bash
sudo apt-get install ffmpeg
```

---

## Monitoring Training

### TensorBoard Logs

```bash
tensorboard --logdir=logs
```

Open browser:
```
http://localhost:6006
```

View real-time metrics:
- Episode returns
- Policy loss
- Critic loss
- Action distributions

---

## Performance Optimization

| Setting              | Value      | Notes                              |
|----------------------|------------|-------------------------------------|
| Parallel Environments| 4096       | Best GPU utilization               |
| Rendering            | Disabled   | Use `--headless` flag              |
| Display Mode (GUI)   | Press `v`  | Disables viewport rendering        |
| GPU Memory           | Monitor    | Reduce `--num_envs` if OOM         |

---

## Troubleshooting

### Error: "input must be Tensor, not tuple"

**Cause**: Observation format mismatch between Isaac Lab and rl_training

**Solution**: Ensure you are using the exact commit specified:
```bash
cd rl_training
git checkout 1e9e10dd5ae7715edd1f29913a54ab976d0c23ff
```

**Alternative fix** (if upgrading is not possible):
```python
# In play.py or train.py, before using observations:
if isinstance(obs, tuple):
    obs = obs[0]
elif isinstance(obs, dict):
    obs = obs["policy"]
```

### Error: "pkg_resources not found"

```bash
pip install setuptools
```

### Error: "GLFW display error" (Docker/XRDP)

```bash
export DISPLAY=:0
```

Then run your training/play command.

### Out of Memory (OOM)

Reduce the number of parallel environments:
```bash
--num_envs=2048  # instead of 4096
```

---

## Cache Management

Isaac Lab generates large temporary files. Clean periodically to prevent disk overflow:

```bash
rm -rf /tmp/IsaacLab/usd_*
```

**Recommendation**: Add to cron job for automated cleanup:
```bash
0 2 * * * rm -rf /tmp/IsaacLab/usd_*
```

---

## Project Structure

```
rl_training/
├── scripts/
│   ├── reinforcement_learning/
│   │   └── rsl_rl/
│   │       ├── train.py       # Training entry point
│   │       ├── play.py        # Inference entry point
│   │       └── ...
│   ├── tools/
│   │   └── list_envs.py       # Verify available environments
│   └── ...
├── source/
│   └── rl_training/           # Main package
├── logs/                       # Training outputs (auto-created)
└── ...
```

Training outputs are saved in `logs/<task>/<date>_<time>_<run_name>/` with:
- Model checkpoints (`.pt` files)
- Training metrics (TensorBoard events)
- Configuration files

---

## Next Steps

1. **Complete Sanity Test**: Run the verification step above
2. **Start Small**: Train with `--num_envs=512` first to verify pipeline
3. **Monitor**: Use TensorBoard to track training progress
4. **Scale**: Increase `--num_envs` to 4096 for production training
5. **Deploy**: Export trained policy for sim-to-real transfer

---

## References

- **IsaacLab**: https://isaac-sim.github.io/IsaacLab/
- **rl_training**: https://github.com/DeepRoboticsLab/rl_training
- **Isaac Sim**: https://docs.omniverse.nvidia.com/app_isaacsim/

---

## Support

For issues specific to:
- **Isaac Lab**: Refer to [official documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
- **rl_training**: Check [repository issues](https://github.com/DeepRoboticsLab/rl_training/issues)
- **DeepRobotics**: Contact the research team

---

## Document Information

- **Last Updated**: March 2026
- **Status**: Verified and tested
- **Python Version**: 3.11
- **Ubuntu Version**: 22.04

---

**⚠️ Important**: Always use the exact commits specified in the Version Pinning section. Version mismatches are the primary source of compatibility issues.
