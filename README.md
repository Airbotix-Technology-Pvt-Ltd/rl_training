# Lite3 RL Training Hub (Airbotix Fork)

This repository is a specialized fork of the DeepRobotics RL Training stack, optimized for **High-Fidelity Simulation-to-Reality RL Development** for the Lite3 quadruped platform using **Isaac Lab** and **Isaac Sim**.

---

> [!IMPORTANT]
> **Airbotix is exclusively focused on the Lite3 platform.** 
> Our research and architecture are dedicated to achieving state-of-the-art results on the Jueying Lite3, starting with the mastery of stair climbing.

---

## 📊 Project Roadmap & Milestones

### **Phase 1: Stair Climbing Foundation (COMPLETED)**
- [x] **Vertical Navigation Mastery (2026-03-18)**: Successfully trained the Lite3 to climb stairs in Isaac Lab simulation environments.
- [x] **Sim-to-Reality RL Pipeline**: Established the reproducible Isaac Lab + rl_training environment.

### **Phase 2: Intelligent Navigation & Drift Correction (ONGOING)**
- [ ] **Drift Correction Training**: Active policy fine-tuning to eliminate rightward bias during stair ascent.
- [ ] **Fast-LIO SLAM Integration**: Synchronizing the stair-climbing policy with robust 3D state estimation.
- [ ] **Nav2 Path Planning**: Policy optimization for autonomous point-to-point (P2P) goals.

---

# RL Training Environment Setup

[**GitHub Source (Legacy Reference)**](https://github.com/DeepRoboticsLab/rl_training/tree/main) | [**Video Guide**](https://youtube.com/playlist?list=PLy9YHJvMnjO0X4tx_NTWugTUMJXUrOgFH&si=gq3xuWtlPac0y1_o)

A complete, reproducible guide for setting up reinforcement learning training environments for the Lite3 using Isaac Lab.

---

## 📋 Prerequisites
- **OS**: Ubuntu 22.04 (tested)
- **GPU**: NVIDIA CUDA-capable GPU (recommended: RTX 3090 or higher)
- **RAM**: Minimum 16GB (32GB recommended for 4096 parallel environments)
- **Disk Space**: 50GB+ available

---

## 📌 Version Pinning (CRITICAL)
To ensure reproducibility and avoid physics/observation errors, you **MUST** use the exact commits specified below. **Version mismatches in Isaac Lab are the primary source of training failures.**

| Component   | Version/Commit | Status |
|-------------|----------------|--------|
| IsaacLab    | [`0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20`](https://github.com/isaac-sim/IsaacLab/commit/0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20) | **REQUIRED** |
| rl_training | [`Airbotix Fork (This Repo)`](https://github.com/Airbotix-Technology-Pvt-Ltd/Lite3_rl_training) | **REQUIRED** |
| Isaac Sim   | 5.1.0 (pip)    | Verified |

---

## ⚡ Installation Steps

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
*Note: The initial run will download required extensions (~10 minutes). Accept the NVIDIA EULA when prompted.*

### 3. Install PyTorch with CUDA 12.8
```bash
pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install Isaac Lab
```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20
./isaaclab.sh --install
```

### 5. Install `rl_training` (This Repo)
```bash
cd ~
git clone --recurse-submodules https://github.com/Airbotix-Technology-Pvt-Ltd/Lite3_rl_training.git
cd Lite3_rl_training
python -m pip install -e source/rl_training
```

---

## ✅ Verification

### Check Available Environments
```bash
python scripts/tools/list_envs.py
```
**Expected task**: `Rough-Deeprobotics-Lite3-v0`

### Run Sanity Test (Inference)
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --num_envs=2
```

---

## 🚀 Training

### Start Training (Headless)
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --num_envs=4096 \
  --headless
```

### Resume Training
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --num_envs=4096 \
  --resume \
  --load_run=<run_folder> \
  --checkpoint=model_9999.pt \
  --headless
```

---

## 🛠️ Troubleshooting & Optimization

| Issue | Solution |
|-------|----------|
| **Observation Tensor Error** | Use the exact git commit pinned in the version table. |
| **GLFW display error** | `export DISPLAY=:0` |
| **Out of Memory (OOM)** | Reduce `--num_envs` to 2048 or lower. |
| **Cache Management** | `rm -rf /tmp/IsaacLab/usd_*` (Recommended: daily cron job). |

---

## ❤️ Credits & Tribute

We pay tribute and express our sincere gratitude to **DeepRobotics** for providing the foundational `rl_training` stack and robust robot models:

- **SDK Deploy**: [DeepRoboticsLab/sdk_deploy](https://github.com/DeepRoboticsLab/sdk_deploy)
- **RL Training**: [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training)
- **Original SDK Reference**: [docs/README_DEEP_ROBOTICS.md](docs/README_DEEP_ROBOTICS.md)

---
*Airbotix Technology Pvt Ltd - Lite3 P2P Autonomous Navigation Project*
*See our [**Contributors Hub**](../Contributors.md) for full project attribution.*
