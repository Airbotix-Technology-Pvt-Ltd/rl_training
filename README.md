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

## 🛠️ Project Identity & Ownership

- **Organization**: Airbotix Technology Pvt Ltd.
- **Lead Developer**: **Sumit Bhardwaj** ([@smtbhd32-ABX](https://github.com/smtbhd32-ABX)).
- **Project Mission**: Mastering autonomous quadruped locomotion, beginning with robust stair-climbing capabilities.
- **Official Source**: [Airbotix-Technology-Pvt-Ltd/Lite3_rl_training](https://github.com/Airbotix-Technology-Pvt-Ltd/Lite3_rl_training)

---

## ⚡ Quick Start (Training & Inference)

### 1. Launch Training (Headless)
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --num_envs=4096 \
  --headless
```

### 2. Run Inference (Play)
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --num_envs=10 \
  --keyboard
```

---

## ❤️ Credits & Tribute

We pay tribute and express our sincere gratitude to **DeepRobotics** (and the DeepRoboticsLab team) for providing the foundational `rl_training` stack and robust robot models. Their original work is the baseline upon which we have built our advanced simulation-to-reality and navigation research:

- **SDK Deploy**: [DeepRoboticsLab/sdk_deploy](https://github.com/DeepRoboticsLab/sdk_deploy)
- **RL Training**: [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training)
- **Motion SDK**: [DeepRoboticsLab/Lite3_MotionSDK](https://github.com/DeepRoboticsLab/Lite3_MotionSDK)

- **Original SDK Reference**: [docs/README_DEEP_ROBOTICS.md](docs/README_DEEP_ROBOTICS.md)

---
*Airbotix Technology Pvt Ltd - Lite3 P2P Autonomous Navigation Project*
