# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OMOMO2Isaac: Pipeline to process the OMOMO human manipulation dataset, optimize hand-object interactions, generate SMPLX humanoid configurations for MuJoCo/Isaac Sim, and retarget motions to humanoid robots (Unitree G1).

## Commands

All scripts are run via `uv` (no traditional build system):

```bash
# Install dependencies
uv sync

# Pipeline steps (run in order):
uv run scripts/process_obj_scale.py --flag train    # 1. Scale object meshes
uv run scripts/process_omomo.py --flag train         # 2. Process sequences + hand optimization
uv run scripts/vis_mujoco.py                         # 3. Visualize in MuJoCo
uv run scripts/humanoid_retarget.py                  # 4. Retarget to G1 robot

# Common flags:
#   --flag {train,test}   Dataset split
#   --visualize           Enable 3D visualization (viser web UI)
```

There is no test suite. Validation is done visually via MuJoCo playback and viser visualization.

## Architecture

The pipeline has three stages:

1. **Object scaling** (`scripts/process_obj_scale.py`) — reads OMOMO dataset pickles, computes per-object mean scale, writes scaled OBJ meshes to `data/objects/`.

2. **Sequence processing** (`scripts/process_omomo.py` → `utils/process.py` → `utils/optimize.py`) — the core pipeline:
   - Loads OMOMO pickles, canonicalizes sequences relative to object COM
   - Runs hand pose optimization (1000 epochs Adam) with collision/contact/ROM/smoothness/prior losses
   - Reconstructs full SMPLX body, computes contact labels, outputs to `sequences/{flag}_sequences.pkl`
   - Generates per-subject MuJoCo XML files in `robots/smplx/` via `smpl-sim`'s `SMPL_Robot`

3. **Retargeting** (`scripts/humanoid_retarget.py`) — maps SMPLX keypoints to Unitree G1 joints via IK optimization using `pytorch-kinematics`.

### Key module roles

- `utils/optimize.py` — Hand optimization engine. `HandConfig` (hyperparameters), `HandResources` (lazy-loaded indices/priors, singleton), `HandOptimizer` (main optimizer). Uses SDF-based collision, heuristic contact masks, 5-group ROM constraints, and GrabPrior.
- `utils/math.py` — Rotation conversions (axis-angle, quaternion wxyz, rotation matrix, euler). Key: `rotate_at_frame_w_obj()` for canonicalization, `expand_to_full_smplx_pose()`.
- `utils/cli_args.py` — Path constants (`DATA_ROOT`, `MODELS_ROOT`, `ROBOTS_ROOT`, `SMPLX_PATH`, `OUTPUT_PATH`, `G1_PATH`, `DEVICE`) and joint ordering arrays (`SMPLH_BONE_ORDER_NAMES`, `MUJOCO_BONE_ORDER_NAMES`).
- `utils/visualization.py` — Viser-based 3D visualization with contact-colored keypoints.

### Data flow

- **Input**: OMOMO dataset pickles in `data/` + object meshes in `data/captured_objects/` + SMPL model files in `models/`
- **Intermediate**: Scaled meshes in `data/objects/`, MuJoCo XMLs in `robots/smplx/`
- **Output**: `sequences/{flag}_sequences.pkl` — nested dict keyed by `object_name → seq_name → {human, object}`

### Important conventions

- Pose representation: axis-angle (3D) throughout; full SMPLX pose is (T, 165) = root_orient(3) + body_pose(63) + hand_pose(90) + jaw/eye(9)
- Quaternions use wxyz ordering
- `numpy<2.0` is required (pinned in pyproject.toml)
- CUDA 12.4 expected for PyTorch; falls back to CPU via `DEVICE` in cli_args
- Joint ordering differs between SMPLH and MuJoCo — conversion happens in `vis_mujoco.py`
