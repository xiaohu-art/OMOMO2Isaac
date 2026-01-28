# OMOMO Hand Optimization Pipeline

A comprehensive pipeline to process the OMOMO dataset, optimize hand-object interactions, generate SMPLX robot configurations, and retarget to specific humanoid robots for simulation in Isaac Sim.

## Overview

This pipeline processes human manipulation sequences from the OMOMO dataset, performing:
- **Hand pose optimization** using physics-based constraints and contact modeling
- **SMPLX body model reconstruction** with optimized hand-object interactions
- **Robot configuration generation** for MuJoCo/Isaac Sim simulation
- **Sequence canonicalization** and data preprocessing

## Requirements

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/xiaohu-art/OMOMO2Isaac
cd OMOMO2Isaac
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Download required model files:
   - Place SMPL/SMPLH/SMPLX model files in `models/smplx/`, `models/smplh/`, `models/smpl/`
   - Place hand optimization resources in `models/exp/` and `models/assets/`

4. Prepare dataset:
   - Place OMOMO dataset pickle files (`train_diffusion_manip_seq_joints24.p`, `test_diffusion_manip_seq_joints24.p`) in `data/`
   - Place object mesh files in `data/captured_objects/`

## Usage

### 1. Scale Objects

Scale object meshes based on dataset annotations:

```bash
uv run scripts/process_obj_scale.py --flag train/test
```

**Options:**
- `--flag`: Dataset split (`train` or `test`)
- `--original-object-path`: Path to original object meshes (default: `data/captured_objects`)
- `--scaled-object-path`: Path to save scaled objects (default: `data/objects`)

### 2. Process OMOMO Sequences

Process manipulation sequences with hand optimization:

```bash
# Process training set
uv run scripts/process_omomo.py --flag train

# Process test set with visualization
uv run scripts/process_omomo.py --flag test --visualize
```

**Options:**
- `--flag`: Dataset split (`train` or `test`)
- `--visualize`: Enable 3D visualization during processing

**Output:**
- Processed sequences saved to `sequences/{flag}_sequences.pkl`
- Generated robot XML files in `robots/smplx/`

## Project Structure

```
pipeline/
├── data/                       # Dataset and object meshes
│   ├── objects/                # Scaled object meshes
│   ├── captured_objects/       # Original object meshes
│   └── *_diffusion_manip_seq_joints24.p  # OMOMO dataset files
├── models/                     # Model files and resources
│   ├── smplx/                  # SMPLX model files
│   ├── smplh/                  # SMPLH model files
│   ├── smpl/                   # SMPL model files
│   ├── exp/                    # Hand optimization resources
│   └── assets/                 # Hand prior models
├── robots/                     # Generated robot configurations
│   └── smplx/                  # MuJoCo XML files
├── sequences/                  # Processed sequence outputs
├── scripts/                    # Processing scripts
│   ├── process_obj_scale.py    # Object scaling script
│   └── process_omomo.py        # Main processing pipeline
└── utils/                      # Utility modules
    ├── optimize.py             # Hand pose optimization
    ├── process.py              # Sequence processing
    ├── math.py                 # Math utilities
    ├── visualization.py        # Visualization tools
    └── cli_args.py             # CLI argument parsing
```

## Hand Optimization

The hand optimization module (`utils/optimize.py`) performs hand pose optimization with:

- **Collision Detection**: Penalizes hand-object penetration
- **Contact Modeling**: Heuristic contact masks based on distance thresholds
- **Range of Motion (ROM)**: Enforces joint angle limits for realistic hand poses
- **Temporal Smoothness**: Regularizes pose changes across frames
- **Prior Regularization**: Uses hand pose priors for natural hand configurations

### Optimization Parameters

Key parameters in `HandConfig` class:
- `THRESH_CLOSE`: Close contact threshold (default: 0.02m)
- `THRESH_FAR`: Far contact threshold (default: 0.20m)
- `PENETRATION_LIMIT`: Maximum allowed penetration (default: 0.03m)
- `LR`: Learning rate (default: 0.001)
- `EPOCHS`: Optimization iterations (default: 1000)

## Output Format

Processed sequences are saved as pickle files with the following structure:

```python
{
    'object_name': {
        'sequence_name': {
            'human': {
                'poses': np.ndarray,        # (T, 165) Full pose parameters
                                            #   [0:3] root orientation (axis-angle)
                                            #   [3:66] body pose (21 joints × 3)
                                            #   [66:156] optimized hand pose (30 joints × 3)
                'betas': np.ndarray,        # (16,) SMPL shape parameters
                'trans': np.ndarray,        # (T, 3) Root translation
                'gender': str,              # Gender identifier ('male', 'female', 'neutral')
                'keypoints': np.ndarray,    # (T, J, 3) Joint positions (body + hand)
                'contacts': np.ndarray,     # (T, J) Contact labels
                                            #   1.0: in contact (< 0.05m)
                                            #   0.0: no contact
                                            #   -1.0: far (> 0.2m)
            },
            'object': {
                'rot': np.ndarray,          # (T, 3, 3) Object rotation matrices
                'trans': np.ndarray,        # (T, 3) Object translation
                'name': str,                # Object name
                'scale': float,             # Object scale
            }
        }
    }
}
```

## Citation

- OMOMO dataset
- SMPLX model