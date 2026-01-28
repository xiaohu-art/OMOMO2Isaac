import os
import torch
import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

DATA_ROOT = str(_PROJECT_ROOT / "data")
MODELS_ROOT = str(_PROJECT_ROOT / "models")
ROBOTS_ROOT = str(_PROJECT_ROOT / "robots")

OBJECTS_PATH = str(Path(DATA_ROOT) / "objects")
SMPLX_PATH = str(Path(MODELS_ROOT) / "smplx")

OUTPUT_PATH = str(_PROJECT_ROOT / "sequences")

SMPLX_ROBOTS_PATH = str(Path(ROBOTS_ROOT) / "smplx")
G1_PATH = str(Path(ROBOTS_ROOT) / "unitree_g1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Pipeline processing scripts")
    
    parser.add_argument(
        "--original-object-path",
        type=str,
        default=os.path.join(DATA_ROOT, "captured_objects"),
        help="Path to the captured objects directory"
    )
    parser.add_argument(
        "--scaled-object-path",
        type=str,
        default=os.path.join(DATA_ROOT, "objects"),
        help="Path to export scaled objects"
    )
    parser.add_argument(
        '--flag',
        type=str,
        default='train',
        choices=['train', 'test'],
        help='Dataset split'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable visualization'
    )
    
    return parser.parse_args()

SMPLH_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]

MUJOCO_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]