"""
Three-stage humanoid retargeting: body IK + ground alignment + hand retargeting.

Stage 1: Full body IK matching SMPLX keypoints to G1 links
Stage 2: Freeze lower body, adjust root Z + upper body for ground alignment
Stage 3: Freeze body, optimize hand DOFs with vector-based fingertip matching
"""

import os
import time
import torch
import joblib
import trimesh
import yourdfpy
import numpy as np
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as sRot

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from utils.cli_args import (
    parse_args, DEVICE,
    OBJECTS_PATH, SMPLX_PATH,
    OUTPUT_PATH, G1_PATH, SMPLH_BONE_ORDER_NAMES
)

G1_URDF_PATH = os.path.join(G1_PATH, "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf")

# Body link mapping: G1 link -> SMPLX joint
JOINT_MAP = {
    "pelvis": "Pelvis",
    "left_hip_pitch_link": "L_Hip",
    "left_knee_link": "L_Knee",
    "left_ankle_roll_link": "L_Ankle",
    "right_hip_pitch_link": "R_Hip",
    "right_knee_link": "R_Knee",
    "right_ankle_roll_link": "R_Ankle",
    "left_shoulder_roll_link": "L_Shoulder",
    "left_elbow_link": "L_Elbow",
    "L_hand_base_link": "L_Wrist",
    "right_shoulder_roll_link": "R_Shoulder",
    "right_elbow_link": "R_Elbow",
    "R_hand_base_link": "R_Wrist",
}

# Hand fingertip link mapping: G1 tip link -> SMPLX fingertip joint
HAND_TIP_MAP = {
    "L": {
        "tip_links": ["L_thumb_tip", "L_index_tip", "L_middle_tip", "L_ring_tip", "L_pinky_tip"],
        "smplx_tips": ["L_Thumb3", "L_Index3", "L_Middle3", "L_Ring3", "L_Pinky3"],
        "wrist_link": "L_hand_base_link",
        "smplx_wrist": "L_Wrist",
    },
    "R": {
        "tip_links": ["R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip"],
        "smplx_tips": ["R_Thumb3", "R_Index3", "R_Middle3", "R_Ring3", "R_Pinky3"],
        "wrist_link": "R_hand_base_link",
        "smplx_wrist": "R_Wrist",
    },
}


# =============================================================================
# Utility functions
# =============================================================================

def parse_joint_limits(urdf_path: str, joint_names: list):
    tree = ET.parse(urdf_path)
    limits = {}
    for joint_el in tree.getroot().findall('joint'):
        limit_el = joint_el.find('limit')
        if limit_el is not None:
            limits[joint_el.get('name')] = (
                float(limit_el.get('lower', 0)),
                float(limit_el.get('upper', 0)),
            )
    lower = torch.tensor([limits[n][0] for n in joint_names], dtype=torch.float32, device=DEVICE)
    upper = torch.tensor([limits[n][1] for n in joint_names], dtype=torch.float32, device=DEVICE)
    return lower, upper


def joint_limit_loss(th, lower, upper):
    below = torch.clamp(lower - th, min=0)
    above = torch.clamp(th - upper, min=0)
    return torch.mean(below ** 2 + above ** 2)


def parse_mimic_joints(urdf_path: str, joint_names: list):
    """Parse mimic joint relationships from URDF.

    Returns: {mimic_idx: (source_idx, multiplier, offset)}
    """
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    tree = ET.parse(urdf_path)
    mimic_map = {}
    for joint_el in tree.getroot().findall('joint'):
        mimic_el = joint_el.find('mimic')
        if mimic_el is not None:
            joint_name = joint_el.get('name')
            source_name = mimic_el.get('joint')
            if joint_name in name_to_idx and source_name in name_to_idx:
                mimic_map[name_to_idx[joint_name]] = (
                    name_to_idx[source_name],
                    float(mimic_el.get('multiplier', 1.0)),
                    float(mimic_el.get('offset', 0.0)),
                )
    return mimic_map


def get_hand_independent_indices(joint_names: list, mimic_map: dict):
    """Get indices of independent (non-mimic) hand joints per side."""
    mimic_indices = set(mimic_map.keys())
    hand_keywords = ('thumb', 'index', 'middle', 'ring', 'pinky')
    result = {}
    for side in ['L', 'R']:
        prefix = f'{side}_'
        result[side] = [
            i for i, n in enumerate(joint_names)
            if n.startswith(prefix)
            and any(kw in n for kw in hand_keywords)
            and i not in mimic_indices
        ]
    return result


def enforce_mimic(th, mimic_map):
    """Overwrite mimic joints from source joints (differentiable)."""
    th = th.clone()
    for mimic_idx, (source_idx, mult, offset) in mimic_map.items():
        th[:, mimic_idx] = mult * th[:, source_idx] + offset
    return th


def build_chain(urdf_path: str):
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    chain = pk.build_chain_from_urdf(urdf_content)
    return chain.to(device=DEVICE)


def to_world(pos_local, trans, rotmat):
    pos_w = rotmat.unsqueeze(1) @ pos_local.unsqueeze(-1)
    return pos_w.squeeze(-1) + trans.unsqueeze(1)


def compute_contact_labels(
    chain: pk.Chain,
    robot_th: torch.Tensor,
    robot_trans: torch.Tensor,
    robot_rotmat: torch.Tensor,
    obj_verts_world: np.ndarray,
    close_thresh: float = 0.05,
    far_thresh: float = 0.20,
):
    """Compute per-link contact labels and distances using cKDTree.

    Args:
        chain: Robot kinematic chain.
        robot_th: (T, n_joints) joint angles.
        robot_trans: (T, 3) root translation.
        robot_rotmat: (T, 3, 3) root rotation matrices.
        obj_verts_world: (T, V, 3) object vertices in world frame.
        close_thresh: Distance below which a link is "in contact" (label=1).
        far_thresh: Distance above which a link is "no contact" (label=-1).

    Returns:
        contact_label: (T, n_links) float array — 1.0 / 0.0 / -1.0
        contact_distance: (T, n_links) float array — min distance per link
        link_names: list of link names corresponding to columns
    """
    T = robot_th.shape[0]
    indices = chain.get_all_frame_indices()
    body_pos = chain.forward_kinematics(robot_th, indices)
    link_names = sorted(body_pos.keys())
    n_links = len(link_names)

    # Stack all link positions → (T, n_links, 3) in local frame, then to world
    local_pos = torch.stack([body_pos[n].get_matrix()[:, :3, 3] for n in link_names], dim=1)
    world_pos = to_world(local_pos, robot_trans, robot_rotmat).detach().cpu().numpy()  # (T, n_links, 3)

    contact_distance = np.zeros((T, n_links), dtype=np.float32)
    contact_label = np.zeros((T, n_links), dtype=np.float32)

    for t in range(T):
        tree = cKDTree(obj_verts_world[t])
        dists, _ = tree.query(world_pos[t])  # (n_links,)
        contact_distance[t] = dists
        contact_label[t, dists < close_thresh] = 1.0
        contact_label[t, dists > far_thresh] = -1.0

    n_contact_frames = np.sum(np.any(contact_label > 0, axis=1))
    print(f"  Contact labels: {n_links} links, {n_contact_frames}/{T} frames with contact")

    return contact_label, contact_distance, link_names


# =============================================================================
# Three-stage retargeting
# =============================================================================

def load_object_mesh(seq_data: dict):
    """Load object mesh and precompute world-space vertices and normals for each frame.

    Returns:
        obj_verts_world: (T, V, 3) ndarray of object vertices in world coordinates.
        obj_normals_world: (T, V, 3) ndarray of vertex normals in world coordinates.
        obj_verts_rest: (V, 3) ndarray of rest-pose vertices.
    """
    obj = seq_data["object"]
    obj_name = str(obj["name"])
    mesh = trimesh.load(os.path.join(OBJECTS_PATH, f"{obj_name}.obj"), force="mesh")
    obj_verts = np.asarray(mesh.vertices, dtype=np.float32)  # (V, 3)
    obj_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)  # (V, 3)
    obj_rot = obj["rot"]    # (T, 3, 3)
    obj_trans = obj["trans"]  # (T, 3)
    # (V,3) @ (T,3,3)^T + (T,1,3) → (T,V,3)
    obj_verts_world = np.einsum("vj,tij->tvi", obj_verts, obj_rot) + obj_trans[:, None, :]
    # Normals rotate but don't translate
    obj_normals_world = np.einsum("vj,tij->tvi", obj_normals, obj_rot)
    return obj_verts_world, obj_normals_world, obj_verts


def run_retarget(chain: pk.Chain, seq_data: dict, fps: int = 30):
    keypoints = torch.tensor(seq_data["human"]["keypoints"], dtype=torch.float32, device=DEVICE)
    T = keypoints.shape[0]

    trans = seq_data["human"]["trans"]
    poses = seq_data["human"]["poses"].reshape(T, -1, 3)
    root_orient = poses[:, 0, :]
    robot_rot = sRot.from_rotvec(root_orient) * sRot.from_euler(
        "xyz", [np.pi / 2, 0.0, np.pi / 2]
    ).inv()
    robot_rotmat = torch.tensor(robot_rot.as_matrix(), dtype=torch.float32, device=DEVICE)

    # Load object mesh (world-space vertices + normals per frame)
    obj_verts_world, obj_normals_world, _ = load_object_mesh(seq_data)
    print(f"Object mesh loaded: {obj_verts_world.shape[1]} vertices, {T} frames")

    # Body link targets
    robot_link_names = list(JOINT_MAP.keys())
    smpl_indices = [SMPLH_BONE_ORDER_NAMES.index(n) for n in JOINT_MAP.values()]
    smpl_keypoints = keypoints[:, smpl_indices, :]

    # Joint config
    joint_names = chain.get_joint_parameter_names()
    joint_lower, joint_upper = parse_joint_limits(G1_URDF_PATH, joint_names)
    mimic_map = parse_mimic_joints(G1_URDF_PATH, joint_names)
    hand_indep = get_hand_independent_indices(joint_names, mimic_map)

    # Initialize
    robot_th = torch.nn.Parameter(torch.zeros(T, chain.n_joints, device=DEVICE))
    robot_trans = torch.nn.Parameter(torch.from_numpy(trans).float().to(DEVICE))
    indices = chain.get_all_frame_indices()

    def stack_positions(body_pos, names):
        return torch.stack(
            [body_pos[name].get_matrix()[:, :3, 3] for name in names], dim=1
        )

    def get_body_keypoints(th, trans, rotmat):
        body_pos = chain.forward_kinematics(th, indices)
        local = stack_positions(body_pos, robot_link_names)
        return to_world(local, trans, rotmat)

    # =====================================================================
    # Stage 1: Full body IK
    # =====================================================================
    print("=" * 60)
    print("Stage 1: Full body IK")
    print("=" * 60)
    opt = torch.optim.Adam([robot_th, robot_trans], lr=0.02)
    for i in range(300):
        opt.zero_grad()

        robot_kp_w = get_body_keypoints(robot_th, robot_trans, robot_rotmat)
        omega = torch.gradient(robot_th, spacing=1.0 / fps, dim=0)[0]

        kp_loss = torch.mean((robot_kp_w - smpl_keypoints) ** 2)
        pos_reg = torch.mean(torch.square(robot_th))
        vel_reg = torch.mean(torch.square(omega))
        jl_loss = joint_limit_loss(robot_th, joint_lower, joint_upper)
        loss = kp_loss + 2e-2 * pos_reg + 1e-3 * vel_reg + 10.0 * jl_loss

        loss.backward()
        opt.step()

        if i % 50 == 0:
            print(f"  iter {i:3d}  loss={loss.item():.6f}  kp={kp_loss.item():.6f}  "
                  f"j_pos={pos_reg.item():.6f}  j_vel={vel_reg.item():.6f}  j_lim={jl_loss.item():.6f}")

    # =====================================================================
    # Stage 2: Ground alignment (freeze lower body)
    # =====================================================================
    print("=" * 60)
    print("Stage 2: Ground alignment")
    print("=" * 60)
    G1_ANKLE_TO_SOLE = 0.035
    # Freeze legs (0-11) + waist (12-14) to preserve torso orientation from Stage 1
    FROZEN_JOINTS = 15

    l_hand_idx = robot_link_names.index("L_hand_base_link")
    r_hand_idx = robot_link_names.index("R_hand_base_link")
    l_ankle_idx = robot_link_names.index("left_ankle_roll_link")
    r_ankle_idx = robot_link_names.index("right_ankle_roll_link")

    GROUND_THRESH = 0.08
    l_toe_idx = SMPLH_BONE_ORDER_NAMES.index("L_Toe")
    r_toe_idx = SMPLH_BONE_ORDER_NAMES.index("R_Toe")
    l_contact = (keypoints[:, l_toe_idx, 2] < GROUND_THRESH).float()
    r_contact = (keypoints[:, r_toe_idx, 2] < GROUND_THRESH).float()

    with torch.no_grad():
        stage1_kp = get_body_keypoints(robot_th, robot_trans, robot_rotmat)
        ee_targets = stage1_kp.detach().clone()

    lower_th = robot_th.data[:, :FROZEN_JOINTS].detach().clone()
    upper_th = torch.nn.Parameter(robot_th.data[:, FROZEN_JOINTS:].detach().clone())
    root_z_offset = torch.nn.Parameter(torch.zeros(T, 1, device=DEVICE))
    opt2 = torch.optim.Adam([upper_th, root_z_offset], lr=0.01)

    for i in range(300):
        opt2.zero_grad()

        full_th = torch.cat([lower_th, upper_th], dim=1)
        adjusted_trans = robot_trans.detach().clone()
        adjusted_trans[:, 2:3] += root_z_offset

        robot_kp_w = get_body_keypoints(full_th, adjusted_trans, robot_rotmat)

        ee_loss = torch.mean((robot_kp_w[:, l_hand_idx, :] - ee_targets[:, l_hand_idx, :]) ** 2) \
                + torch.mean((robot_kp_w[:, r_hand_idx, :] - ee_targets[:, r_hand_idx, :]) ** 2)

        l_sole_z = robot_kp_w[:, l_ankle_idx, 2] - G1_ANKLE_TO_SOLE
        r_sole_z = robot_kp_w[:, r_ankle_idx, 2] - G1_ANKLE_TO_SOLE
        ground_loss = torch.mean(l_contact * l_sole_z ** 2) \
                    + torch.mean(r_contact * r_sole_z ** 2)

        omega = torch.gradient(upper_th, spacing=1.0 / fps, dim=0)[0]
        pos_reg = torch.mean(torch.square(upper_th))
        vel_reg = torch.mean(torch.square(omega))
        jl_loss = joint_limit_loss(upper_th, joint_lower[FROZEN_JOINTS:], joint_upper[FROZEN_JOINTS:])

        loss = 10.0 * ee_loss + 5.0 * ground_loss \
             + 2e-2 * pos_reg + 1e-3 * vel_reg + 10.0 * jl_loss

        loss.backward()
        opt2.step()

        if i % 50 == 0:
            print(f"  iter {i:3d}  loss={loss.item():.6f}  ee={ee_loss.item():.6f}  "
                  f"ground={ground_loss.item():.6f}  j_lim={jl_loss.item():.6f}")

    # Merge stage 2 results
    with torch.no_grad():
        robot_th.data = torch.cat([lower_th, upper_th.data], dim=1)
        robot_trans.data[:, 2:3] += root_z_offset.data

    # =====================================================================
    # Stage 3: Hand retargeting (geometric angle-based, ins-dex approach)
    # =====================================================================
    print("=" * 60)
    print("Stage 3: Hand retargeting (angle-based)")
    print("=" * 60)

    kp_np = keypoints.detach().cpu().numpy()  # (T, 52, 3)

    def _angle_at(kp_frame, origin, p1, p2):
        """Angle in degrees at origin between vectors to p1 and p2."""
        v1 = kp_frame[p1] - kp_frame[origin]
        v2 = kp_frame[p2] - kp_frame[origin]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    def _linear_map(angle, src_low, src_high, dst_low, dst_high):
        """Linearly map angle from [src_low, src_high] to [dst_low, dst_high], clamped."""
        t = (angle - src_low) / (src_high - src_low + 1e-8)
        return np.clip(dst_low + t * (dst_high - dst_low), min(dst_low, dst_high), max(dst_low, dst_high))

    # SMPLX angle limits (calibrated via scripts/calibrate_hand_angles.py)
    # Four fingers: angle at {Finger}1 between Wrist and {Finger}3
    FOUR_FINGER_OPEN = 172.1
    FOUR_FINGER_CLOSED = 52.0

    # Thumb bending: angle at Thumb1 between Thumb3 and Index1
    THUMB_BEND_OPEN = 54.4
    THUMB_BEND_CLOSED = 7.4

    # Thumb rotation: angle at Thumb1 between Thumb3 and Middle1
    # Note: small range (~6°) — thumb yaw sensitivity is limited with this metric
    THUMB_ROT_OPEN = 71.5
    THUMB_ROT_CLOSED = 65.2

    # Robot joint limits (from URDF)
    # Four fingers proximal: [0.0, 1.7] rad
    # Thumb yaw: [-0.1, 1.3] rad
    # Thumb pitch: [-0.1, 0.6] rad

    hand_joint_values = np.zeros((T, chain.n_joints))

    for side in ['L', 'R']:
        w_idx = SMPLH_BONE_ORDER_NAMES.index(f'{side}_Wrist')
        i1_idx = SMPLH_BONE_ORDER_NAMES.index(f'{side}_Index1')
        m1_idx = SMPLH_BONE_ORDER_NAMES.index(f'{side}_Middle1')
        t1_idx = SMPLH_BONE_ORDER_NAMES.index(f'{side}_Thumb1')
        t3_idx = SMPLH_BONE_ORDER_NAMES.index(f'{side}_Thumb3')

        for t in range(T):
            frame = kp_np[t]

            # --- Four fingers: index, middle, ring, pinky ---
            for finger in ['Index', 'Middle', 'Ring', 'Pinky']:
                f1_idx = SMPLH_BONE_ORDER_NAMES.index(f'{side}_{finger}1')
                f3_idx = SMPLH_BONE_ORDER_NAMES.index(f'{side}_{finger}3')
                # Flexion angle at proximal joint
                angle = _angle_at(frame, f1_idx, w_idx, f3_idx)
                # Map: open (170°) -> 0 rad, closed (40°) -> 1.7 rad
                joint_val = _linear_map(angle, FOUR_FINGER_OPEN, FOUR_FINGER_CLOSED, 0.0, 1.7)

                joint_name = f'{side}_{finger.lower()}_proximal_joint'
                j_idx = joint_names.index(joint_name)
                hand_joint_values[t, j_idx] = joint_val

            # --- Thumb bending (pitch) ---
            # Angle at Thumb1 between Thumb3 and Index1
            bend_angle = _angle_at(frame, t1_idx, t3_idx, i1_idx)
            thumb_pitch = _linear_map(bend_angle, THUMB_BEND_OPEN, THUMB_BEND_CLOSED, -0.1, 0.6)

            pitch_name = f'{side}_thumb_proximal_pitch_joint'
            hand_joint_values[t, joint_names.index(pitch_name)] = thumb_pitch

            # --- Thumb rotation (yaw) ---
            # Angle at Thumb1 between Thumb3 and Middle1
            rot_angle = _angle_at(frame, t1_idx, t3_idx, m1_idx)
            thumb_yaw = _linear_map(rot_angle, THUMB_ROT_OPEN, THUMB_ROT_CLOSED, -0.1, 1.3)

            yaw_name = f'{side}_thumb_proximal_yaw_joint'
            hand_joint_values[t, joint_names.index(yaw_name)] = thumb_yaw

        print(f"  {side} hand done")

    # Apply temporal smoothing (exponential moving average)
    alpha = 0.7  # smoothing factor: higher = less smoothing
    for side in ['L', 'R']:
        for j_idx_val in hand_indep[side]:
            for t in range(1, T):
                hand_joint_values[t, j_idx_val] = (
                    alpha * hand_joint_values[t, j_idx_val]
                    + (1 - alpha) * hand_joint_values[t - 1, j_idx_val]
                )

    # Write hand joints into robot_th and enforce mimic
    with torch.no_grad():
        hand_th_tensor = torch.tensor(hand_joint_values, dtype=torch.float32, device=DEVICE)
        for side in ['L', 'R']:
            for j_idx_val in hand_indep[side]:
                robot_th.data[:, j_idx_val] = hand_th_tensor[:, j_idx_val]
        robot_th.data = enforce_mimic(robot_th.data, mimic_map)

    # Print sample values
    for side in ['L', 'R']:
        vals = {joint_names[j]: robot_th.data[0, j].item() for j in hand_indep[side]}
        print(f"  {side} frame 0: " + "  ".join(f"{k.split('_',1)[1][:12]}={v:.3f}" for k, v in vals.items()))

    # =====================================================================
    # Stage 4: Contact-aware hand refinement
    # =====================================================================
    print("=" * 60)
    print("Stage 4: Contact-aware hand refinement")
    print("=" * 60)

    # Build SMPLX contact mask: (T, 5) per side — True when any fingertip should contact
    smplx_contacts = seq_data["human"]["contacts"]  # (T, 52) in SMPLH order
    smplx_tip_indices = {
        "L": [SMPLH_BONE_ORDER_NAMES.index(n) for n in
              ["L_Thumb3", "L_Index3", "L_Middle3", "L_Ring3", "L_Pinky3"]],
        "R": [SMPLH_BONE_ORDER_NAMES.index(n) for n in
              ["R_Thumb3", "R_Index3", "R_Middle3", "R_Ring3", "R_Pinky3"]],
    }
    # Per-side: (T,) — True if ANY fingertip has contact in that frame
    contact_mask = {}
    for side in ["L", "R"]:
        side_contacts = smplx_contacts[:, smplx_tip_indices[side]]  # (T, 5)
        contact_mask[side] = torch.tensor(
            np.any(side_contacts > 0, axis=1), dtype=torch.float32, device=DEVICE,
        )  # (T,)
        n_active = int(contact_mask[side].sum().item())
        print(f"  {side} hand: {n_active}/{T} frames with SMPLX fingertip contact")

    # G1 tip link names per side (same order: thumb, index, middle, ring, pinky)
    g1_tip_links = {
        "L": ["L_thumb_tip", "L_index_tip", "L_middle_tip", "L_ring_tip", "L_pinky_tip"],
        "R": ["R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip"],
    }

    # Arm joint indices per side (7 arm + 6 independent finger = 13 optimizable per side)
    arm_joint_indices = {
        "L": [i for i, n in enumerate(joint_names)
              if n.startswith("left_shoulder") or n.startswith("left_elbow") or n.startswith("left_wrist")],
        "R": [i for i, n in enumerate(joint_names)
              if n.startswith("right_shoulder") or n.startswith("right_elbow") or n.startswith("right_wrist")],
    }

    # Object mesh as torch tensors
    obj_verts_t = torch.tensor(obj_verts_world, dtype=torch.float32, device=DEVICE)  # (T, V, 3)
    obj_normals_t = torch.tensor(obj_normals_world, dtype=torch.float32, device=DEVICE)  # (T, V, 3)

    for side in ["L", "R"]:
        if contact_mask[side].sum() == 0:
            print(f"  {side} hand: no contact frames, skipping")
            continue

        # Indices of joints to optimize: arm (7) + independent finger (6) = 13
        opt_indices = arm_joint_indices[side] + hand_indep[side]

        # Save Stage 1-3 values as prior targets
        prior_values = robot_th.data[:, opt_indices].detach().clone()

        # Create optimizable parameter for this side's joints
        side_params = torch.nn.Parameter(robot_th.data[:, opt_indices].detach().clone())
        opt4 = torch.optim.Adam([side_params], lr=0.005)

        mask = contact_mask[side]  # (T,)
        tip_links = g1_tip_links[side]

        # Precompute smoothed target positions from Stage 3 pose
        with torch.no_grad():
            init_th = enforce_mimic(robot_th.data.clone(), mimic_map)
            init_body = chain.forward_kinematics(init_th, indices)
            init_tip_local = torch.stack(
                [init_body[n].get_matrix()[:, :3, 3] for n in tip_links], dim=1,
            )
            init_tip_world = to_world(init_tip_local, robot_trans.data, robot_rotmat)  # (T, 5, 3)
            # Find nearest vertex per tip per frame
            init_diff = init_tip_world.unsqueeze(2) - obj_verts_t.unsqueeze(1)  # (T, 5, V, 3)
            init_dist_sq = (init_diff ** 2).sum(-1)  # (T, 5, V)
            _, anchor_idx = init_dist_sq.min(dim=2)  # (T, 5)
            anchor_gather = anchor_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
            # Gather target positions and normals
            target_pos = torch.gather(
                obj_verts_t.unsqueeze(1).expand(-1, 5, -1, -1), 2, anchor_gather,
            ).squeeze(2)  # (T, 5, 3)
            target_nrm = torch.gather(
                obj_normals_t.unsqueeze(1).expand(-1, 5, -1, -1), 2, anchor_gather,
            ).squeeze(2)  # (T, 5, 3)
            # Temporal smoothing (bidirectional EMA) to eliminate frame-to-frame jumps
            alpha = 0.3
            for t in range(1, T):
                target_pos[t] = alpha * target_pos[t] + (1 - alpha) * target_pos[t - 1]
                target_nrm[t] = alpha * target_nrm[t] + (1 - alpha) * target_nrm[t - 1]
            for t in range(T - 2, -1, -1):
                target_pos[t] = alpha * target_pos[t] + (1 - alpha) * target_pos[t + 1]
                target_nrm[t] = alpha * target_nrm[t] + (1 - alpha) * target_nrm[t + 1]
            target_nrm = torch.nn.functional.normalize(target_nrm, dim=-1)

        for i in range(200):
            opt4.zero_grad()

            # Assemble full joint tensor with current side params
            full_th = robot_th.data.detach().clone()
            full_th[:, opt_indices] = side_params
            full_th = enforce_mimic(full_th, mimic_map)

            # FK → tip positions in world frame
            body_pos = chain.forward_kinematics(full_th, indices)
            tip_local = torch.stack(
                [body_pos[n].get_matrix()[:, :3, 3] for n in tip_links], dim=1,
            )  # (T, 5, 3)
            tip_world = to_world(tip_local, robot_trans.data, robot_rotmat)  # (T, 5, 3)

            # Distance from each tip to its smoothed target point on the object
            diff_vec = tip_world - target_pos  # (T, 5, 3)
            unsigned_dist = diff_vec.norm(dim=-1)  # (T, 5)
            signed_dist = (diff_vec * target_nrm).sum(-1)  # (T, 5)

            # Contact attraction: pull tips toward anchored surface point
            loss_attract = torch.mean(unsigned_dist * mask.unsqueeze(1)) * 50.0

            # Collision penalty: extra penalty for penetration (signed_dist < 0)
            penetration = torch.clamp(-signed_dist, min=0)  # (T, 5)
            loss_collide = torch.mean(penetration * mask.unsqueeze(1)) * 100.0

            # Prior: stay close to Stage 1-3 solution
            # Arm joints (first 7) get tight prior, finger joints (last 6) get loose prior
            arm_prior = torch.mean((side_params[:, :7] - prior_values[:, :7]) ** 2) * 5.0
            finger_prior = torch.mean((side_params[:, 7:] - prior_values[:, 7:]) ** 2) * 0.5

            # Joint limits
            jl = joint_limit_loss(
                side_params,
                joint_lower[opt_indices],
                joint_upper[opt_indices],
            ) * 10.0

            # Temporal smoothness
            smooth = torch.mean((side_params[1:] - side_params[:-1]) ** 2) * 0.5

            loss = loss_attract + loss_collide + arm_prior + finger_prior + jl + smooth
            loss.backward()
            opt4.step()

            if i % 50 == 0:
                print(f"  {side} iter {i:3d}  loss={loss.item():.4f}  "
                      f"attract={loss_attract.item():.4f}  collide={loss_collide.item():.4f}  "
                      f"arm_prior={arm_prior.item():.4f}  finger_prior={finger_prior.item():.4f}")

        # Write back
        with torch.no_grad():
            robot_th.data[:, opt_indices] = side_params.data
            robot_th.data = enforce_mimic(robot_th.data, mimic_map)

    # =====================================================================
    # Compute contact labels
    # =====================================================================
    print("=" * 60)
    print("Computing contact labels")
    print("=" * 60)
    contact_label, contact_distance, link_names = compute_contact_labels(
        chain, robot_th.data, robot_trans.data, robot_rotmat, obj_verts_world,
    )

    return {
        "root_trans": robot_trans.detach().cpu().numpy(),
        "root_quat": sRot.from_matrix(
            robot_rotmat.detach().cpu().numpy()
        ).as_quat(scalar_first=True),
        "joint_pos": robot_th.detach().cpu().numpy(),
        "object": seq_data["object"],
        "contact_label": contact_label,
        "contact_distance": contact_distance,
        "link_names": link_names,
    }


# =============================================================================
# Visualization
# =============================================================================

def visualize_retarget(results: dict, seq_data: dict, chain: pk.Chain, fps: int = 30):
    import viser
    import viser.transforms as vtf
    from viser.extras import ViserUrdf

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=4.0, height=4.0, cell_size=0.1)

    root_trans, root_quat, joint_pos = results["root_trans"], results["root_quat"], results["joint_pos"]
    T = root_trans.shape[0]

    obj_name = seq_data['object']['name']
    obj_trans = seq_data['object']['trans']
    obj_rot = seq_data['object']['rot']

    obj_base = server.scene.add_frame("/object_frame", show_axes=True)
    mesh_path = os.path.join(OBJECTS_PATH, f"{obj_name}.obj")
    mesh_obj = trimesh.load(mesh_path, force='mesh')
    server.scene.add_mesh_simple(
        "/object_frame/object",
        vertices=mesh_obj.vertices,
        faces=mesh_obj.faces,
        color=(0.2, 0.8, 0.2),
    )

    robot_base = server.scene.add_frame("/robot_base", show_axes=True)
    urdf = yourdfpy.URDF.load(G1_URDF_PATH)
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot_base")

    pk_joint_names = chain.get_joint_parameter_names()

    # --- GUI controls ---
    with server.gui.add_folder("Playback"):
        gui_playing = server.gui.add_checkbox("Playing", initial_value=True)
        gui_frame = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=0)
        gui_fps = server.gui.add_slider("FPS", min=1, max=120, step=1, initial_value=fps)
        gui_loop = server.gui.add_checkbox("Loop", initial_value=True)

    with server.gui.add_folder("Record"):
        gui_record_video = server.gui.add_button("Record Video")

    def _update_frame(t):
        robot_base.position = root_trans[t]
        robot_base.wxyz = root_quat[t]
        cfg = {name: joint_pos[t, i] for i, name in enumerate(pk_joint_names)}
        viser_urdf.update_cfg(cfg)
        obj_base.position = obj_trans[t]
        obj_base.wxyz = vtf.SO3.from_matrix(obj_rot[t]).wxyz

    @gui_record_video.on_click
    def _(_):
        import imageio
        clients = server.get_clients()
        if not clients:
            print("  No client connected!")
            return
        _, client = next(iter(clients.items()))
        was_playing = gui_playing.value
        gui_playing.value = False

        frames = []
        print(f"  Recording {T} frames...")
        for t in range(T):
            gui_frame.value = t
            _update_frame(t)
            time.sleep(0.05)
            img = client.camera.get_render(height=1080, width=1920)
            frames.append(img)

        video_path = os.path.join(OUTPUT_PATH, "retarget_video.mp4")
        imageio.mimwrite(video_path, frames, fps=fps, quality=8)
        print(f"  Saved video: {video_path}")
        gui_playing.value = was_playing

    @gui_frame.on_update
    def _(_):
        if not gui_playing.value:
            _update_frame(gui_frame.value)

    print("Playing visualization... (open browser at http://localhost:8012)")
    print("  Controls: Playing/pause, Frame slider, Record Video button")

    t = 0
    while True:
        if gui_playing.value:
            _update_frame(t)
            gui_frame.value = t
            t = (t + 1) % T if gui_loop.value else min(t + 1, T - 1)
            time.sleep(1.0 / gui_fps.value)
        else:
            time.sleep(0.05)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    chain = build_chain(G1_URDF_PATH)

    sequences_path = os.path.join(OUTPUT_PATH, 'example.pkl')
    sequences = joblib.load(sequences_path)

    for object_name, object_seqs in sequences.items():
        for seq_name, seq_data in object_seqs.items():
            print(f"\nRetargeting {seq_name} with {object_name}")
            print("=" * 60)

            results = run_retarget(chain, seq_data)

            save_path = os.path.join(OUTPUT_PATH, f"{seq_name}_retargeted.pkl")
            joblib.dump(results, save_path)
            print(f"\nSaved to {save_path}")

            if args.visualize:
                visualize_retarget(results, seq_data, chain)

if __name__ == "__main__":
    main()
