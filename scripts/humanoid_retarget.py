"""
Two-stage humanoid retargeting:

Stage 1: GMR-style full body IK + ground contact (per-link weighted pos/ori
         loss with axis mask, plus ankle-to-ground loss; pelvis xy anchored,
         z free so the IK lowers G1 to plant feet).
Stage 2: Contact-aware hand refinement — directly optimizes arm + finger DOFs
         to satisfy fingertip contact targets on the object mesh.
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
from collections import namedtuple
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
from utils.math import quat_from_angle_axis, matrix_from_quat, quat_fk
from utils.process import get_smpl_parents

G1_URDF_PATH = os.path.join(G1_PATH, "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf")

IKEntry = namedtuple("IKEntry", "smplh pos_w ori_w rot_off pos_axes")
IKEntry.__new__.__defaults__ = ((1.0, 1.0, 1.0),)

IK_TARGETS = {
    # G1 link                  IKEntry( smplh_joint   pos_w  ori_w  rot_offset_wxyz [, pos_axes] )
    # -- Body links (rot_offset byte-for-byte from GMR ik_match_table1) --
    # pelvis: pos_axes=(1,1,0) frees z so IK can lower body to plant feet.
    "pelvis":                  IKEntry("Pelvis",     100, 10, ( 0.5, -0.5, -0.5, -0.5), (1.0, 1.0, 0.0)),
    "left_hip_roll_link":      IKEntry("L_Hip",        0, 10, ( 0.4267755048530407, -0.5637931078484661, -0.5637931078484661, -0.4267755048530407)),
    "left_knee_link":          IKEntry("L_Knee",       0, 10, ( 0.5, -0.5, -0.5, -0.5)),
    "right_hip_roll_link":     IKEntry("R_Hip",        0, 10, ( 0.4267755048530407, -0.5637931078484661, -0.5637931078484661, -0.4267755048530407)),
    "right_knee_link":         IKEntry("R_Knee",       0, 10, ( 0.5, -0.5, -0.5, -0.5)),
    "torso_link":              IKEntry("Chest",        0, 10, ( 0.5, -0.5, -0.5, -0.5)),
    "left_shoulder_yaw_link":  IKEntry("L_Shoulder",   0, 10, ( 0.70710678, 0.0, -0.70710678, 0.0)),
    "left_elbow_link":         IKEntry("L_Elbow",      0, 10, ( 1.0, 0.0, 0.0, 0.0)),
    "right_shoulder_yaw_link": IKEntry("R_Shoulder",   0, 10, ( 0.0, 0.70710678, 0.0, 0.70710678)),
    "right_elbow_link":        IKEntry("R_Elbow",      0, 10, ( 0.0, 0.0, 0.0, -1.0)),
    # -- HOI wrist anchor: rot_offset = gmr(wrist_yaw) ⊗ R(wrist_yaw -> hand_base) --
    # "L_hand_base_link":        IKEntry("L_Wrist",    100, 10, ( 0.70710678,  0.0,         0.0,         0.70710678)),
    # "R_hand_base_link":        IKEntry("R_Wrist",    100, 10, ( 0.0,        -0.70710678, -0.70710678,  0.0)),
    # Alternative (ablation): anchor wrist_yaw_link instead of hand_base_link.
    "left_wrist_yaw_link":   IKEntry("L_Wrist",    100, 10, ( 1.0, 0.0, 0.0, 0.0)),
    "right_wrist_yaw_link":  IKEntry("R_Wrist",    100, 10, ( 0.0, 0.0, 0.0, -1.0)),
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


def compute_smplx_world_rotations(poses_aa: np.ndarray, gender: str = "neutral") -> np.ndarray:
    """Compute world rotation matrices for SMPL-X body joints (0-21) via FK on
    local axis-angle along the SMPL-X kinematic tree."""
    body_aa = np.asarray(poses_aa[:, :22, :], dtype=np.float32)
    local_quats = quat_from_angle_axis(body_aa)  # (T, 22, 4) wxyz
    parents = get_smpl_parents(gender).tolist()
    chain_quats, _ = quat_fk(local_quats, np.zeros_like(body_aa), parents)
    return matrix_from_quat(chain_quats)  # (T, 22, 3, 3)


def build_ik_targets(table: dict):
    """Resolve `IK_TARGETS` (`IKEntry` values) into runtime-ready arrays.

    Returns:
        targets: list of dicts {robot_link, smplh_idx, pos_weight, ori_weight,
                 rot_offset_R (3, 3)}.
        pelvis_offset_R: (3, 3) — replaces R_canon at the root so the pelvis
                 IK target (smplx_pelvis_world @ pelvis_offset) is identically
                 satisfied at every frame.
    """
    targets = []
    for robot_link, e in table.items():
        rot_R = sRot.from_quat(np.asarray(e.rot_off, dtype=np.float64),
                                scalar_first=True).as_matrix()
        targets.append({
            "robot_link": robot_link,
            "smplh_idx": SMPLH_BONE_ORDER_NAMES.index(e.smplh),
            "pos_weight": float(e.pos_w),
            "ori_weight": float(e.ori_w),
            "rot_offset_R": rot_R.astype(np.float32),
            "pos_axes": np.asarray(e.pos_axes, dtype=np.float32),
        })
    pelvis_quat = np.asarray(table["pelvis"].rot_off, dtype=np.float64)
    pelvis_R = sRot.from_quat(pelvis_quat, scalar_first=True).as_matrix().astype(np.float32)
    return targets, pelvis_R


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
    gender = str(seq_data["human"].get("gender", "neutral"))

    ik_targets, pelvis_offset_R = build_ik_targets(IK_TARGETS)
    pelvis_offset_R_t = torch.tensor(pelvis_offset_R, dtype=torch.float32, device=DEVICE)
    R_root = torch.tensor(
        sRot.from_rotvec(root_orient).as_matrix(), dtype=torch.float32, device=DEVICE,
    )  # (T, 3, 3)
    robot_rotmat = R_root @ pelvis_offset_R_t  # (T, 3, 3)

    # Load object mesh (world-space vertices + normals per frame)
    obj_verts_world, obj_normals_world, _ = load_object_mesh(seq_data)
    print(f"Object mesh loaded: {obj_verts_world.shape[1]} vertices, {T} frames")

    # Body link targets
    robot_link_names = [t["robot_link"] for t in ik_targets]
    smpl_indices = [t["smplh_idx"] for t in ik_targets]
    smpl_keypoints = keypoints[:, smpl_indices, :]  # (T, n, 3) — pos targets
    pos_weights = torch.tensor([t["pos_weight"] for t in ik_targets],
                                dtype=torch.float32, device=DEVICE)
    ori_weights = torch.tensor([t["ori_weight"] for t in ik_targets],
                                dtype=torch.float32, device=DEVICE)
    pos_axes_mask = torch.tensor(np.stack([t["pos_axes"] for t in ik_targets]),
                                  dtype=torch.float32, device=DEVICE)  # (n, 3)
    rot_offsets_R = torch.tensor(np.stack([t["rot_offset_R"] for t in ik_targets]),
                                  dtype=torch.float32, device=DEVICE)  # (n, 3, 3)

    # Orientation targets: target_world_R[t, n] = smplx_world_R[t, n] @ rot_offset[n]
    smplx_world_R_np = compute_smplx_world_rotations(poses, gender)  # (T, 22, 3, 3)
    smplx_world_R = torch.tensor(smplx_world_R_np, dtype=torch.float32, device=DEVICE)
    smplx_target_R = smplx_world_R[:, smpl_indices]  # (T, n, 3, 3)
    target_world_R = torch.einsum("tnij,njk->tnik", smplx_target_R, rot_offsets_R)

    # Ground contact: pull G1 ankle down on frames where SMPL-H toe is near ground.
    foot_link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    G1_ANKLE_TO_SOLE = 0.035
    GROUND_THRESH = 0.08
    GROUND_W = 100.0
    l_toe_idx = SMPLH_BONE_ORDER_NAMES.index("L_Toe")
    r_toe_idx = SMPLH_BONE_ORDER_NAMES.index("R_Toe")
    l_contact = (keypoints[:, l_toe_idx, 2] < GROUND_THRESH).float()
    r_contact = (keypoints[:, r_toe_idx, 2] < GROUND_THRESH).float()

    # Joint config
    joint_names = chain.get_joint_parameter_names()
    joint_lower, joint_upper = parse_joint_limits(G1_URDF_PATH, joint_names)
    mimic_map = parse_mimic_joints(G1_URDF_PATH, joint_names)
    hand_indep = get_hand_independent_indices(joint_names, mimic_map)

    # Initialize
    robot_th = torch.nn.Parameter(torch.zeros(T, chain.n_joints, device=DEVICE))
    robot_trans = torch.nn.Parameter(torch.from_numpy(trans).float().to(DEVICE))
    indices = chain.get_all_frame_indices()

    # =====================================================================
    # Stage 1: Full body IK + ground contact
    # =====================================================================
    print("=" * 60)
    print("Stage 1: Full body IK + ground contact")
    pos_active = [
        (n, w, tuple(a)) for n, w, a in zip(
            robot_link_names, pos_weights.tolist(), pos_axes_mask.tolist())
        if w > 0
    ]
    ori_active = [(n, w) for n, w in zip(robot_link_names, ori_weights.tolist()) if w > 0]
    print(f"  pos targets: {pos_active}")
    print(f"  ori targets: {len(ori_active)} links @ weight 10")
    print(f"  ground contact: {int(l_contact.sum())}/{T} L, {int(r_contact.sum())}/{T} R frames")
    print("=" * 60)
    opt = torch.optim.Adam([robot_th, robot_trans], lr=0.02)
    for i in range(300):
        opt.zero_grad()

        body_pos = chain.forward_kinematics(robot_th, indices)
        local_pos = torch.stack(
            [body_pos[name].get_matrix()[:, :3, 3] for name in robot_link_names], dim=1,
        )  # (T, n, 3)
        local_R = torch.stack(
            [body_pos[name].get_matrix()[:, :3, :3] for name in robot_link_names], dim=1,
        )  # (T, n, 3, 3)
        foot_local = torch.stack(
            [body_pos[name].get_matrix()[:, :3, 3] for name in foot_link_names], dim=1,
        )  # (T, 2, 3)
        pos_world = (robot_rotmat.unsqueeze(1) @ local_pos.unsqueeze(-1)).squeeze(-1) \
                    + robot_trans.unsqueeze(1)  # (T, n, 3)
        rot_world = robot_rotmat.unsqueeze(1) @ local_R  # (T, n, 3, 3)
        foot_world = (robot_rotmat.unsqueeze(1) @ foot_local.unsqueeze(-1)).squeeze(-1) \
                     + robot_trans.unsqueeze(1)  # (T, 2, 3)

        # Weighted per-link MSE (entries with weight=0 are inert).
        # pos_axes_mask zeros residual on disabled axes (e.g. pelvis z).
        pos_diff_sq = (pos_world - smpl_keypoints) ** 2 * pos_axes_mask.unsqueeze(0)
        pos_per_link = pos_diff_sq.mean(dim=(0, 2))  # (n,)
        ori_per_link = ((rot_world - target_world_R) ** 2).mean(dim=(0, 2, 3))  # (n,)
        pos_loss = (pos_per_link * pos_weights).sum()
        ori_loss = (ori_per_link * ori_weights).sum()

        l_sole_z = foot_world[:, 0, 2] - G1_ANKLE_TO_SOLE
        r_sole_z = foot_world[:, 1, 2] - G1_ANKLE_TO_SOLE
        ground_loss = (l_contact * l_sole_z ** 2).mean() \
                    + (r_contact * r_sole_z ** 2).mean()

        omega = torch.gradient(robot_th, spacing=1.0 / fps, dim=0)[0]
        vel_reg = torch.mean(torch.square(omega))
        jl_loss = joint_limit_loss(robot_th, joint_lower, joint_upper)

        loss = pos_loss + ori_loss + GROUND_W * ground_loss \
             + 1e-3 * vel_reg + 10.0 * jl_loss

        loss.backward()
        opt.step()

        if i % 50 == 0:
            print(f"  iter {i:3d}  loss={loss.item():.4f}  "
                  f"pos={pos_loss.item():.4f}  ori={ori_loss.item():.4f}  "
                  f"ground={ground_loss.item():.5f}  jl={jl_loss.item():.4f}")

    # =====================================================================
    # Stage 2: Contact-aware hand refinement
    # =====================================================================
    print("=" * 60)
    print("Stage 2: Contact-aware hand refinement")
    print("=" * 60)

    # Loss weights and optimizer config for the per-iter optimization.
    W_ATTRACT, W_COLLIDE = 50.0, 500.0
    W_JL, W_SMOOTH = 1000.0, 0.5
    N_ITER, LR = 300, 0.005

    # Warm-start hand joints at the midpoint of joint range
    with torch.no_grad():
        for side in ["L", "R"]:
            for j_idx in hand_indep[side]:
                robot_th.data[:, j_idx] = 0.5 * (joint_lower[j_idx] + joint_upper[j_idx])
        robot_th.data.copy_(enforce_mimic(robot_th.data, mimic_map))

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

        # Init from Stage 1 pose; no anchor prior — attract/collide drive the
        # arm + finger configuration freely, only constrained by joint limits
        # and temporal smoothness.
        side_params = torch.nn.Parameter(
            robot_th.data[:, opt_indices].detach().clone()
        )
        opt2 = torch.optim.Adam([side_params], lr=LR)

        mask = contact_mask[side]  # (T,)
        contact_denom = mask.sum() * 5 + 1e-8

        tip_links = g1_tip_links[side]
        tip_frame_indices = chain.get_frame_indices(*tip_links)

        # Precompute smoothed target positions from Stage 1 pose
        with torch.no_grad():
            init_th = enforce_mimic(robot_th.data, mimic_map)  # enforce_mimic clones internally
            init_body = chain.forward_kinematics(init_th, tip_frame_indices)
            init_tip_local = torch.stack(
                [init_body[n].get_matrix()[:, :3, 3] for n in tip_links], dim=1,
            )
            init_tip_world = to_world(init_tip_local, robot_trans.data, robot_rotmat)  # (T, 5, 3)
            anchor_idx = torch.cdist(init_tip_world, obj_verts_t).argmin(dim=2)  # (T, 5)
            T_idx = torch.arange(T, device=DEVICE).unsqueeze(1).expand(-1, 5)  # (T, 5)
            target_pos = obj_verts_t[T_idx, anchor_idx]    # (T, 5, 3)
            target_nrm = obj_normals_t[T_idx, anchor_idx]  # (T, 5, 3)
            # Temporal smoothing (bidirectional EMA) to eliminate frame-to-frame jumps
            alpha = 0.3
            for t in range(1, T):
                target_pos[t] = alpha * target_pos[t] + (1 - alpha) * target_pos[t - 1]
            for t in range(T - 2, -1, -1):
                target_pos[t] = alpha * target_pos[t] + (1 - alpha) * target_pos[t + 1]
            anchor_idx = torch.cdist(target_pos, obj_verts_t).argmin(dim=2)  # (T, 5)
            target_pos = obj_verts_t[T_idx, anchor_idx]
            target_nrm = obj_normals_t[T_idx, anchor_idx]

        for i in range(N_ITER):
            opt2.zero_grad()

            # Assemble full joint tensor with current side params
            full_th = robot_th.data.detach().clone()
            full_th[:, opt_indices] = side_params
            full_th = enforce_mimic(full_th, mimic_map)

            body_pos = chain.forward_kinematics(full_th, tip_frame_indices)
            tip_local = torch.stack(
                [body_pos[n].get_matrix()[:, :3, 3] for n in tip_links], dim=1,
            )  # (T, 5, 3)
            tip_world = to_world(tip_local, robot_trans.data, robot_rotmat)  # (T, 5, 3)

            # Distance from each tip to its smoothed target point on the object
            diff_vec = tip_world - target_pos  # (T, 5, 3)
            unsigned_dist = diff_vec.norm(dim=-1)  # (T, 5)
            signed_dist = (diff_vec * target_nrm).sum(-1)  # (T, 5)

            loss_attract = (unsigned_dist * mask.unsqueeze(1)).sum() / contact_denom * W_ATTRACT

            # Collision penalty: extra penalty for penetration (signed_dist < 0)
            penetration = torch.clamp(-signed_dist, min=0)  # (T, 5)
            loss_collide = (penetration * mask.unsqueeze(1)).sum() / contact_denom * W_COLLIDE

            # Joint limits (soft penalty; W_JL must be large enough to dominate
            # attract gradient near the boundary, otherwise fingers reverse-bend).
            jl = joint_limit_loss(
                side_params,
                joint_lower[opt_indices],
                joint_upper[opt_indices],
            ) * W_JL

            # Temporal smoothness
            smooth = torch.mean((side_params[1:] - side_params[:-1]) ** 2) * W_SMOOTH

            loss = loss_attract + loss_collide + jl + smooth
            loss.backward()
            opt2.step()

            if i % 50 == 0:
                print(f"  {side} iter {i:3d}  loss={loss.item():.4f}  "
                      f"attract={loss_attract.item():.4f}  collide={loss_collide.item():.4f}  "
                      f"jl={jl.item():.4f}  smooth={smooth.item():.4f}")

        # Write back
        with torch.no_grad():
            robot_th.data[:, opt_indices] = side_params.data
            robot_th.data.copy_(enforce_mimic(robot_th.data, mimic_map))

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

    sequences_path = os.path.join(OUTPUT_PATH, f'{args.flag}_sequences.pkl')
    sequences = joblib.load(sequences_path)

    retarget_root = os.path.join(OUTPUT_PATH, f'{args.flag}_retargeted')
    os.makedirs(retarget_root, exist_ok=True)

    for object_name, object_seqs in sequences.items():
        for seq_name, seq_data in object_seqs.items():
            out_dir = os.path.join(retarget_root, object_name)
            out_file = os.path.join(out_dir, f"{seq_name}.pkl")
            if os.path.exists(out_file):
                continue

            print(f"\nRetargeting {seq_name} with {object_name}")
            print("=" * 60)

            results = run_retarget(chain, seq_data)

            os.makedirs(out_dir, exist_ok=True)
            joblib.dump(results, out_file)
            print(f"\nSaved to {out_file}")

            if args.visualize:
                visualize_retarget(results, seq_data, chain)

    merged = {}
    for object_name in sorted(os.listdir(retarget_root)):
        obj_dir = os.path.join(retarget_root, object_name)
        if not os.path.isdir(obj_dir):
            continue
        merged[object_name] = {}
        for fname in sorted(os.listdir(obj_dir)):
            if not fname.endswith('.pkl'):
                continue
            seq_name = os.path.splitext(fname)[0]
            merged[object_name][seq_name] = joblib.load(os.path.join(obj_dir, fname))

    merged_path = os.path.join(OUTPUT_PATH, f'{args.flag}_retargeted.pkl')
    joblib.dump(merged, merged_path)
    total = sum(len(v) for v in merged.values())
    print(f"\nMerged {total} retargeted sequences across {len(merged)} objects -> {merged_path}")

if __name__ == "__main__":
    main()
