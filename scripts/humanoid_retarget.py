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



# =============================================================================
# Three-stage retargeting
# =============================================================================

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

    return {
        "root_trans": robot_trans.detach().cpu().numpy(),
        "root_quat": sRot.from_matrix(
            robot_rotmat.detach().cpu().numpy()
        ).as_quat(scalar_first=True),
        "joint_pos": robot_th.detach().cpu().numpy(),
        "object": seq_data["object"],
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
