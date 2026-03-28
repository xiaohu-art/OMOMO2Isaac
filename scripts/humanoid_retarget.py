import os
import time
import torch
import joblib
import trimesh
import yourdfpy
import numpy as np
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
from collections import OrderedDict
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
    "R_hand_base_link": "R_Wrist"
}


def parse_joint_limits(urdf_path: str, joint_names: list) -> tuple:
    """Parse joint limits from URDF, returning (lower, upper) tensors ordered by joint_names."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    limits = {}
    for joint_el in root.findall('joint'):
        limit_el = joint_el.find('limit')
        if limit_el is not None:
            limits[joint_el.get('name')] = (
                float(limit_el.get('lower', 0)),
                float(limit_el.get('upper', 0)),
            )
    lower = torch.tensor([limits[n][0] for n in joint_names], dtype=torch.float32, device=DEVICE)
    upper = torch.tensor([limits[n][1] for n in joint_names], dtype=torch.float32, device=DEVICE)
    return lower, upper

def joint_limit_loss(th: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Penalize joint angles outside [lower, upper]. th shape: (T, N_joints)."""
    below = torch.clamp(lower - th, min=0)
    above = torch.clamp(th - upper, min=0)
    return torch.mean(below ** 2 + above ** 2)

def build_chain(urdf_path: str) -> pk.Chain:
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()

    chain = pk.build_chain_from_urdf(urdf_content)
    return chain.to(device=DEVICE)

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

    robot_link_names = list(JOINT_MAP.keys())
    smpl_joint_names = list(JOINT_MAP.values())
    smpl_indices = [SMPLH_BONE_ORDER_NAMES.index(n) for n in smpl_joint_names]
    smpl_keypoints = keypoints[:, smpl_indices, :]

    # Parse joint limits from URDF
    joint_names = chain.get_joint_parameter_names()
    joint_lower, joint_upper = parse_joint_limits(G1_URDF_PATH, joint_names)

    # Initialize optimization variables
    robot_th = torch.nn.Parameter(torch.zeros(T, chain.n_joints).to(DEVICE))
    robot_trans = torch.nn.Parameter(torch.from_numpy(trans).float().to(DEVICE))

    indices = chain.get_all_frame_indices()

    def stack_positions(body_pos, names):
        return torch.stack(
            [body_pos[name].get_matrix()[:, :3, 3] for name in names], dim=1
        )

    def to_world(pos_local, trans, rotmat):
        pos_w = rotmat.unsqueeze(1) @ pos_local.unsqueeze(-1)
        return pos_w.squeeze(-1) + trans.unsqueeze(1)

    def get_robot_keypoints(th: torch.Tensor, trans: torch.Tensor, rotmat: torch.Tensor):
        body_pos = chain.forward_kinematics(th, indices)
        local = stack_positions(body_pos, robot_link_names)
        return to_world(local, trans, rotmat)

    # ---- Stage 1: Normal IK, no ground correction ----
    opt = torch.optim.Adam([robot_th, robot_trans], lr=0.02)
    for i in range(300):
        opt.zero_grad()

        robot_kp_w = get_robot_keypoints(robot_th, robot_trans, robot_rotmat)
        omega = torch.gradient(robot_th, spacing=1.0 / fps, dim=0)[0]

        keypoints_pos_error = torch.mean((robot_kp_w - smpl_keypoints) ** 2)
        joint_pos_reg = torch.mean(torch.square(robot_th))
        joint_vel_reg = torch.mean(torch.square(omega))
        jl_loss = joint_limit_loss(robot_th, joint_lower, joint_upper)
        loss = keypoints_pos_error + 2e-2 * joint_pos_reg + 1e-3 * joint_vel_reg + 10.0 * jl_loss

        loss.backward()
        opt.step()

        if i % 50 == 0:
            print(
                f"[Stage1] iter {i}, loss {loss.item():.6f}, "
                f"kp {keypoints_pos_error.item():.6f}, "
                f"j_pos {joint_pos_reg.item():.6f}, "
                f"j_vel {joint_vel_reg.item():.6f}, "
                f"j_lim {jl_loss.item():.6f}, "
            )

    # ---- Stage 2: Freeze lower body, optimize root Z + upper body for ground alignment ----
    G1_ANKLE_TO_SOLE = 0.035  # collision spheres at z=-0.03, radius=0.005
    LOWER_BODY_JOINTS = 12  # indices 0-11: hip/knee/ankle joints
    l_hand_idx = robot_link_names.index("L_hand_base_link")
    r_hand_idx = robot_link_names.index("R_hand_base_link")
    l_ankle_robot_idx = robot_link_names.index("left_ankle_roll_link")
    r_ankle_robot_idx = robot_link_names.index("right_ankle_roll_link")

    # Detect ground contact frames from SMPLX toe keypoints (ground = Z=0)
    GROUND_THRESH = 0.08
    l_toe_smpl_idx = SMPLH_BONE_ORDER_NAMES.index("L_Toe")
    r_toe_smpl_idx = SMPLH_BONE_ORDER_NAMES.index("R_Toe")
    l_contact = (keypoints[:, l_toe_smpl_idx, 2] < GROUND_THRESH).float()  # (T,)
    r_contact = (keypoints[:, r_toe_smpl_idx, 2] < GROUND_THRESH).float()  # (T,)

    # Record hand positions from stage 1 as targets
    with torch.no_grad():
        stage1_kp = get_robot_keypoints(robot_th, robot_trans, robot_rotmat)
        ee_targets = stage1_kp.detach().clone()

    # Split joint parameters: freeze lower body, optimize upper body
    lower_th = robot_th.data[:, :LOWER_BODY_JOINTS].detach().clone()  # frozen
    upper_th = torch.nn.Parameter(robot_th.data[:, LOWER_BODY_JOINTS:].detach().clone())
    root_z_offset = torch.nn.Parameter(torch.zeros(T, 1, device=DEVICE))
    opt2 = torch.optim.Adam([upper_th, root_z_offset], lr=0.01)

    for i in range(200):
        opt2.zero_grad()

        # Reassemble full joint angles: frozen lower + optimized upper
        full_th = torch.cat([lower_th, upper_th], dim=1)

        # Apply Z offset to root translation
        adjusted_trans = robot_trans.detach().clone()
        adjusted_trans[:, 2:3] += root_z_offset

        robot_kp_w = get_robot_keypoints(full_th, adjusted_trans, robot_rotmat)

        # End-effector preservation: hands stay at stage 1 positions
        ee_loss = torch.mean((robot_kp_w[:, l_hand_idx, :] - ee_targets[:, l_hand_idx, :]) ** 2) \
                + torch.mean((robot_kp_w[:, r_hand_idx, :] - ee_targets[:, r_hand_idx, :]) ** 2)

        # Ground alignment: only when SMPLX foot is in ground contact
        l_sole_z = robot_kp_w[:, l_ankle_robot_idx, 2] - G1_ANKLE_TO_SOLE
        r_sole_z = robot_kp_w[:, r_ankle_robot_idx, 2] - G1_ANKLE_TO_SOLE
        ground_loss = torch.mean(l_contact * l_sole_z ** 2) \
                    + torch.mean(r_contact * r_sole_z ** 2)

        omega = torch.gradient(upper_th, spacing=1.0 / fps, dim=0)[0]
        joint_pos_reg = torch.mean(torch.square(upper_th))
        joint_vel_reg = torch.mean(torch.square(omega))
        jl_loss = joint_limit_loss(upper_th, joint_lower[LOWER_BODY_JOINTS:], joint_upper[LOWER_BODY_JOINTS:])

        loss = 10.0 * ee_loss + 5.0 * ground_loss \
             + 2e-2 * joint_pos_reg + 1e-3 * joint_vel_reg + 10.0 * jl_loss

        loss.backward()
        opt2.step()

        if i % 50 == 0:
            print(
                f"[Stage2] iter {i}, loss {loss.item():.6f}, "
                f"ee {ee_loss.item():.6f}, "
                f"ground {ground_loss.item():.6f}, "
                f"j_pos {joint_pos_reg.item():.6f}, "
                f"j_vel {joint_vel_reg.item():.6f}, "
                f"j_lim {jl_loss.item():.6f}, "
            )

    # Merge results back
    with torch.no_grad():
        robot_th.data = torch.cat([lower_th, upper_th.data], dim=1)
        robot_trans.data[:, 2:3] += root_z_offset.data

    return {
        "root_trans": robot_trans.detach().cpu().numpy(),
        "root_quat": sRot.from_matrix(
                        robot_rotmat.detach().cpu().numpy()
                    ).as_quat(scalar_first=True),
        "joint_pos": robot_th.detach().cpu().numpy(),
        "object": seq_data["object"],
    }

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

    print("Playing visualization...")
    while True:
        for t in range(T):
            robot_base.position = root_trans[t]
            robot_base.wxyz = root_quat[t]
            
            cfg = {name: joint_pos[t, i] for i, name in enumerate(pk_joint_names)}
            viser_urdf.update_cfg(cfg)

            obj_base.position = obj_trans[t]
            obj_base.wxyz = vtf.SO3.from_matrix(obj_rot[t]).wxyz

            time.sleep(1.0 / fps)

def main():
    args = parse_args()

    chain = build_chain(G1_URDF_PATH)

    # sequences_path = os.path.join(OUTPUT_PATH, f'{args.flag}_sequences.pkl')
    sequences_path = os.path.join(OUTPUT_PATH, 'example.pkl')
    sequences = joblib.load(sequences_path)

    for object_name, object_seqs in sequences.items():
        for seq_name, seq_data in object_seqs.items():
            print(f"Retargeting {seq_name} with {object_name}")

            results = run_retarget(chain, seq_data)

            save_path = os.path.join(OUTPUT_PATH, f"{seq_name}_retargeted.pkl")
            joblib.dump(results, save_path)

            if args.visualize:
                visualize_retarget(results, seq_data, chain)

if __name__ == "__main__":
    main()