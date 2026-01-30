import os
import time
import torch
import joblib
import trimesh
import yourdfpy
import numpy as np
import pytorch_kinematics as pk
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

# Joint indices will be populated after chain is built
BODY_JOINT_INDICES = None
LEFT_HAND_JOINT_INDICES = None
RIGHT_HAND_JOINT_INDICES = None

BODY_JOINT_MAP = {
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

HAND_TIP_MAP = {
    # Left Hand
    "L_thumb_tip": "L_Thumb3",
    "L_index_tip": "L_Index3",
    "L_middle_tip": "L_Middle3",
    "L_ring_tip": "L_Ring3",
    "L_pinky_tip": "L_Pinky3",
    # Right Hand
    "R_thumb_tip": "R_Thumb3",
    "R_index_tip": "R_Index3",
    "R_middle_tip": "R_Middle3",
    "R_ring_tip": "R_Ring3",
    "R_pinky_tip": "R_Pinky3",
}

END_EFFECTOR_WEIGHTS = {
    "L_hand_base_link": 5.0,
    "R_hand_base_link": 5.0,
    "left_ankle_roll_link": 5.0,
    "right_ankle_roll_link": 5.0,
}

def get_joint_indices(chain: pk.Chain):
    """
    Get body, left hand, and right hand joint indices from chain.
    
    Returns:
        body_joint_indices: List of body joint indices (legs, waist, arms to wrist)
        left_hand_joint_indices: List of left hand joint indices
        right_hand_joint_indices: List of right hand joint indices
    """
    joint_names = chain.get_joint_parameter_names()
    
    body_indices = []
    left_hand_indices = []
    right_hand_indices = []
    
    for i, name in enumerate(joint_names):
        if name.startswith("L_") and ("thumb" in name or "index" in name or "middle" in name or "ring" in name or "pinky" in name):
            left_hand_indices.append(i)
        elif name.startswith("R_") and ("thumb" in name or "index" in name or "middle" in name or "ring" in name or "pinky" in name):
            right_hand_indices.append(i)
        else:
            body_indices.append(i)
    
    return body_indices, left_hand_indices, right_hand_indices

def build_chain(urdf_path: str) -> pk.Chain:
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()

    chain = pk.build_chain_from_urdf(urdf_content)
    chain = chain.to(device=DEVICE)
    
    body_indices, left_hand_indices, right_hand_indices = get_joint_indices(chain)
    body_indices = torch.tensor(body_indices, dtype=torch.long, device=DEVICE)
    left_hand_indices = torch.tensor(left_hand_indices, dtype=torch.long, device=DEVICE)
    right_hand_indices = torch.tensor(right_hand_indices, dtype=torch.long, device=DEVICE)
    
    return chain, body_indices, left_hand_indices, right_hand_indices

def combine_configs(
    body_th: torch.Tensor, l_hand_th: torch.Tensor, r_hand_th: torch.Tensor, 
    body_indices: torch.Tensor, left_hand_indices: torch.Tensor, right_hand_indices: torch.Tensor, 
    total_dof: int
):
    B = body_th.shape[0]
    th = torch.zeros((B, total_dof), device=body_th.device, dtype=body_th.dtype)
    th[:, body_indices] = body_th
    th[:, left_hand_indices] = l_hand_th
    th[:, right_hand_indices] = r_hand_th
    return th

def get_kps(chain: pk.Chain, th: torch.Tensor, trans: torch.Tensor, rotmat: torch.Tensor, link_names: list):
    body_pos = chain.forward_kinematics(th)
    local_pos = torch.stack(
        [body_pos[name].get_matrix()[:, :3, 3] for name in link_names], 
        dim=1
        )
    pos_w = torch.matmul(rotmat.unsqueeze(1), local_pos.unsqueeze(-1)).squeeze(-1)
    return pos_w + trans.unsqueeze(1)

def retarget_body(
    chain: pk.Chain, 
    body_th: torch.Tensor, l_hand_th: torch.Tensor, r_hand_th: torch.Tensor, 
    robot_trans: torch.Tensor, robot_rotmat: torch.Tensor, 
    keypoints: torch.Tensor, 
    body_indices: torch.Tensor, left_hand_indices: torch.Tensor, right_hand_indices: torch.Tensor, 
    total_dof: int,
    fps: int = 30
):

    body_link_names = list(BODY_JOINT_MAP.keys())
    smpl_joint_names = list(BODY_JOINT_MAP.values())
    smpl_indices = [SMPLH_BONE_ORDER_NAMES.index(n) for n in smpl_joint_names]
    smpl_keypoints = keypoints[:, smpl_indices, :]

    weights = torch.tensor(
        [END_EFFECTOR_WEIGHTS.get(name, 1.0) for name in body_link_names], 
        dtype=torch.float32, 
        device=DEVICE
    ).view(1, -1, 1)

    opt = torch.optim.Adam([body_th, robot_trans], lr=0.02)

    for i in range(200):
        opt.zero_grad()

        robot_th = combine_configs(
            body_th, l_hand_th, r_hand_th, 
            body_indices, left_hand_indices, right_hand_indices, 
            total_dof
        )

        robot_keypoints = get_kps(chain, robot_th, robot_trans, robot_rotmat, body_link_names)
        omega = torch.gradient(body_th, spacing=1.0 / fps, dim=0)[0]
        loss_kp = torch.mean((robot_keypoints - smpl_keypoints) ** 2 * weights)
        loss_qpos = torch.mean(body_th ** 2)
        loss_qvel = torch.mean(omega ** 2)
        loss = loss_kp + 1e-2 * loss_qpos + 1e-3 * loss_qvel
        loss.backward()
        opt.step()

        if i % 50 == 0:
            print(
                f"iter {i}, loss {loss.item():.6f}, "
                f"kp {loss_kp.item():.6f}, "
                f"qpos {loss_qpos.item():.6f}, "
                f"qvel {loss_qvel.item():.6f}, "
            )
    return body_th, robot_trans

def retarget_hand(
    chain: pk.Chain, 
    body_th: torch.Tensor, l_hand_th: torch.Tensor, r_hand_th: torch.Tensor, 
    robot_trans: torch.Tensor, robot_rotmat: torch.Tensor, 
    keypoints: torch.Tensor, 
    body_indices: torch.Tensor, left_hand_indices: torch.Tensor, right_hand_indices: torch.Tensor, 
    total_dof: int,
    fps: int = 30
):
    hand_link_names = list(HAND_TIP_MAP.keys())
    smpl_joint_names = list(HAND_TIP_MAP.values())
    smpl_indices = [SMPLH_BONE_ORDER_NAMES.index(n) for n in smpl_joint_names]
    smpl_keypoints = keypoints[:, smpl_indices, :]

    opt = torch.optim.Adam([l_hand_th, r_hand_th], lr=0.01)

    for i in range(200):
        opt.zero_grad()

        robot_th = combine_configs(
            body_th, l_hand_th, r_hand_th, 
            body_indices, left_hand_indices, right_hand_indices, 
            total_dof
        )

        robot_keypoints = get_kps(chain, robot_th, robot_trans, robot_rotmat, hand_link_names)
        loss_kp = torch.mean((robot_keypoints - smpl_keypoints) ** 2)
        loss_qpos = torch.mean(l_hand_th ** 2) + torch.mean(r_hand_th ** 2)
        loss = loss_kp + 1e-2 * loss_qpos
        loss.backward()
        opt.step()

        if i % 50 == 0:
            print(
                f"iter {i}, loss {loss.item():.6f}, "
                f"kp {loss_kp.item():.6f}, "
                f"qpos {loss_qpos.item():.6f}, "
            )
    return l_hand_th, r_hand_th

def run_retarget(chain: pk.Chain, body_indices: torch.Tensor, left_hand_indices: torch.Tensor, right_hand_indices: torch.Tensor, seq_data: dict, fps: int = 30):
    total_dof = chain.n_joints
    keypoints = torch.tensor(seq_data["human"]["keypoints"], dtype=torch.float32, device=DEVICE)

    """
    Adjust ankle height due to morphology gap
    """
    ANKLE_OFFSET_Z = 0.2
    l_ankle_idx = SMPLH_BONE_ORDER_NAMES.index("L_Ankle")
    r_ankle_idx = SMPLH_BONE_ORDER_NAMES.index("R_Ankle")
    keypoints[:, l_ankle_idx, 2] -= ANKLE_OFFSET_Z
    keypoints[:, r_ankle_idx, 2] -= ANKLE_OFFSET_Z

    T = keypoints.shape[0]
    trans = seq_data["human"]["trans"]
    poses = seq_data["human"]["poses"].reshape(T, -1, 3)
    
    # Root orientation
    root_orient = poses[:, 0, :]
    robot_rot = sRot.from_rotvec(root_orient) * sRot.from_euler(
        "xyz", [np.pi / 2, 0.0, np.pi / 2]
    ).inv()
    robot_rotmat = torch.tensor(robot_rot.as_matrix(), dtype=torch.float32, device=DEVICE)

    body_th = torch.nn.Parameter(torch.zeros(T, body_indices.shape[0]).to(DEVICE))
    l_hand_th = torch.nn.Parameter(torch.zeros(T, left_hand_indices.shape[0]).to(DEVICE))
    r_hand_th = torch.nn.Parameter(torch.zeros(T, right_hand_indices.shape[0]).to(DEVICE))
    robot_trans = torch.nn.Parameter(torch.from_numpy(trans).float().to(DEVICE))

    print(f"[Info] DoF Split: Body={body_indices.shape[0]}, LeftHand={left_hand_indices.shape[0]}, RightHand={right_hand_indices.shape[0]}")

    body_th, robot_trans = retarget_body(
        chain,
        body_th, l_hand_th, r_hand_th, 
        robot_trans, robot_rotmat, 
        keypoints, 
        body_indices, left_hand_indices, right_hand_indices,
        total_dof
    )

    l_hand_th, r_hand_th = retarget_hand(
        chain,
        body_th, l_hand_th, r_hand_th, 
        robot_trans, robot_rotmat, 
        keypoints, 
        body_indices, left_hand_indices, right_hand_indices,
        total_dof
    )

    return {
        "root_trans": robot_trans.detach().cpu().numpy(),
        "root_quat": sRot.from_matrix(robot_rotmat.detach().cpu().numpy()).as_quat(scalar_first=True),
        "joint_pos": combine_configs(
            body_th, l_hand_th, r_hand_th, 
            body_indices, left_hand_indices, right_hand_indices, 
            total_dof
        ).detach().cpu().numpy(),
    }

def visualizition(results: dict, seq_data: dict, chain: pk.Chain, fps: int = 30):
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

    chain, body_indices, left_hand_indices, right_hand_indices = build_chain(G1_URDF_PATH)

    # sequences_path = os.path.join(OUTPUT_PATH, f'{args.flag}_sequences.pkl')
    sequences_path = os.path.join(OUTPUT_PATH, 'example.pkl')
    sequences = joblib.load(sequences_path)

    for object_name, object_seqs in sequences.items():
        for seq_name, seq_data in object_seqs.items():
            print(f"Retargeting {seq_name} with {object_name}")

            results = run_retarget(chain, body_indices, left_hand_indices, right_hand_indices, seq_data)

            if args.visualize:
                visualizition(results, seq_data, chain)

if __name__ == "__main__":
    main()