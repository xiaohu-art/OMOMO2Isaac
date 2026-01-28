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

END_EFFECTOR_WEIGHTS = {
    "L_hand_base_link": 5.0,
    "R_hand_base_link": 5.0,
    "left_ankle_roll_link": 5.0,
    "right_ankle_roll_link": 5.0,
}

def build_chain(urdf_path: str) -> pk.Chain:
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()

    chain = pk.build_chain_from_urdf(urdf_content)
    return chain.to(device=DEVICE)

def run_retarget(chain: pk.Chain, seq_data: dict, fps: int = 30):
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
    root_orient = poses[:, 0, :]
    robot_rot = sRot.from_rotvec(root_orient) * sRot.from_euler(
        "xyz", [np.pi / 2, 0.0, np.pi / 2]
    ).inv()
    robot_rotmat = torch.tensor(robot_rot.as_matrix(), dtype=torch.float32, device=DEVICE)

    robot_link_names = list(JOINT_MAP.keys())
    smpl_joint_names = list(JOINT_MAP.values())
    smpl_indices = [SMPLH_BONE_ORDER_NAMES.index(n) for n in smpl_joint_names]
    smpl_keypoints = keypoints[:, smpl_indices, :]

    weights = torch.tensor(
        [END_EFFECTOR_WEIGHTS.get(name, 1.0) for name in robot_link_names], 
        dtype=torch.float32, 
        device=DEVICE
    ).view(1, -1, 1)

    # Initialize optimization variables
    robot_th = torch.nn.Parameter(torch.zeros(T, chain.n_joints).to(DEVICE))
    robot_trans = torch.nn.Parameter(torch.from_numpy(trans).float().to(DEVICE))
    opt = torch.optim.Adam([robot_th, robot_trans], lr=0.02)
    
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

    for i in range(300):
        opt.zero_grad()

        robot_kp_w = get_robot_keypoints(robot_th, robot_trans, robot_rotmat)
        omega = torch.gradient(robot_th, spacing=1.0 / fps, dim=0)[0]
        
        keypoints_pos_error = torch.mean((robot_kp_w - smpl_keypoints) ** 2 * weights)
        joint_pos_reg = torch.mean(torch.square(robot_th))
        joint_vel_reg = torch.mean(torch.square(omega))
        loss = keypoints_pos_error + 1e-2 * joint_pos_reg + 1e-3 * joint_vel_reg

        loss.backward()
        opt.step()

        if i % 50 == 0:
            print(
                f"iter {i}, loss {loss.item():.6f}, "
                f"kp {keypoints_pos_error.item():.6f}, "
                f"j_pos {joint_pos_reg.item():.6f}, "
                f"j_vel {joint_vel_reg.item():.6f}, "
            )

    return {
        "root_trans": robot_trans.detach().cpu().numpy(),
        "root_quat": sRot.from_matrix(
                        robot_rotmat.detach().cpu().numpy()
                    ).as_quat(scalar_first=True),
        "joint_pos": robot_th.detach().cpu().numpy(),
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

            if args.visualize:
                visualize_retarget(results, seq_data, chain)

if __name__ == "__main__":
    main()