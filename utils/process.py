import os
import torch
import trimesh
import numpy as np

from .cli_args import SMPLX_PATH, OBJECTS_PATH, DEVICE
from .visualization import visualize_with_viser
from .optimize import optimize_hand

def get_smpl_parents(gender, use_joints24=False):
    bm_path = os.path.join(SMPLX_PATH, f"SMPLX_{gender.upper()}.npz")
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table']  # 2 X 52

    if use_joints24:
        parents = ori_kintree_table[0, :23]  # 23
        parents[0] = -1  # Assign -1 for the root joint's parent idx.

        parents_list = parents.tolist()
        parents_list.append(ori_kintree_table[0][37])
        parents = np.asarray(parents_list)  # 24
    else:
        parents = ori_kintree_table[0, :22]  # 22
        parents[0] = -1  # Assign -1 for the root joint's parent idx.

    return parents

# -----------------------------------------------------------------------------
# Core Pipeline
# -----------------------------------------------------------------------------

def process_single_sequence(human_data, object_data, smpl_model, visualize=False):
    """Process a single sequence: hand optimization, SMPL reconstruction, and contact computation.

    Args:
        human_data: Dictionary containing poses, betas, trans, gender.
        object_data: Dictionary containing rot, trans, name, scale.
        smpl_model: SMPL model instance.
        visualize: Whether to enable visualization. Defaults to False.

    Returns:
        Tuple of (result_human, result_obj) dictionaries containing processed data.
    """
    poses = human_data['poses']
    betas = human_data['betas']
    trans = human_data['trans']
    gender = str(human_data['gender'])
    
    obj_rot = object_data['rot']
    obj_trans = object_data['trans']
    obj_name = str(object_data['name'])
    obj_scale = object_data['scale']

    obj_mesh_path = os.path.join(OBJECTS_PATH, f"{obj_name}.obj")
    mesh_obj = trimesh.load(obj_mesh_path, force='mesh')
    obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces
    frame_times = poses.shape[0]
    
    smpl_model.to(DEVICE)
    optimized_hand_pose = optimize_hand(
        frame_times, poses, betas, trans, smpl_model, 
        obj_rot, obj_trans, obj_verts, obj_faces,
        epochs=1000, lr=0.001
    )

    smpl_model.to('cpu')
    optimized_smplx_output = smpl_model(
        pose_body=torch.from_numpy(poses[:, 3:66]).float(), 
        pose_hand=torch.from_numpy(optimized_hand_pose).float(), 
        betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(), 
        root_orient=torch.from_numpy(poses[:, :3]).float(), 
        trans=torch.from_numpy(trans).float()
    )
    
    optimized_verts = optimized_smplx_output.v.detach().numpy()
    optimized_faces = optimized_smplx_output.f.detach().numpy()
    
    pelvis = optimized_smplx_output.Jtr.detach().numpy()[:, 0, :]
    body_keypoints = optimized_smplx_output.Jtr.detach().numpy()[:, :22, :]
    hand_keypoints = optimized_smplx_output.Jtr.detach().numpy()[:, 25:, :]
    keypoints = np.concatenate([body_keypoints, hand_keypoints], axis=1)

    offset = pelvis - trans
    final_trans = trans + offset

    # Transform object vertices to world coordinates: [N, 3] x [T, 3, 3] -> [T, N, 3]
    obj_verts_world = np.matmul(obj_verts[None, :, :], np.transpose(obj_rot, (0, 2, 1))) + obj_trans[:, None, :]

    # Normalize Z: align minimum z-coordinate to 0
    min_z = min(optimized_verts[:, :, 2].min(), obj_verts_world[:, :, 2].min())
    optimized_verts[:, :, 2] -= min_z
    obj_trans[..., 2] -= min_z
    final_trans[..., 2] -= min_z
    keypoints[:, :, 2] -= min_z
    obj_verts_world[:, :, 2] -= min_z

    # Compute contacts based on distance and relative velocity
    dist_matrix = np.linalg.norm(keypoints[:, None, :, :] - obj_verts_world[:, :, None, :], axis=-1)
    distance = dist_matrix.min(axis=1)  # [T, K] - minimum distance from each keypoint to object

    contacts = np.zeros_like(distance, dtype=np.float32)
    contacts[distance < 0.05] = 1.0
    contacts[distance > 0.2] = -1.0

    optimized_poses = np.concatenate([
        poses[:, :3],
        poses[:, 3:66],
        optimized_hand_pose
    ], axis=1)

    if visualize:
        visualize_with_viser(
            optimized_verts, optimized_faces, 
            obj_verts, obj_faces, 
            obj_trans, obj_rot,
            keypoints,
            contacts
        )

    result_obj = {
        'rot': np.array(obj_rot),
        'trans': np.array(obj_trans),
        'name': obj_name,
        'scale': obj_scale,
    }

    result_human = {
        'poses': np.array(optimized_poses),
        'betas': np.array(betas),
        'trans': np.array(final_trans),
        'gender': gender,
        'keypoints': np.array(keypoints),
        'contacts': np.array(contacts),
    }
    
    return result_human, result_obj