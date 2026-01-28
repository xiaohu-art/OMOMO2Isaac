import os
import joblib
import numpy as np
from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from utils.cli_args import DATA_ROOT, OUTPUT_PATH, SMPLX_PATH, parse_args
from utils.math import (
    quat_from_angle_axis, quat_from_matrix,
    matrix_from_quat, axis_angle_from_quat,
    rotate_at_frame_w_obj
)
from utils.process import (
    get_smpl_parents,
    generate_subject_xml,
    process_single_sequence,
)

def load_smpl_models(smplx_path):
    models = {}
    num_betas = 16
    for gender in ['male', 'female', 'neutral']:
        bm_fname = os.path.join(smplx_path, f"SMPLX_{gender.upper()}.npz")
        if not os.path.exists(bm_fname):
            print(f"Warning: Model not found at {bm_fname}")
            continue
            
        models[gender] = BodyModel(
            bm_fname=bm_fname,
            num_betas=num_betas,
            num_expressions=None,
            num_dmpls=None
        )
    return models

def canonicalize_sequence(seq_data, gender):
    trans = seq_data['trans']                                   # T X 3
    global_orient = seq_data['root_orient']                     # T X 3
    body_pose = seq_data['pose_body'].reshape(-1, 21, 3)        # T X 21 X 3
    rest_human_offsets = seq_data['rest_offsets']               # J(24) X 3 (T-pose/Rest pose offsets)
    trans2joint = seq_data['trans2joint']                       # (3,)
    
    obj_trans = seq_data['obj_trans'][:, :, 0]                  # T X 3
    obj_rot = seq_data['obj_rot']                               # T X 3 X 3
    obj_com_pos = seq_data['obj_com_pos']                       # T X 3
    
    timesteps = len(trans)

    joint_aa_rep = np.concatenate(
        (
            global_orient[:, None, :], 
            body_pose
        ), 
        axis=1
    )  # T X 22 X 3
    
    local_quats = quat_from_angle_axis(joint_aa_rep)

    rest_pose_global = np.tile(rest_human_offsets[None, :, :], (timesteps, 1, 1))  # T X J X 3
    rest_pose_global[:, 0, :] = trans

    obj_quats = quat_from_matrix(obj_rot) # T X 4

    parents = get_smpl_parents(gender=gender)

    _, _, new_obj_x, new_obj_q = rotate_at_frame_w_obj(
        rest_pose_global[np.newaxis],       # 1 X T X J X 3
        local_quats[np.newaxis],            # 1 X T X J X 4
        obj_trans.copy()[np.newaxis],       # 1 X T X 3 (Object Translation)
        obj_quats[np.newaxis],              # 1 X T X 4 (Object Rotation)
        trans2joint[np.newaxis],            # 1 X 3
        parents,
        n_past=1,
        floor_z=True
    )

    new_rest_pose_global, new_local_quats, _, _ = rotate_at_frame_w_obj(
        rest_pose_global[np.newaxis],
        local_quats[np.newaxis],
        obj_com_pos[np.newaxis],
        obj_quats[np.newaxis],
        trans2joint[np.newaxis],
        parents,
        n_past=1,
        floor_z=True
    )

    new_seq_root_trans = new_rest_pose_global[0, :, 0, :]   # T X 3
    
    new_local_aa = axis_angle_from_quat(new_local_quats[0])
    
    new_seq_root_orient = new_local_aa[:, 0, :]             # T X 3
    new_seq_pose_body = new_local_aa[:, 1:, :]              # T X 21 X 3
    
    new_obj_rot_mat = matrix_from_quat(new_obj_q[0])
    new_obj_trans = new_obj_x[0]

    padding_zeros_hand = np.zeros((timesteps, 90))
    
    poses = np.concatenate((
        new_seq_root_orient, 
        new_seq_pose_body.reshape(-1, 63), 
        padding_zeros_hand
    ), axis=1)

    return poses, new_seq_root_trans, new_obj_rot_mat, new_obj_trans

def main():
    args = parse_args()

    seq_data_path = os.path.join(DATA_ROOT, f"{args.flag}_diffusion_manip_seq_joints24.p")
    print(f"Loading data from {seq_data_path}...")
    data_dict = joblib.load(seq_data_path)
    
    print("Loading SMPL models...")
    smpl_models = load_smpl_models(SMPLX_PATH)

    all_sequences = {}

    keys = list(data_dict.keys())
    pbar = tqdm(keys)
    
    for index in pbar:
        seq_entry = data_dict[index]
        seq_name = seq_entry['seq_name']
        subject_name = seq_name.split("_")[0]
        object_name = seq_name.split("_")[1]
        
        pbar.set_description(f"Processing {seq_name}")

        if object_name in ["mop", "vacuum"]:
            continue
        
        gender = str(seq_entry['gender'])
        betas = seq_entry['betas'][0]

        xml_filename = f"{subject_name}.xml"
        if not os.path.exists(xml_filename):
            generate_subject_xml(betas, gender, xml_filename)

        new_poses, new_trans, new_obj_rot, new_obj_trans = canonicalize_sequence(seq_entry, gender)
        
        human_input = {
            'poses': new_poses,
            'betas': betas,
            'trans': new_trans,
            'gender': gender,
        }
        
        obj_scale = seq_entry['obj_scale'].mean()
        object_input = {
            'rot': new_obj_rot,
            'trans': new_obj_trans,
            'name': object_name,
            'scale': obj_scale,
        }

        processed_human, processed_obj = process_single_sequence(
            human_input, 
            object_input, 
            smpl_models[gender],
            visualize=args.visualize
        )

        if object_name not in all_sequences:
            all_sequences[object_name] = {}
        
        all_sequences[object_name][seq_name] = {
            'human': processed_human,
            'object': processed_obj,
        }

        break

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, f'{args.flag}_sequences.pkl')
    print(f"Saving {len(all_sequences)} object categories to {output_file}")
    joblib.dump(all_sequences, output_file)

if __name__ == "__main__":
    main()