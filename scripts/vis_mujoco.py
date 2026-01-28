import os
import time
import joblib
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from utils.cli_args import (
    parse_args, 
    OUTPUT_PATH, 
    SMPLX_ROBOTS_PATH, OBJECTS_PATH,
    SMPLH_BONE_ORDER_NAMES, 
    MUJOCO_BONE_ORDER_NAMES
)

SMPLH2MJC = [SMPLH_BONE_ORDER_NAMES.index(name) for name in MUJOCO_BONE_ORDER_NAMES]

def create_scene_xml(human_xml_path, obj_mesh_path, obj_name):
    with open(human_xml_path, 'r') as f:
        human_xml_content = f.read()

    obj_mesh_path = os.path.abspath(obj_mesh_path)
    
    obj_assets = f"""
        <mesh name="{obj_name}" file="{obj_mesh_path}"/>
    """
    obj_bodies = f"""
        <body name="{obj_name}" mocap="true" pos="0 0 0" quat="1 0 0 0">
            <geom type="mesh" mesh="{obj_name}" rgba="0.2 0.8 0.2 1.0" group="1"/>
        </body>
    """

    modified_xml = human_xml_content.replace("</asset>", f"{obj_assets}\n  </asset>")
    modified_xml = modified_xml.replace("</worldbody>", f"{obj_bodies}\n  </worldbody>")
    return modified_xml

def main():
    args = parse_args()

    # sequences_path = os.path.join(OUTPUT_PATH, f'{args.flag}_sequences.pkl')
    sequences_path = os.path.join(OUTPUT_PATH, 'example.pkl')
    sequences = joblib.load(sequences_path)

    playlist = []
    for object_name, obj_seqs in sequences.items():
        for seq_name, seq_data in obj_seqs.items():
            playlist.append({
                "object_name": object_name,
                "seq_name": seq_name,
                "data": seq_data
            })

    current_idx, reload = 0, False
    total_seqs = len(playlist)
    
    print(f"[INFO] Total sequences: {total_seqs}")
    print("[INFO] Controls: [UP] Next Motion, [DOWN] Prev Motion")

    while True:
        current_item = playlist[current_idx]
        object_name = current_item['object_name']
        seq_name = current_item['seq_name']
        sub_name = seq_name.split("_")[0]
        human_data, object_data = current_item['data']['human'], current_item['data']['object']

        human_xml_path = os.path.join(SMPLX_ROBOTS_PATH, f'{sub_name}.xml')
        obj_mesh_path = os.path.join(OBJECTS_PATH, f'{object_name}.obj')
        
        scene_xml = create_scene_xml(human_xml_path, obj_mesh_path, object_name)
        model = mujoco.MjModel.from_xml_string(scene_xml)
        data = mujoco.MjData(model)
        obj_mocap_id = model.body(object_name).mocapid[0]

        trans = human_data["trans"]
        poses = human_data["poses"].reshape(-1, len(SMPLH_BONE_ORDER_NAMES), 3)
        poses_mjc = poses[:, SMPLH2MJC, :]

        root_orient = poses_mjc[:, 0, :]
        body_pose = poses_mjc[:, 1:, :]

        obj_trans = object_data["trans"]
        obj_rot = object_data["rot"]

        num_frames = trans.shape[0]

        def key_callback(keycode):
            nonlocal current_idx, reload
            
            if keycode == 265:
                current_idx = (current_idx - 1 + total_seqs) % total_seqs
                reload = True
            
            elif keycode == 264:
                current_idx = (current_idx + 1) % total_seqs
                reload = True
            return True

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            frame = 0
            while viewer.is_running():

                data.qpos[:3] = trans[frame]
                data.qpos[3:7] = R.from_rotvec(root_orient[frame]).as_quat(scalar_first=True)

                body_aa = body_pose[frame]
                body_euler = R.from_rotvec(body_aa).as_euler('XYZ', degrees=False)
                data.qpos[7:] = body_euler.flatten()

                data.mocap_pos[obj_mocap_id] = obj_trans[frame]
                data.mocap_quat[obj_mocap_id] = R.from_matrix(obj_rot[frame]).as_quat(scalar_first=True)

                mujoco.mj_forward(model, data)
                viewer.sync()
                
                frame = (frame + 1) % num_frames
                time.sleep(1 / 30)

                if reload:
                    reload = False
                    break

if __name__ == "__main__":
    main()