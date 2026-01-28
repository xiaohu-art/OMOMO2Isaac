import os
import trimesh
import joblib
from tqdm import tqdm

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from utils.cli_args import parse_args, DATA_ROOT

def main():
    """Main processing function."""
    args = parse_args()
    
    # Setup paths
    original_object_path = args.original_object_path
    scaled_object_path = args.scaled_object_path
    seq_data_path = os.path.join(DATA_ROOT, args.flag + '_diffusion_manip_seq_joints24.p')
    
    # Load data
    num_objects = len(os.listdir(original_object_path))
    scaled_objects = set(os.listdir(scaled_object_path))
    data_dict = joblib.load(seq_data_path)
    
    # Process objects
    pbar = tqdm(sorted(list(data_dict.keys())))
    for index in pbar:
        seq_name = data_dict[index]['seq_name']
        object_name = seq_name.split("_")[1]
        pbar.set_description(f"Scaling {object_name}, now {len(scaled_objects)} / {num_objects}")

        if object_name in scaled_objects:
            continue
        
        if len(scaled_objects) == num_objects - 4:  # mop_top, mop_bottom, vacuum_top, vacuum_bottom
            break

        # Calculate scale
        obj_scale = data_dict[index]['obj_scale'].mean()  # T X 1

        # Load and scale mesh
        mesh_path = os.path.join(original_object_path, f"{object_name}_cleaned_simplified.obj")
        mesh_obj = trimesh.load(mesh_path, force='mesh')
        obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces
        obj_verts = obj_verts * obj_scale

        # Export scaled mesh
        scaled_mesh = trimesh.Trimesh(obj_verts, obj_faces)
        export_path = os.path.join(scaled_object_path, f"{object_name}.obj")
        scaled_mesh.export(export_path)
        scaled_objects.add(object_name)


if __name__ == "__main__":
    main()