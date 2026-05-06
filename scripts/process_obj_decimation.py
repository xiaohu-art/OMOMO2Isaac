import os
import argparse
import trimesh
from tqdm import tqdm

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from utils.cli_args import DATA_ROOT, OBJECTS_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=OBJECTS_PATH,
                        help="Directory of scaled .obj meshes")
    parser.add_argument("--output_dir", type=str, default=os.path.join(DATA_ROOT, "objects_lite"),
                        help="Directory to write decimated meshes")
    parser.add_argument("--target_verts", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-decimate and overwrite existing outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    obj_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".obj"))
    pbar = tqdm(obj_files)

    for fname in pbar:
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        pbar.set_description(f"Decimating {fname}")

        if os.path.exists(out_path) and not args.overwrite:
            continue

        mesh = trimesh.load(in_path, force="mesh")
        n_verts, n_faces = len(mesh.vertices), len(mesh.faces)

        if n_verts <= args.target_verts:
            mesh.export(out_path)
            tqdm.write(f"  {fname}: {n_verts}V/{n_faces}F (already <= target, copied)")
            continue

        target_faces = max(int(n_faces * args.target_verts / n_verts), 4)
        mesh_lite = mesh.simplify_quadric_decimation(face_count=target_faces)
        mesh_lite.export(out_path)

        tqdm.write(
            f"  {fname}: {n_verts}V/{n_faces}F -> "
            f"{len(mesh_lite.vertices)}V/{len(mesh_lite.faces)}F"
        )

    print(f"\nDone. Decimated meshes written to {args.output_dir}")


if __name__ == "__main__":
    main()
