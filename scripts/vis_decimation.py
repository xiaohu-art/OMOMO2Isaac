import argparse
import os
import time

import numpy as np
import trimesh
import viser

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from utils.cli_args import OBJECTS_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, default="clothesstand")
    parser.add_argument("--target_verts", type=int, default=2048)
    parser.add_argument("--separation", type=float, default=1.5,
                        help="X-axis offset between original and decimated mesh")
    args = parser.parse_args()

    mesh_path = os.path.join(OBJECTS_PATH, f"{args.object}.obj")
    mesh = trimesh.load(mesh_path, force="mesh")

    target_faces = max(int(len(mesh.faces) * args.target_verts / len(mesh.vertices)), 4)
    mesh_lite = mesh.simplify_quadric_decimation(face_count=target_faces)

    print(f"Original:  {len(mesh.vertices):6d} verts, {len(mesh.faces):6d} faces")
    print(f"Decimated: {len(mesh_lite.vertices):6d} verts, {len(mesh_lite.faces):6d} faces")

    # Center each mesh on its own bbox center, then shift along X.
    def centered(m, x_offset):
        v = np.asarray(m.vertices, dtype=np.float32).copy()
        v -= v.mean(axis=0, keepdims=True)
        v[:, 0] += x_offset
        return v

    half = args.separation / 2.0
    v_orig = centered(mesh, -half)
    v_lite = centered(mesh_lite, +half)
    f_orig = np.asarray(mesh.faces, dtype=np.int32)
    f_lite = np.asarray(mesh_lite.faces, dtype=np.int32)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=4.0, height=4.0, cell_size=0.1)

    server.scene.add_mesh_simple(
        name="/original",
        vertices=v_orig,
        faces=f_orig,
        color=(90, 200, 255),
        wireframe=False,
        opacity=1.0,
    )
    server.scene.add_mesh_simple(
        name="/decimated",
        vertices=v_lite,
        faces=f_lite,
        color=(255, 160, 90),
        wireframe=False,
        opacity=1.0,
    )
    server.scene.add_mesh_simple(
        name="/decimated_wire",
        vertices=v_lite,
        faces=f_lite,
        color=(40, 40, 40),
        wireframe=True,
        opacity=1.0,
    )

    print("Viser running. Ctrl+C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
