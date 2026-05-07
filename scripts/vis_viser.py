"""
Viser-based browser for retargeted humanoid sequences.

Reads `{flag}_retargeted/{object}/{seq}.pkl` and lets you pick an object +
sequence from dropdowns, then plays the retargeted G1 motion alongside the
manipulated object.

Usage:
    uv run scripts/vis_viser.py --flag train
    uv run scripts/vis_viser.py --flag test
"""

import os
import time
import joblib
import trimesh
import yourdfpy
import numpy as np
import pytorch_kinematics as pk

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from utils.cli_args import parse_args, OBJECTS_PATH, OUTPUT_PATH, G1_PATH

G1_URDF_PATH = os.path.join(G1_PATH, "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf")


def discover_sequences(retarget_root: str) -> dict:
    """Walk `{retarget_root}/{object}/{seq}.pkl` and return {object: [seq_names]}."""
    catalog = {}
    if not os.path.isdir(retarget_root):
        return catalog
    for obj_name in sorted(os.listdir(retarget_root)):
        obj_dir = os.path.join(retarget_root, obj_name)
        if not os.path.isdir(obj_dir):
            continue
        seqs = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(obj_dir)
            if f.endswith(".pkl")
        )
        if seqs:
            catalog[obj_name] = seqs
    return catalog


def main():
    args = parse_args()
    retarget_root = os.path.join(OUTPUT_PATH, f"{args.flag}_retargeted")
    catalog = discover_sequences(retarget_root)
    if not catalog:
        print(f"No retargeted sequences found at {retarget_root}")
        print("Run scripts/humanoid_retarget.py first.")
        return

    total = sum(len(v) for v in catalog.values())
    print(f"Found {total} sequences across {len(catalog)} objects in {retarget_root}")

    import viser
    import viser.transforms as vtf
    from viser.extras import ViserUrdf

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=4.0, height=4.0, cell_size=0.1)

    # Robot URDF + persistent base frames
    urdf = yourdfpy.URDF.load(G1_URDF_PATH)
    robot_base = server.scene.add_frame("/robot_base", show_axes=True)
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot_base")
    obj_base = server.scene.add_frame("/object_frame", show_axes=True)

    chain = pk.build_chain_from_urdf(open(G1_URDF_PATH).read())
    pk_joint_names = chain.get_joint_parameter_names()

    # Mutable animation state — overwritten on each sequence load.
    state = {
        "root_trans": np.zeros((1, 3), dtype=np.float32),
        "root_quat": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "joint_pos": np.zeros((1, len(pk_joint_names)), dtype=np.float32),
        "obj_trans": np.zeros((1, 3), dtype=np.float32),
        "obj_rot": np.eye(3, dtype=np.float32)[None],
        "T": 1,
        "obj_mesh_handle": None,
    }

    obj_names = sorted(catalog.keys())
    first_obj = obj_names[0]
    first_seq = catalog[first_obj][0]

    with server.gui.add_folder("Selection"):
        gui_label = server.gui.add_markdown("(loading...)")
        gui_object = server.gui.add_dropdown(
            "Object", options=obj_names, initial_value=first_obj,
        )
        gui_sequence = server.gui.add_dropdown(
            "Sequence", options=catalog[first_obj], initial_value=first_seq,
        )

    with server.gui.add_folder("Playback"):
        gui_playing = server.gui.add_checkbox("Playing", initial_value=True)
        gui_frame = server.gui.add_slider("Frame", min=0, max=10000, step=1, initial_value=0)
        gui_fps = server.gui.add_slider("FPS", min=1, max=120, step=1, initial_value=30)
        gui_loop = server.gui.add_checkbox("Loop", initial_value=True)

    def load_sequence(obj_name: str, seq_name: str):
        path = os.path.join(retarget_root, obj_name, f"{seq_name}.pkl")
        if not os.path.exists(path):
            print(f"  missing: {path}")
            return
        results = joblib.load(path)

        # Replace object mesh
        if state["obj_mesh_handle"] is not None:
            state["obj_mesh_handle"].remove()
        mesh_path = os.path.join(OBJECTS_PATH, f"{results['object']['name']}.obj")
        mesh_obj = trimesh.load(mesh_path, force="mesh")
        state["obj_mesh_handle"] = server.scene.add_mesh_simple(
            "/object_frame/object",
            vertices=mesh_obj.vertices,
            faces=mesh_obj.faces,
            color=(0.2, 0.8, 0.2),
        )

        # Swap animation arrays
        state["root_trans"] = np.asarray(results["root_trans"], dtype=np.float32)
        state["root_quat"] = np.asarray(results["root_quat"], dtype=np.float32)
        state["joint_pos"] = np.asarray(results["joint_pos"], dtype=np.float32)
        state["obj_trans"] = np.asarray(results["object"]["trans"], dtype=np.float32)
        state["obj_rot"] = np.asarray(results["object"]["rot"], dtype=np.float32)
        state["T"] = int(state["root_trans"].shape[0])

        gui_label.content = f"**{obj_name} / {seq_name}**  (T = {state['T']})"
        try:
            gui_frame.max = state["T"] - 1
            gui_frame.value = 0
        except Exception:
            pass
        print(f"  loaded {obj_name}/{seq_name}  T={state['T']}")

    @gui_object.on_update
    def _(_):
        seqs = catalog[gui_object.value]
        gui_sequence.options = seqs
        gui_sequence.value = seqs[0]  # triggers gui_sequence.on_update

    @gui_sequence.on_update
    def _(_):
        load_sequence(gui_object.value, gui_sequence.value)

    def update_frame(t: int):
        t = max(0, min(int(t), state["T"] - 1))
        robot_base.position = state["root_trans"][t]
        robot_base.wxyz = state["root_quat"][t]
        cfg = {name: state["joint_pos"][t, i] for i, name in enumerate(pk_joint_names)}
        viser_urdf.update_cfg(cfg)
        obj_base.position = state["obj_trans"][t]
        obj_base.wxyz = vtf.SO3.from_matrix(state["obj_rot"][t]).wxyz

    @gui_frame.on_update
    def _(_):
        if not gui_playing.value:
            update_frame(gui_frame.value)

    # Initial load
    load_sequence(first_obj, first_seq)

    print("Open browser at http://localhost:8080 (or whichever port viser prints)")
    t = 0
    while True:
        if gui_playing.value:
            update_frame(t)
            gui_frame.value = t
            t = (t + 1) % state["T"] if gui_loop.value else min(t + 1, state["T"] - 1)
            time.sleep(1.0 / max(1, gui_fps.value))
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
