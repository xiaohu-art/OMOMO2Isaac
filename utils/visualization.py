import time
import numpy as np
import viser
import viser.transforms as vtf

def visualize_with_viser(
    body_verts,     # T, V, 3
    body_faces,     # F, 3
    obj_verts,
    obj_faces,
    obj_trans,     # T, 3
    obj_rot,       # T, 3, 3
    keypoints,     # T, K, 3
    contacts,      # T, K
    fps: int =30
    ):
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=4.0, height=4.0, cell_size=0.1)

    T = body_verts.shape[0]

    body_handle = server.scene.add_mesh_simple(
        name="/smplx",
        vertices=body_verts[0],
        faces=body_faces,
        color=(90, 200, 255),
        wireframe=False,
        opacity=0.8
    )

    obj_handle = server.scene.add_mesh_simple(
        name="/object",
        vertices=obj_verts,
        faces=obj_faces,
        color=(0.2, 0.8, 0.2),
    )

    initial_colors = get_contact_colors(contacts[0])
    joint_handle = server.scene.add_point_cloud(
        name="/joints",
        points=keypoints[0],
        colors=initial_colors,
        point_size=0.02,
        opacity=1.0
    )

    t = 0
    try:
        while True:
            body_handle.vertices = body_verts[t]

            obj_pos = obj_trans[t]
            obj_rotmat = obj_rot[t]
            obj_handle.position = obj_pos
            obj_handle.wxyz = vtf.SO3.from_matrix(obj_rotmat).wxyz

            joint_handle.points = keypoints[t]
            joint_handle.colors = get_contact_colors(contacts[t])

            t = (t+1) % T
            time.sleep(1 / fps)
    except KeyboardInterrupt:
        print("\n Viser server stopped.")

def get_contact_colors(contacts_frame):
    """
    contacts_frame: [K] shape
    Returns: [K, 3] colors (RGB, 0-1)
    """
    K = contacts_frame.shape[0]
    colors = np.zeros((K, 3))
    
    # Contact (1.0) -> Red
    mask_contact = (contacts_frame == 1.0)
    colors[mask_contact] = [1.0, 0.0, 0.0]
    
    # Near (0.0) -> Yellow
    mask_near = (contacts_frame == 0.0)
    colors[mask_near] = [1.0, 1.0, 0.0]
    
    # Far (-1.0) -> Green
    mask_far = (contacts_frame == -1.0)
    colors[mask_far] = [0.0, 1.0, 0.0]
    
    return colors