import time
import viser
import viser.transforms as vtf

def visualize_with_viser(
    body_verts,     # T, V, 3
    body_faces,     # F, 3
    obj_verts,
    obj_faces,
    obj_trans,     # T, 3
    obj_rot,       # T, 3, 3
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
    )

    obj_handle = server.scene.add_mesh_simple(
        name="/object",
        vertices=obj_verts,
        faces=obj_faces,
        color=(0.2, 0.8, 0.2),
    )

    t = 0
    try:
        while True:
            body_handle.vertices = body_verts[t]

            obj_pos = obj_trans[t]
            obj_rotmat = obj_rot[t]
            obj_handle.position = obj_pos
            obj_handle.wxyz = vtf.SO3.from_matrix(obj_rotmat).wxyz

            t = (t+1) % T
            time.sleep(1 / fps)
    except KeyboardInterrupt:
        print("\n Viser server stopped.")