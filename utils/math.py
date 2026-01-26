import numpy as np

def normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalizes a given input array to unit length.

    Args:
        x: Input array of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized array of shape (N, dims).
    """
    norm = np.linalg.norm(x, axis=-1, ord=2, keepdims=True)
    norm = np.clip(norm, a_min=eps, a_max=None)
    return x / norm

def quat_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalizes a quaternion array.

    Args:
        x: Quaternion array. Shape is (..., 4).
        eps: Epsilon to prevent numerical instabilities. Defaults to 1e-8.

    Returns:
        The normalized quaternion array. Shape is (..., 4).
    """
    res = normalize(x, eps=eps)
    return res

def quat_from_angle_axis(angle_axis: np.ndarray) -> np.ndarray:
    """Convert rotations given as angle-axis to quaternions.
    
    The angle-axis representation: the length of the vector is the angle,
    and the direction is the rotation axis.

    Args:
        angle_axis: Angle-axis representation. Shape is (..., 3).
                    The norm of the vector is the angle in radians,
                    and the normalized direction is the rotation axis.

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4).
    """
    angles = np.linalg.norm(angle_axis, axis=-1, keepdims=True)  # (..., 1)
    axes = normalize(angle_axis)  # (..., 3)
    
    theta = angles / 2  # (..., 1)
    xyz = axes * np.sin(theta)  # (..., 3)
    w = np.cos(theta)  # (..., 1)
    quat = np.concatenate([w, xyz], axis=-1)  # (..., 4)
    return normalize(quat)

def _sqrt_positive_part(x: np.ndarray) -> np.ndarray:
    """Returns sqrt(max(0, x)) but with a zero sub-gradient where x is 0.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L91-L99
    """
    ret = np.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = np.sqrt(x[positive_mask])
    return ret


def quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: The rotation matrices. Shape is (..., 3, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L102-L161
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = np.unstack(matrix.reshape(batch_dim + (9,)), axis=-1)

    q_abs = _sqrt_positive_part(
        np.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = np.stack(
        [
            np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = np.array(0.1, dtype=q_abs.dtype)
    q_abs_max = np.maximum(q_abs[..., np.newaxis], flr)
    quat_candidates = quat_by_rijk / (2.0 * q_abs_max)

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    best_indices = np.argmax(q_abs, axis=-1)[..., np.newaxis, np.newaxis]  # (..., 1, 1)
    return np.take_along_axis(quat_candidates, best_indices, axis=-2).squeeze(axis=-2)

def matrix_from_quat(quaternions: np.ndarray) -> np.ndarray:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = np.unstack(quaternions, axis=-1)
    two_s = 2.0 / (quaternions * quaternions).sum(axis=-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_from_quat(quat: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = np.linalg.norm(quat[..., 1:], axis=-1)
    half_angle = np.arctan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = np.where(
        np.abs(angle) > eps, np.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles[..., np.newaxis]

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes are not compatible for broadcasting.
    """
    # Ensure last dimension is 4 (quaternion dimension)
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        msg = f"Last dimension must be 4 for quaternions. Got shapes: {q1.shape} and {q2.shape}."
        raise ValueError(msg)
    
    # Use numpy broadcasting to handle shape mismatches
    # This will automatically broadcast compatible shapes
    try:
        q1_broadcast, q2_broadcast = np.broadcast_arrays(q1, q2)
    except ValueError as e:
        msg = f"Cannot broadcast quaternion shapes: {q1.shape} and {q2.shape}."
        raise ValueError(msg) from e
    
    # Store the broadcasted shape for output
    shape = q1_broadcast.shape
    
    # reshape to (N, 4) for multiplication
    q1 = q1_broadcast.reshape(-1, 4)
    q2 = q2_broadcast.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([w, x, y, z], axis=-1).reshape(shape)

def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = np.cross(xyz, vec, axis=-1) * 2
    return (vec + quat[:, 0:1] * t + np.cross(xyz, t, axis=-1)).reshape(shape)

def quat_between(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes quaternion rotations between two 3D-vector arrays.

    Args:
        x: Array of 3D vectors. Shape is (..., 3).
        y: Array of 3D vectors. Shape is (..., 3).

    Returns:
        Array of quaternions in (w, x, y, z). Shape is (..., 4).
    """
    res = np.concatenate(
        [
            np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis]
            + np.sum(x * y, axis=-1)[..., np.newaxis],
            np.cross(x, y),
        ],
        axis=-1,
    )
    return res

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1).reshape(shape)


def quat_inv(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Computes the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        The inverse quaternion in (w, x, y, z). Shape is (..., 4).
    """
    return quat_conjugate(q) / np.clip(np.sum(q**2, axis=-1, keepdims=True), a_min=eps, a_max=None)

def quat_fk(lrot: np.ndarray, lpos: np.ndarray, parents: list) -> tuple:
    """Performs Forward Kinematics (FK) on local quaternions and local positions.

    Computes global representations from local quaternions and local positions
    using the parent-child relationships defined in the kinematic tree.

    Args:
        lrot: Array of local quaternions. Shape is (..., num_joints, 4).
        lpos: Array of local positions. Shape is (..., num_joints, 3).
        parents: List of parent indices for each joint. Root joint has parent -1.

    Returns:
        Tuple of (global_quaternions, global_positions):
            - global_quaternions: Shape is (..., num_joints, 4).
            - global_positions: Shape is (..., num_joints, 3).
    """
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            quat_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        gr.append(quat_mul(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
    return res

def quat_ik(grot: np.ndarray, gpos: np.ndarray, parents: list) -> tuple:
    """Performs Inverse Kinematics (IK) on global quaternions and global positions.

    Computes local representations from global quaternions and global positions
    using the parent-child relationships defined in the kinematic tree.

    Args:
        grot: Array of global quaternions. Shape is (..., num_joints, 4).
        gpos: Array of global positions. Shape is (..., num_joints, 3).
        parents: List of parent indices for each joint. Root joint has parent -1.

    Returns:
        Tuple of (local_quaternions, local_positions):
            - local_quaternions: Shape is (..., num_joints, 4).
            - local_positions: Shape is (..., num_joints, 3).
    """
    res = [
        np.concatenate(
            [
                grot[..., :1, :],
                quat_mul(quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
            ],
            axis=-2,
        ),
        np.concatenate(
            [
                gpos[..., :1, :],
                quat_apply(
                    quat_inv(grot[..., parents[1:], :]),
                    gpos[..., 1:, :] - gpos[..., parents[1:], :],
                ),
            ],
            axis=-2,
        ),
    ]

    return res

def rotate_at_frame_w_obj(
    X: np.ndarray,
    Q: np.ndarray,
    obj_x: np.ndarray,
    obj_q: np.ndarray,
    trans2joint_list: np.ndarray,
    parents: list,
    n_past: int = 1,
    floor_z: bool = False,
) -> tuple:
    """Re-orients the animation data according to the last frame of past context.

    Rotates the character and object poses so that the character's forward
    direction at the key frame (n_past-1) aligns with a reference direction.

    Args:
        X: Array of local positions. Shape is (batch_size, timesteps, num_joints, 3).
        Q: Array of local quaternions. Shape is (batch_size, timesteps, num_joints, 4).
        obj_x: Array of object positions. Shape is (batch_size, timesteps, 3).
        obj_q: Array of object quaternions. Shape is (batch_size, timesteps, 4).
        trans2joint_list: Array of translation offsets from object to joint.
            Shape is (batch_size, 3).
        parents: List of parent indices for each joint. Root joint has parent -1.
        n_past: Number of frames in the past context. Defaults to 1.
        floor_z: If True, floor is on z-axis (project to xy plane).
            If False, floor is on y-axis (project to xz plane). Defaults to False.

    Returns:
        Tuple of (rotated_X, rotated_Q, rotated_obj_x, rotated_obj_q):
            - rotated_X: Rotated local positions. Shape is (batch_size, timesteps, num_joints, 3).
            - rotated_Q: Rotated local quaternions. Shape is (batch_size, timesteps, num_joints, 4).
            - rotated_obj_x: Rotated object positions. Shape is (batch_size, timesteps, 3).
            - rotated_obj_q: Rotated object quaternions. Shape is (batch_size, timesteps, 4).
    """
    # Get global quats and global poses (FK)
    global_q, global_x = quat_fk(Q, X, parents)

    key_glob_Q = global_q[:, n_past - 1 : n_past, 0:1, :]  # (B, 1, 1, 4)
    if floor_z: 
        # The floor is on z = xxx. Project the forward direction to xy plane. 
        forward = np.array([1, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :] * quat_apply(
            key_glob_Q, np.array([1, 0, 0])[np.newaxis, np.newaxis, np.newaxis, :]
        ) # In rest pose, x direction is the body left direction, root joint point to left hip joint.  
    else: 
        # The floor is on y = xxx. Project the forward direction to xz plane. 
        forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] * quat_apply(
            key_glob_Q, np.array([1, 0, 0])[np.newaxis, np.newaxis, np.newaxis, :]
        ) # In rest pose, x direction is the body left direction, root joint point to left hip joint.  
        # forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
        #     key_glob_Q, np.array([0, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :]
        # ) # In rest pose, z direction is forward direction. This also works. 

    forward = normalize(forward)
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
    new_glob_Q = quat_mul(quat_inv(yrot), global_q)
    new_glob_X = quat_apply(quat_inv(yrot), global_x)

    # Process object rotation and translation 
    # new_obj_x = quat_mul_vec(quat_inv(yrot[:, 0, :, :]), obj_x)
    # print(yrot[:, 0, :, :].shape)       1, 1, 4
    # print(obj_q.shape)                  1, T, 4
    # print(quat_inv(yrot[:, 0, :, :]).shape) 1, 1, 4
    new_obj_q = quat_mul(quat_inv(yrot[:, 0, :, :]), obj_q)

    # Apply corresponding rotation to the object translation 
    obj_trans = obj_x + trans2joint_list[:, np.newaxis, :] # N X T X 3  
    obj_trans = quat_apply(quat_inv(yrot[:, 0, :, :]), obj_trans) # N X T X 3
    obj_trans = obj_trans - trans2joint_list[:, np.newaxis, :] # N X T X 3 
    new_obj_x = obj_trans.copy()  

    # back to local quat-pos
    Q, X = quat_ik(new_glob_Q, new_glob_X, parents)
    
    return X, Q, new_obj_x, new_obj_q