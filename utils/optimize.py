import numpy as np
import pickle as pkl
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from .cli_args import DEVICE, MODELS_ROOT

def load_grab_prior(root_path):
    root = Path(root_path)
    lhand_path = root / 'lh_prior.pkl'
    rhand_path = root / 'rh_prior.pkl'
    with open(lhand_path, 'rb') as f:
        lhand_data = pkl.load(f)
    with open(rhand_path, 'rb') as f:
        rhand_data = pkl.load(f)
    return lhand_data, rhand_data


def grab_prior(root_path):
    lhand_data, rhand_data = load_grab_prior(root_path)

    prior = np.concatenate([lhand_data['mean'], rhand_data['mean']], axis=0)
    lhand_prec = lhand_data['precision']
    rhand_prec = rhand_data['precision']

    return prior, lhand_prec, rhand_prec

class HandPrior:
    HAND_POSE_NUM=45
    def __init__(self, prior_path,
                 prefix=66,
                 device=DEVICE,
                 dtype=torch.float,
                 type='grab'):
        "prefix is the index from where hand pose starts, 66 for SMPL-H"
        self.prefix = prefix
        if type == 'grab':
            prior, lhand_prec, rhand_prec = grab_prior(prior_path)
            self.mean = torch.tensor(prior, dtype=dtype).unsqueeze(axis=0).to(device)
            self.lhand_prec = torch.tensor(lhand_prec, dtype=dtype).unsqueeze(axis=0).to(device)
            self.rhand_prec = torch.tensor(rhand_prec, dtype=dtype).unsqueeze(axis=0).to(device)
        else:
            raise NotImplemented("Only grab hand prior is supported!")

    def __call__(self, full_pose,left_or_right):
        "full_pose also include body poses, this function can be used to compute loss"
        #print(full_pose.shape,'FP')
        if left_or_right==0:
            temp = full_pose[:, :] - self.mean[:,:45]
            lhand = torch.matmul(temp[:, :], self.lhand_prec)
            return lhand
        else:
            temp = full_pose[:, :] - self.mean[:,45:]
            rhand = torch.matmul(temp[:, :], self.rhand_prec)
            return rhand

# =============================================================================
# Configuration & Constants
# =============================================================================

class HandConfig:
    """Configuration for hand optimization parameters and paths."""
    
    # Path configuration
    HAND_MEAN_TITLE = 'behave'
    BASE_EXP_PATH = str(Path(MODELS_ROOT) / 'exp')
    ASSETS_PATH = str(Path(MODELS_ROOT) / 'assets')
    
    # Optimization hyperparameters
    LR = 0.001
    EPOCHS = 1000
    
    # Contact and collision thresholds
    THRESH_CLOSE = 0.02
    THRESH_FAR = 0.20
    PENETRATION_LIMIT = 0.03
    
    # Range of Motion (ROM) Limits [Max, Min]
    # Defined as class attributes to avoid repeated creation in loops
    ROM_GROUP_1_MAX = [1.10, 0.09, 0.13]
    ROM_GROUP_1_MIN = [-0.8, -0.08, -0.2]
    
    ROM_GROUP_2_MAX = [1.10, 0.15, 0.12]
    ROM_GROUP_2_MIN = [-0.1, -0.10, -0.15]
    
    ROM_GROUP_3_MAX = [1.10, 0.15, 0.10]
    ROM_GROUP_3_MIN = [-0.1, -0.10, -0.35]
    
    # Pinky & Thumb Specifics
    ROM_PINKY_MAX = [1.10, 0.5, 1.10]
    ROM_PINKY_MIN = [[-0.8,-0.5,-0.8], [-0.4,-0.5,-0.8], [-0.5,-0.5,-0.8]]
    
    ROM_THUMB_MAX = [[0.45,0.45,1.5], [0.45,0.45,-0.1], [0.45,0.45,1.5]]
    ROM_THUMB_MIN = [[-0.5,-0.5,-0.2], [-0.5,-0.5,-0.8], [-0.5,-0.5,-0.8]]

    @staticmethod
    def to_tensor(data):
        """Convert data to torch tensor on the configured device."""
        return torch.tensor(data, dtype=torch.float32).to(DEVICE)

# =============================================================================
# Resource Manager
# =============================================================================

class HandResources:
    """Loading and caching manager for static resource files (.npy, indices, etc.)."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HandResources, cls).__new__(cls)
            cls._instance.loaded = False
        return cls._instance

    def load_if_needed(self):
        """Load all resources if not already loaded."""
        if self.loaded:
            return

        print("Loading Hand Optimization Resources...")
        base = Path(HandConfig.BASE_EXP_PATH)
        idx_base = base / 'index_778'

        def load_indices_list(prefix_path):
            """Helper function to load a list of index files."""
            lst = []
            prefix = Path(prefix_path)
            for i in range(5):
                for j in range(3):
                    p = prefix.with_name(f"{prefix.name}_{i}_{j}.npy")
                    if p.exists():
                        lst.append(np.load(str(p)))
            p_end = prefix.with_name(f"{prefix.name}_5.npy")
            if p_end.exists():
                lst.append(np.load(str(p_end)))
            return lst

        # Load hand indices
        self.r_idx_det = load_indices_list(idx_base / 'rhand_index' / 'hand_778')
        self.r_small_idx = load_indices_list(idx_base / 'rhand_small_index' / 'hand_778_small')
        self.l_idx_det = load_indices_list(idx_base / 'lhand_index' / 'lhand_778')
        self.l_small_idx = load_indices_list(idx_base / 'lhand_small_index' / 'lhand_778_small')

        # Load hand means
        self.r_mean = np.load(str(base / f'{HandConfig.HAND_MEAN_TITLE}_rhand_mean.npy'))
        self.l_mean = np.load(str(base / f'{HandConfig.HAND_MEAN_TITLE}_lhand_mean.npy'))

        # Load SMPL indices
        self.r_smplx_idx = np.load(str(base / 'rhand_smplx_ids.npy'))
        self.l_smplx_idx = np.load(str(base / 'lhand_smplx_ids.npy'))

        # Load prior model
        self.prior_model = HandPrior(prior_path=str(Path(HandConfig.ASSETS_PATH)), device=DEVICE)

        # Pre-convert ROM tensors
        self.rom_tensors = {
            'g1_max': HandConfig.to_tensor(HandConfig.ROM_GROUP_1_MAX),
            'g1_min': HandConfig.to_tensor(HandConfig.ROM_GROUP_1_MIN),
            'g2_max': HandConfig.to_tensor(HandConfig.ROM_GROUP_2_MAX),
            'g2_min': HandConfig.to_tensor(HandConfig.ROM_GROUP_2_MIN),
            'g3_max': HandConfig.to_tensor(HandConfig.ROM_GROUP_3_MAX),
            'g3_min': HandConfig.to_tensor(HandConfig.ROM_GROUP_3_MIN),
            'pinky_max': HandConfig.to_tensor(HandConfig.ROM_PINKY_MAX),
            'pinky_min': HandConfig.to_tensor(HandConfig.ROM_PINKY_MIN),
            'thumb_max': HandConfig.to_tensor(HandConfig.ROM_THUMB_MAX),
            'thumb_min': HandConfig.to_tensor(HandConfig.ROM_THUMB_MIN),
        }
        
        self.loaded = True

# =============================================================================
# Core Optimization
# =============================================================================

class HandOptimizer:
    def __init__(self, smpl_model, frame_times, poses, betas, trans, 
                 obj_data):
        self.res = HandResources()
        self.res.load_if_needed()
        
        self.smpl_model = smpl_model
        self.T = frame_times
        self.device = DEVICE
        
        # Human Inputs
        self.body_pose = torch.from_numpy(poses[:, 3:66]).float().to(self.device)
        self.root_orient = torch.from_numpy(poses[:, :3]).float().to(self.device)
        self.trans = torch.from_numpy(trans).float().to(self.device)
        self.betas = torch.from_numpy(betas[None, :]).repeat(self.T, 1).float().to(self.device)
        
        # Object Inputs
        self.obj_verts_world, self.obj_normals_world = self._prepare_object(obj_data)
        
        # Initialize Hand Params (Optimized Variable)
        l_mean = torch.from_numpy(self.res.l_mean).float().reshape(1, -1, 3).repeat(self.T, 1, 1).to(self.device)
        r_mean = torch.from_numpy(self.res.r_mean).float().reshape(1, -1, 3).repeat(self.T, 1, 1).to(self.device)
        
        init_pose = torch.cat([l_mean, r_mean], dim=1) # [T, 30, 3] (15 joints * 2 hands)
        self.hand_pose_param = init_pose.clone().detach().requires_grad_(True)
        
        self.optimizer = optim.Adam([self.hand_pose_param], lr=HandConfig.LR)

    def _prepare_object(self, obj_data):
        """Prepare object vertices and normals in world space"""
        rot = torch.from_numpy(obj_data['rot']).float().to(self.device)
        trans = torch.from_numpy(obj_data['trans']).float().to(self.device)
        verts = torch.from_numpy(obj_data['verts']).float().to(self.device)
        faces = torch.tensor(obj_data['faces']).float().unsqueeze(0).repeat(self.T, 1, 1).to(self.device)
        
        world_verts = obj_forward_torch(verts, rot, trans)
        world_normals = compute_vertex_normals(world_verts, faces)
        return world_verts, world_normals

    def run(self, epochs=1000):
        # 1. Calculate Initial Contact Masks (Pre-optimization)
        with torch.no_grad():
            masks = self._compute_contact_masks(self.hand_pose_param)
        
        # 2. Optimization Loop
        pbar = tqdm(range(epochs), desc="Optimizing Hand")
        for epoch in pbar:
            self.optimizer.zero_grad()
            
            # Forward SMPL
            smpl_out = self.smpl_model(
                pose_body=self.body_pose,
                pose_hand=self.hand_pose_param.reshape(-1, 90),
                betas=self.betas,
                root_orient=self.root_orient,
                trans=self.trans
            )
            
            # Left Hand Loss
            l_loss, l_info = self._compute_single_hand_loss(
                smpl_out, is_right=False, masks=masks['left'], epoch=epoch
            )
            # Right Hand Loss
            r_loss, r_info = self._compute_single_hand_loss(
                smpl_out, is_right=True, masks=masks['right'], epoch=epoch
            )
            
            total_loss = l_loss + r_loss
            total_loss.backward()
            self.optimizer.step()
            
            # Logging
            if epoch % 10 == 0:
                pbar.set_postfix({
                    'L_Coll': f"{l_info['coll']:.2f}",
                    'L_Attr': f"{l_info['attr']:.2f}",
                    'R_Coll': f"{r_info['coll']:.2f}",
                    'R_Attr': f"{r_info['attr']:.2f}",
                })
                
        return self.hand_pose_param.detach().reshape(self.T, 90).cpu().numpy()

    def _compute_contact_masks(self, hand_params):
        """Compute contact weights for both hands once at initialization"""
        smpl_out = self.smpl_model(
            pose_body=self.body_pose,
            pose_hand=hand_params.reshape(-1, 90),
            betas=self.betas,
            root_orient=self.root_orient,
            trans=self.trans
        )
        v = smpl_out.v
        
        # Right Hand
        wr, wr_lin, wr_opt = calculate_contact_masks(
            v[:, self.res.r_smplx_idx], self.obj_verts_world, self.obj_normals_world
        )
        # Left Hand
        wl, wl_lin, wl_opt = calculate_contact_masks(
            v[:, self.res.l_smplx_idx], self.obj_verts_world, self.obj_normals_world
        )
        
        return {
            'right': {'strict': wr, 'linear': wr_lin, 'opt': wr_opt},
            'left':  {'strict': wl, 'linear': wl_lin, 'opt': wl_opt}
        }

    def _compute_single_hand_loss(self, smpl_out, is_right, masks, epoch):
        """Calculate all loss components for one hand"""
        # Select resources
        if is_right:
            indices_det = self.res.r_idx_det
            small_indices = self.res.r_small_idx
            smpl_idx = self.res.r_smplx_idx
            param_offset = 15 # Right hand is the second half (indices 15-30)
            euler_flip = torch.tensor([1.0, 1.0, 1.0]).to(self.device)
            mean_target = torch.from_numpy(self.res.r_mean).to(self.device).reshape(15, 3)
        else:
            indices_det = self.res.l_idx_det
            small_indices = self.res.l_small_idx
            smpl_idx = self.res.l_smplx_idx
            param_offset = 0  # Left hand is the first half (indices 0-15)
            euler_flip = torch.tensor([-1.0, -1.0, 1.0]).to(self.device)
            mean_target = torch.from_numpy(self.res.l_mean).to(self.device).reshape(15, 3)

        # Extract hand data
        h_verts = smpl_out.v[:, smpl_idx]
        h_params = self.hand_pose_param[:, param_offset:param_offset+15, :] # [T, 15, 3]

        # Weights
        w_touch_lin = masks['linear']
        
        # 1. SDF Calculation
        _, signed_dist, _, = point2point_signed(h_verts, self.obj_verts_world, self.obj_normals_world)
        
        loss_coll = torch.tensor(0.0, device=self.device)
        loss_attr = torch.tensor(0.0, device=self.device)

        # 2. Region-based Collision & Attraction
        for i in range(len(indices_det)):
            # Collision: Punish penetration
            sd_region = signed_dist[:, indices_det[i]]
            
            # Penetration mask: dist < 0
            mask_pen = (sd_region < 0)
            
            # Only punish if frame needs optimization
            # Logic borrowed from original: if any vertex penetrates in frame
            mask_frame_pen = mask_pen.any(dim=-1)
            
            if mask_frame_pen.any():
                pen_vals = sd_region[mask_frame_pen]
                # Filter noise
                valid_pen = (pen_vals < 0) & (pen_vals.abs() < HandConfig.PENETRATION_LIMIT)
                
                # Weight by contact probability
                frame_weights = w_touch_lin[mask_frame_pen]
                
                # Mean per penetrating vertex, then sum
                num_pen = valid_pen.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
                loss_coll += torch.sum(pen_vals.abs() * valid_pen.float() * frame_weights / num_pen) * 2.0

            # Attraction: Pull "clean" frames closer
            # Original logic: If NO penetration in frame, pull specific points (small_indices) to surface
            mask_clean = ~mask_frame_pen
            if mask_clean.any():
                sd_clean = signed_dist[mask_clean][:, small_indices[i]]
                clean_weights = w_touch_lin[mask_clean]
                # loss_attr += 5.0 * torch.sum((sd_clean.abs() ** 2) * clean_weights) # Apply weight? 
                # Original code multiplied SDF by w_touch_l before slicing.
                loss_attr += 5.0 * torch.sum((sd_clean * clean_weights).abs()**2)

        # 3. Range of Motion (ROM)
        # Flip Euler angles for left hand to match right-hand constraints
        euler = h_params[:, :, [2, 1, 0]] * euler_flip.view(1, 1, 3)
        roms = self.res.rom_tensors
        
        loss_rom = torch.tensor(0.0, device=self.device)
        loss_rom += restrict_angles_loss(euler[:, [0, 3, 9]], roms['g1_max'], roms['g1_min'])
        loss_rom += restrict_angles_loss(euler[:, [1, 2, 4, 5]], roms['g2_max'], roms['g2_min'])
        loss_rom += restrict_angles_loss(euler[:, [10, 11]], roms['g3_max'], roms['g3_min'])
        loss_rom += restrict_angles_loss(euler[:, [6, 7, 8]], roms['pinky_max'], roms['pinky_min'])
        loss_rom += restrict_angles_loss(euler[:, 12:15], roms['thumb_max'], roms['thumb_min'])
        
        # 4. Regularization (Temporal & Mean Pose)
        loss_reg = torch.tensor(0.0, device=self.device)
        if epoch > 100:
            # Smoothness
            loss_reg += smooth_loss(h_params, h_verts) * 0.5
            
            # Regress to Mean Pose (when not touching)
            # h_params: [T, 15, 3], mean_target: [15, 3], w_touch_lin: [T, 1]
            loss_reg += 0.05 * torch.sum((h_params - mean_target.unsqueeze(0))**2 * (1.0 - w_touch_lin).unsqueeze(-1))
            
        # 5. Prior Model (VPoser/GrabPrior)
        if epoch > 200:
            # Flatten to [Batch, 45] (15 joints * 3)
            flat_pose = h_params.contiguous().view(-1, 45)
            # Prior model expects specific format. 
            prior_loss = self.res.prior_model(flat_pose, left_or_right=int(is_right))
            loss_reg += 0.1 * torch.sum(prior_loss**2 * masks['opt'])

        total = loss_coll + loss_attr + loss_rom + loss_reg
        info = {
            'coll': loss_coll.item(),
            'attr': loss_attr.item(),
            'rom': loss_rom.item(),
            'reg': loss_reg.item()
        }
        return total, info

def obj_forward_torch(obj_verts, obj_rot, obj_trans):
    """Batched Object Transformation"""
    if obj_verts.dim() == 2:
        obj_verts = obj_verts.unsqueeze(0)
    # (B, N, 3) @ (B, 3, 3)^T + (B, 1, 3)
    return torch.matmul(obj_verts, obj_rot.transpose(1, 2)) + obj_trans.unsqueeze(1)

def compute_vertex_normals(vertices, faces):
    """Compute vertex normals from faces"""
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    # print((torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None].shape)
    # print(faces.shape,"FACES")# expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]
    # print(vertices_faces.shape,"VERTICES FACES")

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                                   vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                                   vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                                   vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

def point2point_signed(x, y, y_normals=None):
    """
    Compute signed distance from point cloud x to y.
    Uses pure PyTorch implementation without external dependencies.
    return: (dist, signed_dist, idx_near)
    """
    N, P1, D = x.shape
    P2 = y.shape[1]
    
    # Compute pairwise distances: (N, P1, P2)
    # x: (N, P1, D), y: (N, P2, D)
    # cdist computes distances between each pair: (N, P1, P2)
    dist_matrix = torch.cdist(x, y)  # (N, P1, P2)
    
    # Find nearest point indices: (N, P1)
    idx_near = torch.argmin(dist_matrix, dim=2)  # (N, P1)
    
    # Gather actual closest points
    idx_expanded = idx_near.view(N, P1, 1).expand(N, P1, D).long()
    y_near_x = y.gather(1, idx_expanded)  # Points on surface y closest to x
    
    x2y_vec = x - y_near_x
    dist = x2y_vec.norm(dim=2)
    
    signed_dist = dist
    if y_normals is not None:
        # Dot product with normal to determine sign
        y_nn = y_normals.gather(1, idx_expanded)
        # (N, P1, 3) dot (N, P1, 3) -> (N, P1)
        dot = (y_nn * x2y_vec).sum(dim=-1)
        sign = dot.sign()
        signed_dist = dist * sign

    return dist, signed_dist, idx_near

def calculate_contact_masks(hand_verts, obj_verts, obj_normals):
    """
    Calculate contact masks based on distance (Heuristic Contact Masks)
    Returns:
        w_strict (Tensor): Strict contact mask (Binary)
        w_linear (Tensor): Linear transition mask (0.0 - 1.0)
        w_opt (Tensor): Valid optimization region mask (Binary)
    """
    _, h2o_signed, _ = point2point_signed(hand_verts, obj_verts, y_normals=obj_normals)
    
    # Frame-wise minimum distance
    min_dist, _ = torch.min(h2o_signed, dim=1) # [T]
    
    mask_close = min_dist <= HandConfig.THRESH_CLOSE
    mask_far = min_dist > HandConfig.THRESH_FAR
    mask_mid = (~mask_close) & (~mask_far)
    
    # Linear interpolation for mid range
    # 1.0 at Close, 0.0 at Far
    k = 1.0 / (HandConfig.THRESH_CLOSE - HandConfig.THRESH_FAR)
    b = -HandConfig.THRESH_FAR * k
    
    w_linear = torch.zeros_like(min_dist)
    w_linear[mask_close] = 1.0
    w_linear[mask_mid] = min_dist[mask_mid] * k + b
    
    w_strict = mask_close.float().reshape(-1, 1)
    w_linear = w_linear.detach().reshape(-1, 1)
    w_opt = (~mask_far).float().detach().reshape(-1, 1)
    
    return w_strict, w_linear, w_opt

@torch.jit.script
def restrict_angles_loss(theta, theta_max, theta_min):
    diff_max = (theta - theta_max).clamp(min=0)
    diff_min = (theta_min - theta).clamp(min=0)
    return torch.sum(diff_max ** 2) + torch.sum(diff_min ** 2)

@torch.jit.script
def smooth_loss(params, verts):
    # Params Smoothness
    diff1 = params[1:] - params[:-1]
    loss = 0.5 * torch.sum(diff1 ** 2)

    # Vertex Smoothness
    v_diff1 = verts[1:] - verts[:-1]
    loss += 0.5 * torch.sum(v_diff1 ** 2)
    
    if params.shape[0] > 2:
        diff2 = diff1[1:] - diff1[:-1]
        loss += 0.25 * torch.sum(diff2 ** 2)
    
        v_diff2 = v_diff1[1:] - v_diff1[:-1]
        loss += 0.25 * torch.sum(v_diff2 ** 2)

    return loss

def optimize_hand(
    frame_times,
    poses,
    betas,
    trans,
    smpl_model,
    obj_rot,
    obj_trans,
    obj_verts,
    obj_faces,
    epochs=1000,
    lr=0.001
):
    """
    Wrapper function to maintain backward compatibility.
    Instantiates the HandOptimizer and runs it.
    """
    # Override config with arguments
    HandConfig.EPOCHS = epochs
    HandConfig.LR = lr
    
    obj_data = {
        'rot': obj_rot,
        'trans': obj_trans,
        'verts': obj_verts,
        'faces': obj_faces
    }
    
    optimizer = HandOptimizer(
        smpl_model, frame_times, poses, betas, trans, obj_data
    )
    
    return optimizer.run(epochs=epochs)