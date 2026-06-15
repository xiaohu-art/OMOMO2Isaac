"""Force-closure metrics for retargeted dexterous grasps.

Implements the Ferrari-Canny wrench-margin Q_FC used by SynManDex as an
admission score (not a physical guarantee):

    Q_FC(q) = min_{||w||2=1} max_{f in F(q), ||f||1<=1}  w^T G(q) f

Contacts are modelled with an m-sided linearized Coulomb friction cone; the
grasp map G stacks per-contact force/torque about the object centre of mass.
Q_FC > 0 certifies force closure (origin strictly inside the grasp wrench set).

Conventions:
    - Contact normals `N` must point INTO the object (the direction the finger
      pushes). trimesh vertex normals point outward, so pass `-vertex_normal`.
    - Torque is scaled by a characteristic length L so force/torque share units.
"""

import numpy as np
import torch
from scipy.spatial import ConvexHull


def friction_cone_edges(n: np.ndarray, mu: float, m: int = 8) -> np.ndarray:
    """Linearize a Coulomb friction cone into `m` edge force directions.

    Args:
        n: (3,) unit inward normal.
        mu: friction coefficient (cone half-angle = atan(mu)).
        m: number of polyhedral edges.

    Returns:
        (m, 3) edge force vectors (unit normal component + mu tangent).
    """
    n = n / (np.linalg.norm(n) + 1e-9)
    # Stable tangent basis orthogonal to n.
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = a - (a @ n) * n
    t1 /= np.linalg.norm(t1) + 1e-9
    t2 = np.cross(n, t1)
    ang = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    return n[None] + mu * (np.cos(ang)[:, None] * t1[None]
                           + np.sin(ang)[:, None] * t2[None])  # (m, 3)


def grasp_wrench_set(P: np.ndarray, N: np.ndarray, com: np.ndarray,
                     mu: float = 0.5, m: int = 8, L: float = None,
                     soft_finger: bool = True, mu_t: float = 0.1) -> np.ndarray:
    """Build the discretized grasp wrench set.

    Args:
        P: (K, 3) contact positions.
        N: (K, 3) inward contact normals.
        com: (3,) object centre of mass.
        mu, m: friction params.
        L: characteristic length for torque scaling (default: max |p - com|).
        soft_finger: model fingertips as soft contacts that resist torsion about
            the contact normal (essential — wrap/power grasps are NOT full force
            closure under a hard point-contact model).
        mu_t: torsional friction coefficient (soft-finger moment bound).

    Returns:
        (K * m [+ 2K], 6) primitive wrenches [force(3); scaled_torque(3)].
    """
    if L is None:
        L = float(np.linalg.norm(P - com, axis=1).max()) + 1e-6
    W = []
    for p, n in zip(P, N):
        E = friction_cone_edges(n, mu, m)        # (m, 3) forces
        tau = np.cross(p - com, E) / L           # (m, 3) scaled torques
        W.append(np.concatenate([E, tau], axis=1))  # (m, 6)
        if soft_finger:                          # +/- torsional moment about the normal
            nn = n / (np.linalg.norm(n) + 1e-9)
            for sgn in (1.0, -1.0):
                W.append(np.concatenate([np.zeros(3), sgn * mu_t * nn / L])[None])
    return np.concatenate(W, axis=0)             # (K * m [+ 2K], 6)


def q_fc(P: np.ndarray, N: np.ndarray, com: np.ndarray,
         mu: float = 0.5, m: int = 8, min_contacts: int = 3,
         soft_finger: bool = True, mu_t: float = 0.1) -> float:
    """Ferrari-Canny wrench margin. >0 iff the grasp is force-closure.

    Returns the signed radius of the largest origin-centred ball inscribed in
    the grasp wrench set (negative / -1.0 when not force-closure or degenerate).
    """
    P = np.asarray(P, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    if len(P) < min_contacts:                    # 6D closure needs >=3 non-collinear contacts
        return -1.0
    W = grasp_wrench_set(P, N, com, mu, m, soft_finger=soft_finger, mu_t=mu_t)
    try:
        hull = ConvexHull(W, qhull_options="QJ")  # QJ joggle avoids coplanar-degeneracy failures
    except Exception:
        return -1.0
    b = hull.equations[:, -1]                     # facet offsets: a_i.x + b_i <= 0, ||a_i||=1
    # Distance from origin to facet i = -b_i; all <=0 iff origin interior.
    return float(-b.max())


# =============================================================================
# Differentiable proxy (for optimization, SynManDex Eq 6 L_FC = -Q_FC)
# =============================================================================

def make_wrench_dirs(D: int = 64, device="cpu", seed: int = 0) -> torch.Tensor:
    """Fixed set of D unit test wrench directions in R^6 (for the min over w)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(D, 6, generator=g)
    return (w / w.norm(dim=1, keepdim=True)).to(device)


def fc_margin_batch(P: torch.Tensor, N: torch.Tensor, com: torch.Tensor,
                    contact_w: torch.Tensor, w_dirs: torch.Tensor,
                    mu: float = 0.5, m: int = 8, L: float = None,
                    beta: float = 20.0, neg: float = 1e4,
                    soft_finger: bool = True, mu_t: float = 0.1) -> torch.Tensor:
    """Differentiable per-frame force-closure margin (soft Ferrari-Canny).

    Approximates  Q_FC = min_{||w||=1} max_{f admissible} w^T G f  with a fixed
    set of test directions and soft min/max (logsumexp). Differentiable w.r.t.
    contact positions P (hence joint angles via FK).

    Args:
        P: (T, K, 3) contact positions (robot fingertips, differentiable).
        N: (T, K, 3) INWARD contact normals (object surface, detached).
        com: (T, 3) object centre of mass (detached).
        contact_w: (T, K) in {0, 1} — which fingers are in contact this frame.
        w_dirs: (D, 6) fixed unit test wrench directions.
        mu, m: friction coefficient and cone discretization.
        L: characteristic length for torque scaling (default: batch max |p-com|).
        beta: softness of min/max (larger = closer to hard, sharper gradient).
        neg: large value to mask out non-contact fingers from the max.

    Returns:
        (T,) approximate Q_FC per frame (>0 ~ force closure).
    """
    Tn, K = P.shape[0], P.shape[1]
    n = N / (N.norm(dim=-1, keepdim=True) + 1e-9)
    # Stable tangent basis orthogonal to n.
    a = torch.where(n[..., :1].abs() < 0.9,
                    n.new_tensor([1.0, 0.0, 0.0]).expand_as(n),
                    n.new_tensor([0.0, 1.0, 0.0]).expand_as(n))
    t1 = a - (a * n).sum(-1, keepdim=True) * n
    t1 = t1 / (t1.norm(dim=-1, keepdim=True) + 1e-9)
    t2 = torch.cross(n, t1, dim=-1)
    ang = torch.arange(m, device=P.device, dtype=P.dtype) * (2.0 * np.pi / m)
    cos = torch.cos(ang).view(1, 1, m, 1)
    sin = torch.sin(ang).view(1, 1, m, 1)
    E = n.unsqueeze(2) + mu * (cos * t1.unsqueeze(2) + sin * t2.unsqueeze(2))  # (T,K,m,3)
    arm = P - com.unsqueeze(1)                                                 # (T,K,3)
    if L is None:
        L = arm.norm(dim=-1).max().detach() + 1e-6
    tau = torch.cross(arm.unsqueeze(2).expand_as(E), E, dim=-1) / L            # (T,K,m,3)
    Wp = torch.cat([E, tau], dim=-1)                                           # (T,K,m,6)
    if soft_finger:  # +/- torsional moment about the normal (resists twist)
        zero = torch.zeros_like(n).unsqueeze(2)                                # (T,K,1,3)
        twist = (mu_t * n / L).unsqueeze(2)                                    # (T,K,1,3)
        soft = torch.cat([
            torch.cat([zero, twist], dim=-1),
            torch.cat([zero, -twist], dim=-1),
        ], dim=2)                                                             # (T,K,2,6)
        Wp = torch.cat([Wp, soft], dim=2)                                      # (T,K,m+2,6)
    proj = torch.einsum("df,tkmf->tdkm", w_dirs, Wp)                           # (T,D,K,m[+2])
    # Mask out non-contact fingers: their primitives can never win the max.
    proj = proj + (1.0 - contact_w.clamp(0, 1)).view(Tn, 1, K, 1) * (-neg)
    proj = proj.reshape(Tn, w_dirs.shape[0], -1)
    support = torch.logsumexp(beta * proj, dim=2) / beta                       # (T,D) ~ max_f
    qhat = -torch.logsumexp(-beta * support, dim=1) / beta                     # (T,)  ~ min_w
    return qhat
