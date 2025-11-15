import numpy as np
import trimesh
import open3d as o3d
import open3d.core as o3c  # type: ignore
import open3d.t.geometry as tgeo  # type: ignore
import networkx as nx

def compute_signed_distances(scene, points, grid_size, device):
    points_tensor = o3c.Tensor(points, dtype=o3c.Dtype.Float32, device=device)
    signed_dists_tensor = scene.compute_signed_distance(points_tensor)
    signed_dists_raw = signed_dists_tensor.cpu().numpy().reshape((grid_size, grid_size, grid_size))
    # Normalize -0 to 0
    signed_distance_invert = - signed_dists_raw
    signed_dists_no_negzero = np.where(signed_distance_invert == -0.0, 0.0, signed_distance_invert)
    # Set positive values to np.inf
    signed_dists_processed = np.where(signed_dists_no_negzero < -1e-6, -np.inf, signed_dists_no_negzero)
    return signed_dists_processed