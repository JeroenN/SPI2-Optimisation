import numpy as np
import trimesh
import open3d as o3d
import open3d.core as o3c  # type: ignore
import open3d.t.geometry as tgeo  # type: ignore
import networkx as nx

def compute_combined_distance_field(mesh_signed_dists, X, Y, Z, spheres):
    dists_to_mesh = mesh_signed_dists.copy()
    if not spheres:
        return dists_to_mesh

    dist_to_spheres = np.full_like(dists_to_mesh, np.inf, dtype=np.float32)
    for cx, cy, cz, r in [(c[0], c[1], c[2], radius) for c, radius in spheres]:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) - r
        dist_to_spheres = np.minimum(dist_to_spheres, dist)

    combined_score = np.minimum(dists_to_mesh, dist_to_spheres)
    return combined_score