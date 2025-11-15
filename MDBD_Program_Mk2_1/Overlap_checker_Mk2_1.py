import numpy as np
import trimesh
import open3d as o3d
import open3d.core as o3c  # type: ignore
import open3d.t.geometry as tgeo  # type: ignore
import networkx as nx

def check_no_overlap(center, radius, spheres, tolerance=1e-6):
    for existing_center, existing_radius in spheres:
        distance = np.linalg.norm(np.array(center) - np.array(existing_center))
        if distance < (radius + existing_radius) - tolerance:
            return False
    return True