import numpy as np
import trimesh
import open3d as o3d
import open3d.core as o3c        # type: ignore
import open3d.t.geometry as tgeo # type: ignore
from load_stl_Mk2_1 import * 

def create_grid_points(mesh, grid_size):
    min_corner = np.min(mesh.vertices, axis=0)
    max_corner = np.max(mesh.vertices, axis=0)

    # Generate grid coordinates
    x = np.linspace(min_corner[0], max_corner[0], grid_size)
    y = np.linspace(min_corner[1], max_corner[1], grid_size)
    z = np.linspace(min_corner[2], max_corner[2], grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T.astype(np.float32)

    # Compute center of the bounding box
    center = (min_corner + max_corner) / 2.0
    centre_point = center.astype(np.float32)

    # Compute spacing (assumes uniform spacing along each axis)
    spacing = (max_corner - min_corner) / (grid_size - 1)

    # Define Cartesian reference frame dictionary
    ref_frame = {
        "origin": center,
        "spacing": spacing
    }

    return X, Y, Z, grid_points, ref_frame, centre_point