import numpy as np
import trimesh
import open3d as o3d
import open3d.core as o3c  # type: ignore
import open3d.t.geometry as tgeo  # type: ignore
import networkx as nx

def generate_fine_grid(world_coord, spacing, fine_resolution=100):
    x_center, y_center, z_center = world_coord

    # Estimate the grid spacing (assumes uniform spacing in all directions)
    dx, dy, dz = spacing

    # Create a fine cube around the world_coord spanning one coarse voxel
    x0 = x_center - dx / 2
    x1 = x_center + dx / 2
    y0 = y_center - dy / 2
    y1 = y_center + dy / 2
    z0 = z_center - dz / 2
    z1 = z_center + dz / 2

    # Generate fine grid
    x_fine = np.linspace(x0, x1, fine_resolution)
    y_fine = np.linspace(y0, y1, fine_resolution)
    z_fine = np.linspace(z0, z1, fine_resolution)

    X_refined, Y_refined, Z_refined = np.meshgrid(x_fine, y_fine, z_fine, indexing='ij')
    fine_grid_points = np.vstack((X_refined.ravel(), Y_refined.ravel(), Z_refined.ravel())).T.astype(np.float32)

    return X_refined, Y_refined, Z_refined, fine_grid_points