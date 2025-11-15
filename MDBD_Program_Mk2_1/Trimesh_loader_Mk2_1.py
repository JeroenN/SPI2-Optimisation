import numpy as np
import trimesh
import open3d as o3d
import open3d.core as o3c        # type: ignore
import open3d.t.geometry as tgeo # type: ignore
from load_stl_Mk2_1 import * 

def load_mesh_trimesh(file_path):
    mesh = trimesh.load_mesh(file_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = next(iter(mesh.geometry.values()))
    return mesh