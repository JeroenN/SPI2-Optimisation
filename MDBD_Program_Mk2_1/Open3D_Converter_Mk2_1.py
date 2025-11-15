import numpy as np
import trimesh
import open3d as o3d
import open3d.core as o3c        # type: ignore
import open3d.t.geometry as tgeo # type: ignore
from load_stl_Mk2_1 import * 

def convert_to_open3d(mesh_trimesh, device):
    vertices = o3c.Tensor(np.asarray(mesh_trimesh.vertices), dtype=o3c.Dtype.Float32, device=device)
    triangles = o3c.Tensor(np.asarray(mesh_trimesh.faces), dtype=o3c.Dtype.Int32, device=device)
    mesh_o3d_t = tgeo.TriangleMesh(device=device)
    mesh_o3d_t.vertex["positions"] = vertices
    mesh_o3d_t.triangle["indices"] = triangles
    return mesh_o3d_t