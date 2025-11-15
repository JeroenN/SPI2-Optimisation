import numpy as np
import trimesh
import open3d as o3d
import open3d.core as o3c  # type: ignore
import open3d.t.geometry as tgeo  # type: ignore
import networkx as nx

def build_contact_graph(spheres, tolerance=1e-3):
    G = nx.Graph()
    
    # Add nodes with center and radius as attributes
    for i, (center, radius) in enumerate(spheres):
        G.add_node(i, center=center, radius=radius)

    # Add edges for tangent spheres
    for i in range(len(spheres)):
        for j in range(i + 1, len(spheres)):
            c1, r1 = spheres[i]
            c2, r2 = spheres[j]

            center_dist = np.linalg.norm(np.array(c1) - np.array(c2))
            expected_dist = r1 + r2

            if abs(center_dist - expected_dist) <= tolerance:
                G.add_edge(i, j)

    return G