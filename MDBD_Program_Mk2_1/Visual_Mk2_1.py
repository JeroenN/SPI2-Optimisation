import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore
import networkx as nx
import matplotlib.pyplot as plt
import os
import csv

def load_csv_data(csv_path):
    centre_point = None
    spheres = []
    graph_edges = []
    section = None

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            if row[0].startswith('#'):
                if 'Center Point' in row[0]:
                    section = 'center'
                elif 'Sphere Data' in row[0]:
                    section = 'spheres'
                elif 'Tangent Sphere Pairs' in row[0]:
                    section = 'graph'
                continue
            if section == 'center' and 'CenterX' not in row[0]:
                centre_point = tuple(map(float, row))
            elif section == 'spheres' and 'SphereIndex' not in row[0]:
                center = tuple(map(float, row[1:4]))
                radius = float(row[4])
                spheres.append((center, radius))
            elif section == 'graph' and 'SphereA' not in row[0]:
                graph_edges.append((int(row[0]) - 1, int(row[1]) - 1))  # zero-based

    return centre_point, spheres, graph_edges

def create_sphere_meshes(spheres):
    meshes = []
    for center, radius in spheres:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.translate(center)
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    return meshes

def create_center_point_mesh(centre_point, radius=0.05):
    point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    point_sphere.translate(centre_point)
    point_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # red
    point_sphere.compute_vertex_normals()
    return point_sphere

def visualize_with_transparency(mesh, sphere_meshes, centre_point_mesh):
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Mesh and Spheres", 1024, 768)

    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene_widget)

    # Translucent material for base mesh
    translucent_material = rendering.MaterialRecord()
    translucent_material.shader = "defaultLitTransparency"
    translucent_material.base_color = [0.6, 0.6, 0.6, 0.3]

    scene_widget.scene.add_geometry("mesh", mesh, translucent_material)

    # Solid spheres
    solid_material = rendering.MaterialRecord()
    solid_material.shader = "defaultLit"
    solid_material.base_color = [0.1, 0.5, 0.8, 1.0]

    for i, sphere in enumerate(sphere_meshes):
        scene_widget.scene.add_geometry(f"sphere_{i}", sphere, solid_material)

    # Red point for center
    center_material = rendering.MaterialRecord()
    center_material.shader = "defaultLit"
    center_material.base_color = [1.0, 0.0, 0.0, 1.0]  # red
    scene_widget.scene.add_geometry("centre_point", centre_point_mesh, center_material)

    bbox = mesh.get_axis_aligned_bounding_box()
    scene_widget.setup_camera(60, bbox, bbox.get_center())

    gui.Application.instance.run()

def visualize_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title("Contact Graph of Tangent Spheres")
    plt.show()

if __name__ == "__main__":
    # === MODIFY THESE PATHS AS NEEDED ===
    stl_file_path = r'C:\Users\Thomas\OneDrive - TU Eindhoven\Desktop\Desktop\3. Graduation Project\cylinder.STL'
    csv_file_path = r'C:\Users\Thomas\OneDrive - TU Eindhoven\Desktop\Desktop\3. Graduation Project\Cylinder_R64_F128_S20.csv'

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    mesh.compute_vertex_normals()

    # Load sphere data and center
    centre_point, spheres_data, graph_edges = load_csv_data(csv_file_path)
    sphere_meshes = create_sphere_meshes(spheres_data)
    centre_point_mesh = create_center_point_mesh(centre_point, radius=0.01)

    # Visualize in 3D with transparency
    visualize_with_transparency(mesh, sphere_meshes, centre_point_mesh)

    # Visualize contact graph separately
    visualize_graph(graph_edges)