import os
os.environ["OPEN3D_RENDER_ENGINE"] = "osmesa"
import open3d as o3d
import csv
import random
import numpy as np
from Runner_GA_Mk0_12 import csv_filename


def load_optimized_csv(csv_path):
    objects = {}
    ports = []
    routing = []
    bbox = None

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

        for i, row in enumerate(rows):
            if not row:
                sphere_end_idx = i
                break
            if row[0] == 'Object':
                continue
            obj_index = int(row[0])
            x, y, z = map(float, row[2:5])
            r = float(row[5])
            if obj_index not in objects:
                objects[obj_index] = []
            objects[obj_index].append(((x, y, z), r))

        port_start_idx = sphere_end_idx + 2
        port_end_idx = port_start_idx
        for i in range(port_start_idx, len(rows)):
            if not rows[i]:
                port_end_idx = i
                break
            if len(rows[i]) < 6 or rows[i][0] == 'Object':
                continue
            x, y, z = map(float, rows[i][2:5])
            ports.append((x, y, z))

        for row in rows[port_end_idx + 2:]:
            if len(row) < 5:
                continue
            if row[0] == 'Bounding Box':
                break
            point_type = row[1]
            x, y, z = map(float, row[2:5])
            if point_type == "Start":
                routing.append([[x, y, z]])
            elif point_type.startswith("Control") or point_type == "End":
                routing[-1].append([x, y, z])

        for i in range(len(rows) - 6, len(rows)):
            row = rows[i]
            if row and row[0] in ['Min X', 'Max X', 'Min Y', 'Max Y', 'Min Z', 'Max Z']:
                val = float(row[1])
                if bbox is None:
                    bbox = []
                bbox.append(val)

    return objects, ports, routing, bbox


def create_sphere_meshes_by_object(objects):
    meshes_by_object = {}
    for obj_index, spheres in objects.items():
        meshes = []
        for center, radius in spheres:
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            mesh.translate(center)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(random_color_rgb())
            meshes.append(mesh)
        meshes_by_object[obj_index] = meshes
    return meshes_by_object


def create_port_meshes(ports, radius=0.05):
    port_meshes = []
    for pos in ports:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.paint_uniform_color([1.0, 0.0, 0.0])
        mesh.translate(pos)
        mesh.compute_vertex_normals()
        port_meshes.append(mesh)
    return port_meshes


def random_color_rgb():
    return [random.uniform(0.2, 1.0) for _ in range(3)]


def create_cylinder_between(p1, p2, radius=0.05, color=[1.0, 0.5, 0.0]):
    p1 = np.array(p1)
    p2 = np.array(p2)
    direction = p2 - p1
    height = np.linalg.norm(direction)

    if height < 1e-6:
        return None

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color(color)

    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, direction)
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_axis, direction / height))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cylinder.rotate(R, center=np.zeros(3))

    midpoint = (p1 + p2) / 2
    cylinder.translate(midpoint)

    return cylinder


def create_routing_meshes(routing, radius=0.05):
    routing_meshes = []
    for path in routing:
        color = random_color_rgb()
        for i in range(len(path) - 1):
            cyl = create_cylinder_between(path[i], path[i + 1], radius, color)
            if cyl:
                routing_meshes.append(cyl)
    return routing_meshes


def create_bounding_box_lineset(bbox):
    min_x, max_x, min_y, max_y, min_z, max_z = bbox
    points = [
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z],
    ]
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    colors = [[0.0, 1.0, 0.0] for _ in lines]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visualize_spheres(meshes_by_object, port_meshes, routing_meshes, bbox_values):
    geometries = []

    for meshes in meshes_by_object.values():
        geometries.extend(meshes)

    geometries.extend(port_meshes)
    geometries.extend(routing_meshes)

    if bbox_values:
        geometries.append(create_bounding_box_lineset(bbox_values))

    # Uses legacy visualizer which works fine on AMD
    o3d.visualization.draw(geometries)


if __name__ == "__main__":
    csv_file_path = csv_filename

    objects, ports, routing, bbox = load_optimized_csv(csv_file_path)
    meshes_by_object = create_sphere_meshes_by_object(objects)
    port_meshes = create_port_meshes(ports)
    routing_meshes = create_routing_meshes(routing)

    visualize_spheres(meshes_by_object, port_meshes, routing_meshes, bbox)
