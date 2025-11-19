import open3d as o3d
import random
from objects import Component
from objects import Sphere
from objects import Port
from typing import Tuple
from pathlib import Path
import numpy as np
import jax
import open3d as o3d
import numpy as np
import jax


def random_color_rgb():
    return [random.uniform(0.2, 1.0) for _ in range(3)]



def create_meshes_components(components: Tuple[Component, ...]):
    meshes_per_component = {}

    
    for idx, component in enumerate(components):

        centers = [np.asarray(jax.device_get(c), dtype=np.float64).reshape(3,) 
                   for c in component.sphere_centers]

        radii = [float(jax.device_get(r)) for r in component.sphere_radii]

        meshes = []

        for center, radius in zip(centers, radii):

            # Validate radius
            if not (radius > 0 and np.isfinite(radius)):
                raise ValueError(f"Invalid sphere radius: {radius}")

            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))

            # Validate mesh
            if mesh is None or len(mesh.vertices) == 0:
                raise RuntimeError("Sphere creation failed.")

            mesh.translate(center.tolist())   # safe: python list of 3 floats
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(random_color_rgb())
            meshes.append(mesh)

        meshes_per_component[idx] = meshes

    return meshes_per_component


def create_meshes_ports(ports, radius=0.05):
    port_meshes = []
    for pos in ports:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.paint_uniform_color([1.0, 0.0, 0.0])
        mesh.translate(pos)
        mesh.compute_vertex_normals()
        port_meshes.append(mesh)
    return port_meshes

def visualize(components: Tuple[Component, ...], image_name: str):
    meshes_per_component = create_meshes_components(components)
    dir = Path(__file__).parent / "visualization"
    dir.mkdir(exist_ok=True)
    image_path = dir / image_name
    create_image(meshes_per_component, None, image_path)

def create_image(
    meshes_per_component,
    port_meshes,
    image_path,
    width=1920,
    height=1080
):
    # Collect geometries
    geometries = []
    for meshes in meshes_per_component.values():
        geometries.extend(meshes)

    #geometries.extend(port_meshes)

    # Create offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])   # white background (optional)

    # Add geometries
    for i, g in enumerate(geometries):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        scene.add_geometry(f"geom{i}", g, mat)

    # Set up camera automatically
    bounds = scene.bounding_box
    center = bounds.get_center()
    extent = bounds.get_extent().max()
    scene.camera.look_at(center, center + [0, 0, extent], [0, 1, 0])

    # Render and save
    img = renderer.render_to_image()
    o3d.io.write_image(image_path, img)