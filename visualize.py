import random
from pathlib import Path
from typing import Tuple

import numpy as np
import jax
from vedo import Sphere, Plotter, Mesh
from component import Component


def create_meshes_components(components: Tuple[Component, ...]):
    meshes_per_component = {}

    for idx, component in enumerate(components):
        centers = [
            np.asarray(jax.device_get(c), dtype=np.float64).reshape(3,)
            for c in component.sphere_centers
        ]
        radii = [float(jax.device_get(r)) for r in component.sphere_radii]
        meshes = []
        for center, radius in zip(centers, radii):
            if not (radius > 0 and np.isfinite(radius)):
                raise ValueError(f"Invalid sphere radius: {radius}")

            s = Sphere(r=radius, pos=center.tolist())
            s.c(component.color)
            meshes.append(s)

        meshes_per_component[idx] = meshes

    return meshes_per_component


def visualize(components: Tuple[Component, ...], image_name: str):
    meshes_per_component = create_meshes_components(components)
    outdir = Path(__file__).parent / "visualization"
    outdir.mkdir(exist_ok=True)
    image_path = outdir / image_name
    create_image(meshes_per_component, None, image_path)


def create_image(meshes_per_component, port_meshes, image_path, width=1920, height=1080):
    actors = []
    for meshes in meshes_per_component.values():
        actors.extend(meshes)

    p = Plotter(offscreen=True, size=(width, height), bg="white")
    for a in actors:
        p.add(a)

    # Center camera on first component
    first_component_meshes = meshes_per_component[0]
    first_vertices = np.vstack([np.asarray(m.vertices) for m in first_component_meshes])
    center = first_vertices.mean(axis=0)

    all_vertices = np.vstack([np.asarray(m.vertices) for meshes in meshes_per_component.values() for m in meshes])
    center = all_vertices.mean(axis=0)

    # Compute bounding box and camera distance
    bbox_min = all_vertices.min(axis=0)
    bbox_max = all_vertices.max(axis=0)
    bbox_extent = bbox_max - bbox_min
    max_extent = np.linalg.norm(bbox_extent)

    camera_distance = max_extent * 1.5  # 1.5 gives a bit of padding
    p.camera.SetFocalPoint(center)

    # Define spherical angles for a nice 3D view
    azimuth_deg = 45   # rotation around vertical axis
    elevation_deg = 30 # rotation up/down

    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    # Compute offset in Cartesian coordinates
    x = camera_distance * np.cos(el) * np.cos(az)
    y = camera_distance * np.cos(el) * np.sin(az)
    z = camera_distance * np.sin(el)

    # Set camera position and view-up
    p.camera.SetPosition(center + np.array([x, y, z]))
    p.camera.SetViewUp([0, 0, 1])  # Z-up

    p.show(resetcam=False)
    p.screenshot(str(image_path))
    p.close()
