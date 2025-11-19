import csv
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from dataclasses import dataclass
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List

from dataclasses import dataclass
import jax.numpy as jnp
from typing import Tuple

from visualize import visualize
from objects import Component
from objects import Sphere
from objects import Port

import optax

def euler_to_rotation_matrix(euler_angles: jnp.ndarray) -> jnp.ndarray:
    rx, ry, rz = euler_angles

    Rx = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(rx), -jnp.sin(rx)],
        [0, jnp.sin(rx),  jnp.cos(rx)]
    ])

    Ry = jnp.array([
        [ jnp.cos(ry), 0, jnp.sin(ry)],
        [ 0,           1, 0],
        [-jnp.sin(ry), 0, jnp.cos(ry)]
    ])

    Rz = jnp.array([
        [jnp.cos(rz), -jnp.sin(rz), 0],
        [jnp.sin(rz),  jnp.cos(rz), 0],
        [0,            0,           1]
    ])

    rotation_matrix = Rz @ Ry @ Rx
    return rotation_matrix


def load_components(csv_files: List[str]) -> Tuple[Component, ...]:
    components: List[Component] = []

    for file_path in csv_files:
        spheres: List[Sphere] = []
        ports: List[Port] = []

        with open(file_path, "r") as f:
            reader = csv.reader(f)
            mode = "spheres"

            for row in reader:
                if not row:
                    mode = "ports"
                    continue

                if mode == "spheres":
                    try:
                        x, y, z, r = map(float, row)
                        spheres.append(Sphere(center=jnp.array([x, y, z]), radius=r))
                    except ValueError:
                        continue
                elif mode == "ports":
                    try:
                        x, y, z, port_number = float(row[0]), float(row[1]), float(row[2]), int(row[3])
                        ports.append(Port(position=jnp.array([x, y, z]), port_number=port_number))
                    except ValueError:
                        continue

        comp = Component.from_objects(tuple(spheres), tuple(ports))
        components.append(comp)

    return tuple(components)

def volume_loss(components: Tuple[Component, ...]) -> float:
    all_centers = jnp.concatenate([comp.sphere_centers for comp in components], axis=0)  # shape (total_spheres,3)
    all_radii = jnp.concatenate([comp.sphere_radii for comp in components], axis=0)      # shape (total_spheres,)

    min_corner = jnp.min(all_centers - all_radii[:, None], axis=0)  # shape (3,)
    max_corner = jnp.max(all_centers + all_radii[:, None], axis=0)  # shape (3,)

    lengths = max_corner - min_corner  # shape (3,)

    volume = jnp.prod(lengths)
    return volume

def flatten_spheres(components: Tuple[Component, ...]):
    component_ids = jnp.concatenate([
    jnp.full(c.sphere_centers.shape[0], i, dtype=jnp.int32) 
    for i, c in enumerate(components)
    ])

    centers = jnp.concatenate([c.sphere_centers for c in components], axis=0)
    radii   = jnp.concatenate([c.sphere_radii for c in components], axis=0)
    return centers, radii, component_ids

def pairwise_signed_distances(centers, radii, component_ids):
    mask = component_ids[:, None] != component_ids[None, :]
    diff = centers[:, None, :] - centers[None, :, :]   # shape (N, N, 3)
    center_dist = jnp.linalg.norm(diff, axis=-1)       # (N, N)

    rad_sum = radii[:, None] + radii[None, :]          # (N, N)

    signed_distances = center_dist - rad_sum                   

    return jnp.min(signed_distances[mask])


def component_collision_constraint(components: Tuple[Component, ...]):
    signed_distances = []

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            ci = components[i]
            cj = components[j]

            delta = ci.sphere_centers[:, None, :] - cj.sphere_centers[None, :, :]
            dist = jnp.linalg.norm(delta, axis=-1)  # shape (Ni, Nj)

            radii_sum = ci.sphere_radii[:, None] + cj.sphere_radii[None, :]  # shape (Ni, Nj)

            signed_distances.append(dist - radii_sum)

    signed_distances = jnp.concatenate([d.flatten() for d in signed_distances])

    return 1/jnp.min(signed_distances) 



 # Transfroms the components according to the parameters
 # IMPORTANT: Skip the first component, this component always stays in place
def transform_components(components, params):
    R = jax.vmap(euler_to_rotation_matrix)(params['rotation'])

    transformed = []
    for idx, comp in enumerate(components):
        if idx == 0:
            transformed.append(comp)
        else:
            transformed.append(comp.transform(R[idx - 1], params['translation'][idx - 1]))
    return tuple(transformed)

def total_loss(params, components: Tuple[Component, ...], w_volume=1.0, w_component_collision=1.0):
    transformed_components = transform_components(components, params)
    volume = volume_loss(transformed_components)
    component_collision = component_collision_constraint(transformed_components)
    return w_volume * volume + w_component_collision * component_collision


def enforce_range_rotation_params(rotation_params: jnp.ndarray) -> jnp.ndarray:
    return (rotation_params + jnp.pi) % (2*jnp.pi) - jnp.pi


def sgd_step(params: dict, lr: float, optimizer: optax.adam, opt_state, components: Tuple[Component, ...]):
    loss, grads = jax.value_and_grad(total_loss)(params, components)

    updates, opt_state = optimizer.update(grads, opt_state, params)

    params = optax.apply_updates(params, updates)
    
    params['rotation'] = enforce_range_rotation_params(params['rotation'])
    return params, loss

def create_random_params(num_components):
    key = jax.random.PRNGKey(1)
    rotation_params = jax.random.uniform(key, shape=(num_components-1, 3), minval=-jnp.pi, maxval=jnp.pi)
    translation_params = jax.random.uniform(key, shape=(num_components-1, 3), minval=-10, maxval=10)
    return rotation_params, translation_params

def run():
    component_folder = Path(__file__).parent / "files" / "components"
    component_files = [f for f in component_folder.rglob("*.csv") if f.is_file()]
    components = load_components(component_files)
    # rotation_params, translation_params = create_random_params(len(components))

    # lr = 1e-5

    # for step in range(10):
    #     rotation_params, translation_params, loss = sgd_step(rotation_params, translation_params, lr, components)
    #     if step % 1 == 0:
    #         print(f"step {step}, loss {loss:.6f}")



