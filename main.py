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
from component import Component

import vedo

import optax
import time

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
    colors = ["lightblue", "yellow", "red", "purple", "green", "orange"]
    for idx, file_path in enumerate(csv_files):
        sphere_centers = []
        sphere_radii = []
        port_positions = []
        port_numbers = []

        with open(file_path, "r") as f:
            reader = csv.reader(f)
            mode = "spheres"

            for row in reader:
                if not row:
                    mode = "ports"
                    continue

                if mode == "spheres":
                        x, y, z, r = map(float, row)
                        sphere_centers.append([x,y,z])
                        sphere_radii.append(r)

                elif mode == "ports":
                        x, y, z, port_number = float(row[0]), float(row[1]), float(row[2]), int(row[3])
                        port_positions.append([x,y,z])
                        port_numbers.append(port_number)

        comp = Component(jnp.array(sphere_centers), jnp.array(sphere_radii), jnp.array(port_positions), jnp.array(port_numbers), colors[idx])
        components.append(comp)

    return tuple(components)

def volume_loss(components: Tuple[Component, ...]) -> float:
    all_centers = jnp.concatenate([comp.sphere_centers for comp in components], axis=0)  
    all_radii = jnp.concatenate([comp.sphere_radii for comp in components], axis=0)    

    min_corner = jnp.min(all_centers - all_radii[:, jnp.newaxis], axis=0)  
    max_corner = jnp.max(all_centers + all_radii[:, jnp.newaxis], axis=0)  

    lengths = max_corner - min_corner  

    volume = jnp.prod(lengths)
    return volume

def component_collision_constraint_new(components: Tuple[Component, ...]):
    counts = jnp.array([comp.sphere_centers.shape[0] for comp in components])
    component_ids = jnp.repeat(jnp.arange(len(components)), counts)

    all_centers = jnp.concatenate([comp.sphere_centers for comp in components], axis=0) 
    all_radii = jnp.concatenate([comp.sphere_radii for comp in components], axis=0)
    # Basically gives all the pairs such that j > i
    N = all_centers.shape[0]
    idx_i, idx_j = jnp.triu_indices(N, k=1) 

    valid = component_ids[idx_i] != component_ids[idx_j]
    idx_i = idx_i[valid]
    idx_j = idx_j[valid]

    comps_i = all_centers[idx_i]
    comps_j = all_centers[idx_j]
    radii_i = all_radii[idx_i]
    radii_j = all_radii[idx_j]

    delta = comps_i - comps_j
    distance = jnp.linalg.norm(delta, axis=-1)
    radii_sum = radii_i + radii_j
    signed_distance = jnp.abs(distance) - radii_sum
    min = jnp.min(signed_distance)
    #if min <= 0:
    #    min = 1e-6
        
    return 1.0 / min



def component_collision_constraint(components: Tuple[Component, ...]):
    signed_distances = []

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            ci = components[i]
            cj = components[j]

            Ni = ci.sphere_centers.shape[0]
            Nj = cj.sphere_centers.shape[0]

            a = jnp.broadcast_to(ci.sphere_centers[:, jnp.newaxis, :], (Ni, Nj, 3))
            b = jnp.broadcast_to(cj.sphere_centers[jnp.newaxis, :, :], (Ni, Nj, 3))

            delta = a - b

            eclidian_dist = jnp.linalg.norm(delta, axis=-1)

            radii_sum = ci.sphere_radii[:, jnp.newaxis] + cj.sphere_radii[jnp.newaxis, :]

            signed_distances.append(jnp.abs(eclidian_dist - radii_sum))

    signed_distances = jnp.concatenate([d.flatten() for d in signed_distances])

    
    return 1/jnp.min(signed_distances)


 # IMPORTANT: Skips the first component, this component always stays in place
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
    component_collision = component_collision_constraint_new(transformed_components)
    return w_volume * volume + w_component_collision * component_collision


def enforce_range_rotation_params(rotation_params: jnp.ndarray) -> jnp.ndarray:
    return (rotation_params + jnp.pi) % (2*jnp.pi) - jnp.pi


def sgd_step(params: dict, lr: float, optimizer: optax.adam, opt_state, components: Tuple[Component, ...]):
    loss, grads = jax.value_and_grad(total_loss)(params, components)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    #params['rotation'] = enforce_range_rotation_params(params['rotation'])
    return params, loss

#TODO: make the radius based on the sizes of the components
def create_random_params(num_components, min_radius=3.0, max_radius=6.0):
    seed = int(time.time() * 1e6) % (2**32 - 1)
    key = jax.random.PRNGKey(seed)

    keys = jax.random.split(key, num_components - 1)

    rotation_params = jax.vmap(
        lambda k: jax.random.uniform(k, (3,), minval=-jnp.pi, maxval=jnp.pi)
    )(keys)

    def random_direction(k):
        v = jax.random.normal(k, (3,))
        return v / jnp.linalg.norm(v)

    dirs = jax.vmap(random_direction)(keys)

    radii = jax.vmap(
        lambda k: jax.random.uniform(k, (), minval=min_radius, maxval=max_radius)
    )(keys)

    translation_params = dirs * radii[:, jnp.newaxis]

    return rotation_params, translation_params

def run():
    component_folder = Path(__file__).parent / "files" / "components"
    component_files = [f for f in component_folder.rglob("*.csv") if f.is_file()]
    components = load_components(component_files)
    rotation_params, translation_params = create_random_params(len(components))

    lr = 1e-5

    for step in range(10):
        rotation_params, translation_params, loss = sgd_step(rotation_params, translation_params, lr, components)
        if step % 1 == 0:
            print(f"step {step}, loss {loss:.6f}")



