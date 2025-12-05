
import jax
from typing import Tuple
import jax.numpy as jnp
from dataclasses import dataclass
from jax import ShapeDtypeStruct

@dataclass(frozen=True)
class Component:
    sphere_centers: jnp.ndarray   
    sphere_radii: jnp.ndarray     
    port_positions: jnp.ndarray   
    port_numbers: jnp.ndarray     
    color: str

    def transform(self, rotation_matrix: jnp.ndarray, translation_vector: jnp.ndarray) -> "Component":
        center_component = self.sphere_centers.mean(axis=0)
        new_centers = ((self.sphere_centers - center_component) @ rotation_matrix.T) + center_component + translation_vector 
        new_positions = ((self.port_positions  - center_component) @ rotation_matrix.T) + center_component + translation_vector

        return Component(
            sphere_centers=new_centers,
            sphere_radii=self.sphere_radii,
            port_positions=new_positions,
            port_numbers=self.port_numbers,
            color = self.color
        )

