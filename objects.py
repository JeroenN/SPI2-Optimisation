
import jax
from typing import Tuple
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass(frozen=True)
class Sphere:
    center: jnp.ndarray 
    radius: float

@dataclass(frozen=True)
class Port:
    position: jnp.ndarray
    port_number: int

@dataclass(frozen=True)
class Component:
    sphere_centers: jnp.ndarray   # shape (N,3)
    sphere_radii: jnp.ndarray     # shape (N,)
    port_positions: jnp.ndarray   # shape (M,3)
    port_numbers: jnp.ndarray     # shape (M,)
    color: float

    @classmethod
    def from_objects(cls, spheres: Tuple[Sphere, ...], ports: Tuple[Port, ...]) -> "Component":
        sphere_centers = jnp.stack([s.center for s in spheres]) if spheres else jnp.zeros((0, 3))
        sphere_radii = jnp.array([s.radius for s in spheres]) if spheres else jnp.zeros((0,))
        port_positions = jnp.stack([p.position for p in ports]) if ports else jnp.zeros((0, 3))
        port_numbers = jnp.array([p.port_number for p in ports], dtype=int) if ports else jnp.zeros((0,), dtype=int)
        return cls(
            sphere_centers=sphere_centers,
            sphere_radii=sphere_radii,
            port_positions=port_positions,
            port_numbers=port_numbers,
            color = cls.color
        )
    
    def transform(self, rotation_matrix: jnp.ndarray, translation_vector: jnp.ndarray) -> "Component":
        new_centers = (self.sphere_centers @ rotation_matrix.T) + translation_vector
        new_positions = (self.port_positions @ rotation_matrix.T) + translation_vector

        return Component(
            sphere_centers=new_centers,
            sphere_radii=self.sphere_radii,
            port_positions=new_positions,
            port_numbers=self.port_numbers,
            color = self.color
        )

    def to_objects(self) -> Tuple[Tuple[Sphere, ...], Tuple[Port, ...]]:
        spheres = tuple(Sphere(center=c, radius=r) for c, r in zip(self.sphere_centers, self.sphere_radii))
        ports = tuple(Port(position=p, port_number=int(n)) for p, n in zip(self.port_positions, self.port_numbers))
        return spheres, ports
