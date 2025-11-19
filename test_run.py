
from main import *

def test_run():
    colors = np.linspace(start = 0.2, stop = 1.0, num = 3)
    c0 = Component(
        sphere_centers=jnp.array([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0]]),
        sphere_radii=jnp.array([0.5, 0.5]),
        port_positions= jnp.array([[0.0, 0.0, 0.0]]),
        port_numbers= jnp.array([1]),
        color= colors[0]
    )

    c1 = Component(
        sphere_centers=jnp.array([[3.0, 0.0, 0.0],
                                [4.0, 0.0, 0.0]]),
        sphere_radii=jnp.array([0.5, 0.5]),
        port_positions= jnp.array([[0.0, 0.0, 0.0]]),
        port_numbers= jnp.array([1]),
        color= colors[1]
    )

    c2 = Component(
        sphere_centers=jnp.array([[0.0, 3.0, 0.0],
                                [0.0, 4.0, 0.0]]),
        sphere_radii=jnp.array([0.5, 0.5]),
        port_positions= jnp.array([[0.0, 0.0, 0.0]]),
        port_numbers= jnp.array([1]),
        color = colors[2]
    )

    components = (c0, c1, c2)
    lr = 1e-3
    
    rotation_params = jnp.zeros((len(components)-1,3))
    translation_params = jnp.zeros((len(components)-1,3))
    params = {'rotation': rotation_params, 'translation': translation_params}
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    for step in range(1000):
        params, loss = sgd_step(params, lr,optimizer,opt_state, components)
        if step % 1 == 0:
            print(f"step {step}, loss {loss:.6f}")
        
        if step % 100 == 0:
            transformed_components = transform_components(components, params)
            volume = volume_loss(transformed_components)
            collision = component_collision_constraint(transformed_components)

            print(f"\nSTEP {step}, volume {volume}, collision {collision}")
            print(f"translation: \n {params['translation']},\n rotation: \n {params['rotation']}\n")
            visualize(transformed_components, f"visualization_{step}")


test_run()