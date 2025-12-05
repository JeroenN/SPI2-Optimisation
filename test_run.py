
from main import *

print(jax.default_backend())

def test_run1():
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

    components = [c0, c1, c2]
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


def test_run2():
    lr = 0.005
    component_folder = Path(__file__).parent / "files" / "components"
    component_files = [f for f in component_folder.rglob("*.csv") if f.is_file()]
    components = load_components(component_files)
    rotation_params, translation_params = create_random_params(len(components))
    #translation_params = jnp.array([[-2.6277635, 0.00518192, -0.48108062],[ 2.1418629,  -0.00431631, -0.11753592]])
    #rotation_params = jnp.array([[-1.5624535,  -0.00696707,  0.00822425],[ 1.8177378,  -1.5606427,  -1.8292603 ]])

    params = {'rotation': rotation_params, 'translation': translation_params}

    transformed_components = transform_components(components, params)

    optimizer = optax.adam(learning_rate= lr, b1=0.8, b2=0.95)
    opt_state = optimizer.init(params)
    sgd_step = make_sgd_step(optimizer)

    for step in range(10_000):
        params = sgd_step(params,opt_state, components)[0]
        
        if step % 50 == 0:
            transformed_components = transform_components(components, params)
            volume = volume_loss(transformed_components)
            collision = component_collision_constraint_new(transformed_components)

            print(f"\nSTEP {step}, volume {volume}, collision {collision}")
            print(f"translation: \n {params['translation']},\n rotation: \n {params['rotation']}\n")
            visualize(transformed_components, f"visualization_{step}")

test_run2()