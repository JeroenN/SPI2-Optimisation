from casadi import nlpsol
from casadi import MX, Function, jacobian
from casadi_placement_Volume_Mk0_12 import *
from casadi import inf
from casadi import *

def solve_with_ipopt(x0, csv_files, connections, nr_control_points, alpha, routing_radius_a, routing_radius_b):
    n_objects = len(csv_files)
    n_connection_vars = len(connections) * 4 * nr_control_points
    n_var = (n_objects - 1) * 6 + n_connection_vars

    x = MX.sym("x", n_var)
    volume, g, g_list, FW_Objects, routing, routing_length, bbox = casadi_placement_Volume(x, csv_files, connections, nr_control_points, alpha, routing_radius_a, routing_radius_b)
    
    # Objective Function
    f = volume + routing_length

    n_constraints = len(g_list)

    # IPOPT problem setup
    nlp = {
        'x': x,
        'f': f,
        'g': g
    }

    solver = nlpsol("solver", "ipopt", nlp, {
        "ipopt.print_level": 0,
        "print_time": False,
        "ipopt.tol": 1e-6,
        "ipopt.max_iter": 1000
    })

    # Bounds for object-related variables
    lbx = [-3.14, -3.14, -3.14, -20, -20, -20] * (n_objects - 1)
    ubx = [3.14, 3.14, 3.14, 20, 20, 20] * (n_objects - 1)

    lbx += [-20] * n_connection_vars
    ubx += [20] * n_connection_vars

    # Constraint bounds: distances >= 0 (i.e., no overlap)
    lbg = [0.0] * n_constraints
    ubg = [inf] * n_constraints

    res = solver(x0=x0,
                 lbx=lbx,
                 ubx=ubx,
                 lbg=lbg,
                 ubg=ubg)

    x_opt = res["x"].full().flatten()
    f_opt = res["f"].full().item()

    status = solver.stats()["return_status"]
    iter_count = solver.stats()["iter_count"]

    print(f"Status: {status}")
    print(f"iter_count: {iter_count}")
    print(f"Volume: {f_opt}")

    return x_opt, f_opt