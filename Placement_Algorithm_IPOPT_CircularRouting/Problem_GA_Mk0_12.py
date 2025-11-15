from pymoo.core.problem import ElementwiseProblem
import numpy as np
from IPOPT_Mk0_12 import *

class GAProblem(ElementwiseProblem):
    def __init__(self, csv_files, connections, nr_control_points, alpha, routing_radius_a, routing_radius_b):
        self.connections = connections
        self.csv_files = csv_files
        self.nr_control_points = nr_control_points
        self.alpha = alpha
        self.routing_radius_a = routing_radius_a
        self.routing_radius_b = routing_radius_b

        n_objects = len(csv_files)
        n_connection_vars = len(connections) * 4 * nr_control_points
        n_object_vars = (n_objects - 1) * 6
        n_var = n_object_vars + n_connection_vars

        # Bounds for object-related variables
        xl = [-3.14, -3.14, -3.14, -20, -20, -20] * (n_objects - 1)
        xu = [3.14, 3.14, 3.14, 20, 20, 20] * (n_objects - 1)

        # Append bounds for connection-related variables
        xl += [-20] * n_connection_vars
        xu += [20] * n_connection_vars

        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=0,
            xl=xl,
            xu=xu
        )

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            x_opt, f_opt = solve_with_ipopt(np.array(x, dtype=float), self.csv_files, self.connections, self.nr_control_points, self.alpha, self.routing_radius_a, self.routing_radius_b)
            out["F"] = f_opt

        except Exception as e:
            print(f"[Warning] IPOPT failed for x = {x}, assigning penalty. Error: {e}")
            out["F"] = 1e6