import numpy as np
from pymoo.core.problem import Problem
from casadi import MX, Function, jacobian
from casadi_placement_Volume_Mk0_12 import * 

class MyProblemCasadi(Problem):
    def __init__(self, csv_files):
        self.csv_files = csv_files

        n_objects = len(self.csv_files)
        n_var = (n_objects - 1) * 6

        x = MX.sym("x", n_var)
        f, g, g_list, FW_Objects = casadi_placement_Volume(x, self.csv_files)

        self.f_func = Function("f", [x], [f])
        self.g_func = Function("g", [x], [g])
        self.df_func = Function("df", [x], [jacobian(f, x)])
        self.dg_func = Function("dg", [x], [jacobian(g, x)])

        xl = [-3.14, -3.14, -3.14, -20, -20, -20] * (n_objects - 1)
        xu = [3.14, 3.14, 3.14, 20, 20, 20] * (n_objects - 1)

        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_constr=len(g_list),
                         xl=xl,
                         xu=xu,
                         elementwise_evaluation=False)


    def _evaluate(self, x, out, *args, **kwargs):
        F = []
        G = []

        for xi in x:
            xi_col = xi.reshape((-1, 1))
            f_val = self.f_func(xi_col).full().flatten()
            g_val = self.g_func(xi_col).full().flatten()

            G.append([-(g_val)])
            F.append(f_val)

        out["F"] = np.array(F)
        out["G"] = np.array(G)