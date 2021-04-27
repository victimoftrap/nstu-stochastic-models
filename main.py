from labs.lab4 import optimal_plan
from labs.lab5 import dual_procedure_optimal_plan

import numpy as np


if __name__ == '__main__':
    N = 3

    s = 2
    p1 = -1
    p2 = 0.5
    F = np.array([[-0.8, 1], [p1, 0]])
    psi = np.array([[p2], [1]])
    H = np.array([1, 0])
    R = [[0.1]]
    x0 = np.array([[0], [0]])

    U = 2
    U_bounds = [0, 10]

    F_derivs = [np.array([[0, 0], [1, 0]]),
                np.array([[0, 0], [0, 0]])]
    psi_derivs = [np.array([[0], [0]]),
                  np.array([[1], [0]])]
    H_derivs = [np.array([0, 0]),
                np.array([0, 0])]
    R_derivs = [[[0]], [[0]]]
    x0_derivs = [np.array([[0], [0]]),
                 np.array([[0], [0]])]

    # optimal_plan(N, s, F, psi, H, R, x0, U_bounds, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs)
    dual_procedure_optimal_plan('A', N, s, F, psi, H, R, x0, U_bounds, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs)
