from labs.fisher_information import compute_fisher_information

from collections import namedtuple
import random

import numpy as np
from scipy.optimize import minimize, minimize_scalar

delta = 0.01

PlanElement = namedtuple("PlanElement", ["u", "p"])


def __x_a_criteria__(fisher_by_plan):
    val = np.trace(np.linalg.inv(fisher_by_plan))
    # print(f"X[M(ksi)] A-optimal: {val}")
    return val


def __x_d_criteria__(fisher_by_plan):
    val = - np.log(np.linalg.det(fisher_by_plan))
    # print(f"X[M(ksi)] D-optimal: {val}")
    return val


def __compute_fisher_by_plan__(plan, N, s, F, psi, H, R, x0, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs):
    fisher_plan = np.zeros((s, s))
    for plan_point in plan:
        fisher_zero = compute_fisher_information(
            N, s, F, psi, H, R, x0, plan_point.u, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
        )
        fisher_plan += plan_point.p * fisher_zero
    return fisher_plan


def __next_epsilon_plan__(current_plan, new_point):
    next_plan = []
    plan_coeff = new_point.p / len(current_plan)
    for point in current_plan:
        next_plan.append(PlanElement(point.u, point.p - plan_coeff))
    next_plan.append(new_point)
    return next_plan


def __clean_up_plan__(plan):
    without_repeats = [list(plan[0])]
    for i in range(1, len(plan)):
        next_point = plan[i]
        merged = False
        for el in without_repeats:
            u_in_plan = el[0]
            if all(np.isclose(u_in_plan[j], next_point.u[j]) for j in range(len(u_in_plan))):
                el[1] += next_point.p
                merged = True
                break

        if not merged:
            without_repeats.append(list(next_point))

    without_zero_weights = []
    accum = 0
    for el in without_repeats:
        if el[1] < delta:
            accum += el[1]
        else:
            without_zero_weights.append(el)

    for el in without_zero_weights:
        el[1] += accum / len(without_zero_weights)
    return list(map(lambda point: PlanElement(point[0], point[1]), without_zero_weights))


def __default_plan__(s, N):
    q = int((s * (s + 1) / 2) + 1)
    default_plan = []
    for i in range(q):
        eps_u = [1 for i in range(N)]
        eps_p = 1 / q
        default_plan.append(PlanElement(eps_u, eps_p))
    return default_plan


def dual_procedure_optimal_plan(
        optimality_criterion, initial_plan,
        N, s, F, psi, H, R, x0, u_bounds, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
):
    def __mju_a__(u_of_point):
        fisher_plan = __compute_fisher_by_plan__(
            current_plan, N, s, F, psi, H, R, x0, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
        )
        f_plan_inverted = np.linalg.inv(fisher_plan)

        fisher_by_u_of_point = compute_fisher_information(
            N, s, F, psi, H, R, x0, u_of_point, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
        )
        return np.trace(f_plan_inverted @ f_plan_inverted @ fisher_by_u_of_point)

    def __mju_a_for_maximization__(u_of_point):
        return - __mju_a__(u_of_point)

    def __eta_a__(plan_star):
        fisher_plan = np.zeros((s, s))
        for j in range(len(plan_star)):
            fisher = compute_fisher_information(
                N, s, F, psi, H, R, x0, plan_star[j].u, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
            )
            fisher_plan += plan_star[j].p * fisher
        return np.trace(np.linalg.inv(fisher_plan))

    def __x_a__(tau):
        probably_plan = __next_epsilon_plan__(current_plan, PlanElement(u_maximized, tau))
        return __x_a_criteria__(
            __compute_fisher_by_plan__(
                probably_plan, N, s, F, psi, H, R, x0, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
            )
        )

    def __mju_d__(u_of_point):
        fisher_plan = __compute_fisher_by_plan__(
            current_plan, N, s, F, psi, H, R, x0, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
        )
        f_plan_inverted = np.linalg.inv(fisher_plan)

        fisher_by_u_of_point = compute_fisher_information(
            N, s, F, psi, H, R, x0, u_of_point, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
        )
        return np.trace(f_plan_inverted @ fisher_by_u_of_point)

    def __mju_d_for_maximization(u_of_point):
        return - __mju_d__(u_of_point)

    def __eta_d__(plan_star):
        return s

    def __x_d__(tau):
        probably_plan = __next_epsilon_plan__(current_plan, PlanElement(u_maximized, tau))
        return __x_d_criteria__(
            __compute_fisher_by_plan__(
                probably_plan, N, s, F, psi, H, R, x0, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
            )
        )

    if optimality_criterion == 'A':
        maximization_mju = __mju_a_for_maximization__
        compute_mju = __mju_a__
        compute_eta = __eta_a__
        compute_x = __x_a__
    else:
        maximization_mju = __mju_d_for_maximization
        compute_mju = __mju_d__
        compute_eta = __eta_d__
        compute_x = __x_d__
    print(f"Двойственная процедура построения непрерывного {optimality_criterion}- оптимального плана\n")

    epsilon_zero_plan = initial_plan if initial_plan is not None else __default_plan__(s, N)
    print(f"Начальный невырожденный план:\n{epsilon_zero_plan}\n")

    current_plan = epsilon_zero_plan
    k = 0
    while True:
        print(f"План на {k} итерации:\n{current_plan}")
        u_value = np.array([random.uniform(u_bounds[0], u_bounds[1]) for i in range(N)])
        u_max_result = minimize(
            maximization_mju, u_value,
            method='SLSQP',
            bounds=np.array([u_bounds for i in range(N)]),
        )
        u_maximized = u_max_result.x

        mju_value = compute_mju(u_maximized)
        eta_value = compute_eta(current_plan)
        if abs(mju_value - eta_value) <= delta:
            break
        if mju_value > eta_value:
            tau_k_result = minimize_scalar(
                compute_x,
                method='bounded',
                bounds=np.array([0, 1]),
            )
            tau_minimized = tau_k_result.x

            new_plan_point = PlanElement(u_maximized, tau_minimized)
            print(f"Новая точка плана:\n{new_plan_point}")
            next_plan = __next_epsilon_plan__(current_plan, new_plan_point)
            print(f"Новый план:\n{next_plan}")
            cleaned_plan = __clean_up_plan__(next_plan)
            print(f"Очищенный план:\n{cleaned_plan}\n")

            k += 1
            current_plan = cleaned_plan

    print(f"\nИтоговый план:\n{current_plan}")
    return current_plan
