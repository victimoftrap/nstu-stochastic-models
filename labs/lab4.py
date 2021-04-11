from labs.fisher_information import compute_fisher_information

from collections import namedtuple

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

PlanElement = namedtuple("PlanElement", ["u", "p"])

delta = 0.001


def __x_a_criteria__(fisher_by_plan):
    return np.trace(np.linalg.inv(fisher_by_plan))


def __x_d_criteria__(fisher_by_plan):
    return - np.log(np.linalg.det(fisher_by_plan))


def optimal_plan(N, s, F, psi, H, R, x0, u_bounds, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs):
    def __minimize_us__(us):
        iterated_fisher = np.zeros((s, s))
        for j in range(q):
            u_j = us[j]
            p_weight = current_plan[j].p
            fisher = compute_fisher_information(
                N, s, F, psi, H, R, x0, u_j,
                F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
            )
            iterated_fisher += p_weight * fisher
        return __x_a_criteria__(iterated_fisher)

    def __minimize_ps__(ps):
        iterated_fisher = np.zeros((s, s))
        for j in range(q):
            p_j = ps[j]
            iterated_fisher += p_j * updated_fisher_matrices[j]
        return __x_a_criteria__(iterated_fisher)

    def __mju_a_criteria__(u_tk_plus_1, plan):
        fisher_plan = np.zeros((s, s))
        for j in range(q):
            fisher = compute_fisher_information(
                N, s, F, psi, H, R, x0, plan[j].u,
                F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
            )
            fisher_plan += plan[j].p * fisher
        print(u_tk_plus_1)
        fisher_by_u_tk_plus_1 = compute_fisher_information(
            N, s, F, psi, H, R, x0, u_tk_plus_1[j],
            F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
        )
        return np.trace(fisher_plan @ fisher_by_u_tk_plus_1)

    def __eta_a_criteria__(plan_star):
        fisher_plan = np.zeros((s, s))
        for j in range(q):
            fisher = compute_fisher_information(
                N, s, F, psi, H, R, x0, plan_star[j].u,
                F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
            )
            fisher_plan += plan_star[j].p * fisher
        return np.trace(np.linalg.inv(fisher_plan))

    # Шаг 1
    q = int((s * (s + 1) / 2) + 1)

    # Начальный план
    epsilon_zero_plan = []
    for i in range(q):
        eps_u = 1
        eps_p = 1 / q
        epsilon_zero_plan.append(PlanElement(eps_u, eps_p))
    print(f"Начальный невырожденный план:\n{epsilon_zero_plan}\n")

    # Информационные матрицы от начального плана
    fisher_matrices = []
    for i in range(q):
        u = epsilon_zero_plan[i].u
        fisher_zero = compute_fisher_information(
            N, s, F, psi, H, R, x0, u, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
        )
        print(f"Матрица одноточечного плана от U0[{i}]:\n{fisher_zero}")
        fisher_matrices.append(fisher_zero)

    # Нормализованная матрица всего плана
    normalized_fisher = np.zeros((s, s))
    for i in range(q):
        weight = epsilon_zero_plan[i].p
        fisher_matrix = fisher_matrices[i]
        normalized_fisher += weight * fisher_matrix
    print(f"Матрица всего плана (с учётом весов):\n{normalized_fisher}\n")

    current_plan = epsilon_zero_plan
    k = 0
    while True:
        print(f"План на {k} итерации:\n{current_plan}")
        # Шаг 2
        # Выберем значения U из текущего плана для последующей оптимизации
        current_us = np.array([plan_el.u for plan_el in current_plan])
        # Минимизируем U с помощью minimize из scipy.optimize
        us_result = minimize(
            __minimize_us__, current_us,
            method='SLSQP',
            bounds=np.array([u_bounds for i in range(q)])
        )

        # Создаём новый план из новых U и старых весов P
        updated_us_epsilon_plan = [PlanElement(us_result.x[i], current_plan[i].p) for i in range(q)]
        print(f"План после минимизации U:\n{updated_us_epsilon_plan}")

        # Создаём новые информационные матрицы от плана с обновлёнными U
        updated_fisher_matrices = []
        for i in range(q):
            mini_u = updated_us_epsilon_plan[i].u
            fisher_upd = compute_fisher_information(
                    N, s, F, psi, H, R, x0, mini_u, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
            )
            updated_fisher_matrices.append(fisher_upd)
            print(f"Матрица одноточечного плана от U{k + 1}[{i}]:\n{fisher_upd}")

        # Шаг 3
        # Выберем значения P из текущего плана для последующей оптимизации
        current_ps = np.array([plan_el.p for plan_el in updated_us_epsilon_plan])
        # Минимизируем веса
        ps_result = minimize(
            __minimize_ps__, current_ps,
            method='SLSQP',
            bounds=np.array([[0., 1.] for i in range(q)]),
        )
        # Создаём полностью новый план из новых U и P
        next_epsilon_plan = [PlanElement(us_result.x[i], ps_result.x[i]) for i in range(q)]
        print(f"План после минимизации весов p:\n{next_epsilon_plan}")

        # Шаг 4
        inequality_value = 0
        # Считем разницу между векторами для нормы
        delta_us = us_result.x - current_us
        # Считаем ту часть с весами
        for i in range(q):
            inequality_value += (ps_result.x[i] - current_ps[i]) ** 2
        inequality_value += np.linalg.norm(delta_us) ** 2

        if inequality_value <= delta:
            break
            # if abs(__mju_a_criteria__(us_result.x, next_epsilon_plan) - __eta_a_criteria__(next_epsilon_plan)) <= delta:
            #     break

        # Иначе идём на шаги 2-3
        print("Продолжаем...\n")
        k += 1
        current_plan = next_epsilon_plan
