from labs.fisher_information import compute_fisher_information
from labs.Л3 import pIMF

from collections import namedtuple

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

PlanElement = namedtuple("PlanElement", ["u", "p"])

delta = 0.01


def __x_a_criteria__(fisher_by_plan):
    val = np.trace(np.linalg.inv(fisher_by_plan))
    print(f"X[M(ksi)] A-optimal: {val}")
    return val


def __x_d_criteria__(fisher_by_plan):
    val = - np.log(np.linalg.det(fisher_by_plan))
    print(f"X[M(ksi)] D-optimal: {val}")
    return val


def optimal_plan(N, s, F, psi, H, R, x0, u_bounds, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs):
    def __grad_a_us_criteria__(us):
        fisher_plan = np.zeros((s, s))
        for j in range(q):
            fisher = compute_fisher_information(
                N, s, F, psi, H, R, x0, us[j * N:(j + 1) * N],
                F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
            )
            fisher_plan += current_plan[j].p * fisher
        f_2_degree = np.linalg.inv(fisher_plan) @ np.linalg.inv(fisher_plan)

        u_gradient = []
        for j in range(q):
            derivs = pIMF(N, np.array([us[j] for i in range(N)]))
            u_optima = []
            for der in derivs:
                u_optima.append(- current_plan[j].p * np.trace(f_2_degree @ der))
            u_gradient.append(np.array(u_optima))
        return np.array(u_gradient)

    def __grad_a_ps_criteria__(ps):
        fisher_plan = np.zeros((s, s))
        f_matrices = []
        for j in range(q):
            fisher = compute_fisher_information(
                N, s, F, psi, H, R, x0, updated_us_epsilon_plan[j].u,
                F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
            )
            f_matrices.append(fisher)
            fisher_plan += ps[j] * fisher
        f_2_degree = np.linalg.inv(fisher_plan) @ np.linalg.inv(fisher_plan)

        p_gradient = []
        for mat in f_matrices:
            p_val = - np.trace(f_2_degree @ mat)
            p_gradient.append(np.linalg.norm(p_val))
        return np.array(p_gradient)

    def __minimize_us__(us):
        iterated_fisher = np.zeros((s, s))
        for j in range(q):
            fisher = compute_fisher_information(
                N, s, F, psi, H, R, x0, us[j * N:(j + 1) * N],
                F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
            )
            iterated_fisher += current_plan[j].p * fisher
        return __x_a_criteria__(iterated_fisher)

    def __minimize_ps__(ps):
        iterated_fisher = np.zeros((s, s))
        for j in range(q):
            iterated_fisher += ps[j] * updated_fisher_matrices[j]
        return __x_a_criteria__(iterated_fisher)

    def __mju_a_criteria__(u_of_point, plan):
        fisher_plan = np.zeros((s, s))
        for j in range(q):
            fisher = compute_fisher_information(
                N, s, F, psi, H, R, x0, plan[j].u,
                F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
            )
            fisher_plan += plan[j].p * fisher
        f_plan_inverted = np.linalg.inv(fisher_plan)

        fisher_by_u_of_point = compute_fisher_information(
            N, s, F, psi, H, R, x0, u_of_point,
            F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
        )
        return np.trace(f_plan_inverted @ f_plan_inverted @ fisher_by_u_of_point)

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
        eps_u = [1 for i in range(N)]
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
        # print(f"Матрица одноточечного плана от U0[{i}]:\n{fisher_zero}")
        fisher_matrices.append(fisher_zero)

    # Нормализованная матрица всего плана
    normalized_fisher = np.zeros((s, s))
    for i in range(q):
        weight = epsilon_zero_plan[i].p
        fisher_matrix = fisher_matrices[i]
        normalized_fisher += weight * fisher_matrix
    # print(f"Матрица всего плана (с учётом весов):\n{normalized_fisher}\n")

    current_plan = epsilon_zero_plan
    k = 0
    while True:
        print(f"План на {k} итерации:\n{current_plan}")
        # Шаг 2
        # Выберем значения U из текущего плана для последующей оптимизации
        current_us = []
        for elem in current_plan:
            for u_tk in elem.u:
                current_us.append(u_tk)
        current_us = np.array(current_us)

        # Минимизируем U с помощью minimize из scipy.optimize
        us_result = minimize(
            __minimize_us__, current_us,
            method='SLSQP',
            bounds=np.array([u_bounds for i in range(N * q)]),
            jac=__grad_a_us_criteria__,
        )

        # Создаём новый план из новых U и старых весов P
        updated_us_epsilon_plan = [PlanElement(us_result.x[i * N:(i + 1) * N], current_plan[i].p) for i in range(q)]
        # print(f"План после минимизации U:\n{updated_us_epsilon_plan}")

        # Создаём новые информационные матрицы от плана с обновлёнными U
        updated_fisher_matrices = []
        for i in range(q):
            mini_u = updated_us_epsilon_plan[i].u
            fisher_upd = compute_fisher_information(
                N, s, F, psi, H, R, x0, mini_u, F_derivs, psi_derivs, H_derivs, R_derivs, x0_derivs
            )
            updated_fisher_matrices.append(fisher_upd)
            # print(f"Матрица одноточечного плана от U{k + 1}[{i}]:\n{fisher_upd}")

        # Шаг 3
        # Выберем значения P из текущего плана для последующей оптимизации
        current_ps = np.array([plan_el.p for plan_el in updated_us_epsilon_plan])

        # Условие, задающее, что сумма всех весов должна быть равна 1
        # sum_of_all_weights_1_cond = LinearConstraint(
        #     [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        #     [0, 0, 0, 0], [1, 1, 1, 1]
        # )
        p_variable_matrix = []
        for i in range(q):
            if i == 0:
                p_variable_matrix.append([1 for i in range(q)])
            else:
                p_variable_matrix.append([0 for i in range(q)])
        sum_of_all_weights_1_cond = LinearConstraint(p_variable_matrix, [0 for i in range(q)], [1 for i in range(q)])

        # Минимизируем веса
        ps_result = minimize(
            __minimize_ps__, current_ps,
            method='SLSQP',
            bounds=np.array([[0., 1.] for i in range(q)]),
            constraints=sum_of_all_weights_1_cond,
            jac=__grad_a_ps_criteria__,
        )
        # Создаём полностью новый план из новых U и P
        next_epsilon_plan = [PlanElement(us_result.x[i * N:(i + 1) * N], ps_result.x[i]) for i in range(q)]
        # print(f"План после минимизации весов p:\n{next_epsilon_plan}")

        # Шаг 4
        inequality_value = 0
        # Считаем ту часть с весами
        for i in range(q):
            ps_difference = (ps_result.x[i] - current_ps[i]) ** 2
            us_difference = np.linalg.norm(us_result.x[i * N:(i + 1) * N] - current_us[i * N:(i + 1) * N]) ** 2
            inequality_value += ps_difference + us_difference

        if inequality_value <= delta:
            step_5_condition_values = []
            for i in range(q):
                mju_value = __mju_a_criteria__(us_result.x[i * N:(i + 1) * N], next_epsilon_plan)
                eta_value = __eta_a_criteria__(next_epsilon_plan)
                step_5_condition_values.append(abs(mju_value - eta_value))

            if all(val <= delta for val in step_5_condition_values):
                # Конец алгоритма
                current_plan = next_epsilon_plan
                break

        # Иначе идём на шаги 2-3
        k += 1
        current_plan = next_epsilon_plan
        print("")

    print(f"\nИтоговый план:\n{current_plan}\n")
    return current_plan


def cleanup_plan(plan):
    cleaned_plan = [list(plan[0])]
    for i in range(1, len(plan)):
        next_point = plan[i]
        merged = False
        for el in cleaned_plan:
            u_in_plan = el[0]
            if all(np.isclose(u_in_plan[j], next_point.u[j]) for j in range(len(u_in_plan))):
                el[1] += next_point.p
                merged = True
                break

        if not merged:
            cleaned_plan.append(next_point)
    return list(map(lambda point: PlanElement(point[0], point[1]), cleaned_plan))
