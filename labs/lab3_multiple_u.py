import numpy as np


def pIMF(N, u_all_t, s, F, psi, H, R, Xt0, pF, pPsi, pH, pR, pXt0):
    # Объявление массива векторов Xa
    Xa = []

    # Добавление нулевого элемента в массив Xa
    Xa.append(0)

    # Матрицы C
    C = []
    for i in range(3):
        C.append(i)
        if i == 0:
            C[0] = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        elif i == 1:
            C[1] = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        else:
            C[2] = np.array([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

    # Транспонированные матрицы C
    C_t = []
    for i in range(3):
        C_t.append(i)
        if i == 0:
            C_t[0] = C[0].transpose()
        elif i == 1:
            C_t[1] = C[1].transpose()
        else:
            C_t[2] = C[2].transpose()

    U = u_all_t

    # Матрица Fa
    Fa = np.zeros((3 * len(F), 3 * len(F)))
    for row in range(len(F)):
        for col in range(len(F)):
            for cell in range(3):
                current_position = cell * len(F)
                Fa[current_position + row, current_position + col] = F[row, col]

                if cell != 0:
                    Fa[current_position + row, col] = pF[cell - 1][row, col]
    # Вектор ПсиА
    psiA = np.zeros((3 * len(psi), 1))
    for cell in range(3):
        for row in range(len(psi)):
            if cell == 0:
                setting_value = psi[row, 0]
            else:
                setting_value = pPsi[cell - 1][row, 0]
            psiA[cell * len(psi) + row, 0] = setting_value

    # Производная информационной матрицы Фишера до начала алгоритма
    # [u1(t0), u1(t1)]
    # [u2(t0), u2(t1)]
    dpM = np.zeros((len(U[0]), N, s, s))
    for signal in range(len(dpM)):
        for time in range(len(dpM[0])):
            dpM[signal, time] = np.zeros((s, s))

    all_k_pr2 = []
    for k in range(N):
        Xa.append(k + 1)
        Xa[k + 1] = np.zeros((3 * len(psi), 1))

        # Шаг 3
        u_vector = U[k]

        # Шаг 4
        if k == 0:
            # Формирование вектора Xa
            for cell in range(3):
                for row in range(len(psi)):
                    if cell == 0:
                        setting_value = F[row].dot(Xt0) + psi[row].dot(u_vector)
                    else:
                        setting_value = pF[cell - 1][row].dot(Xt0) + F[row].dot(pXt0) + pPsi[cell - 1][row].dot(u_vector)
                    Xa[k + 1][cell * len(psi) + row, 0] = setting_value
        else:
            Xa[k + 1] = Fa.dot(Xa[k]) + psiA.dot(u_vector)
        # print(Xa[k+1])

        pUa = np.array([[[1]]])
        # Шаг 5
        all_b_pr2 = []
        for b in range(N):
            # Шаг 6
            # Соотношение (36)
            all_pr1 = []
            for i in range(len(pUa)):
                if b == k:
                    pr1 = psiA.dot(pUa[i])
                else:
                    pr1 = np.zeros((len(psiA), 1))
                # print(f"PSIa * du(t{k})/du{i + 1}(t{b}): \n{pr1}\n")
                all_pr1.append(pr1)

            # Соотношение (35)
            all_pr2 = []
            for i in range(len(pUa)):
                if k == 0:
                    pr2 = all_pr1[i]
                else:
                    pr2 = Fa @ all_k_pr2[k - 1][b][i] + all_pr1[i]
                # print(f"dXa(t{k + 1})/du{i + 1}(t{b}): \n{pr2}\n")
                all_pr2.append(pr2)
            all_b_pr2.append(all_pr2)

            # Шаг 7
            for i in range(len(all_pr2)):
                pr2 = all_pr2[i]
                # Транспонирование матриц и векторов
                Xa_t = Xa[k + 1].transpose()
                pH_t = pH.transpose()
                H_t = H.transpose()
                pr_t = pr2.transpose()

                # Вычисление дельта производной ИМФ
                for row in range(1, 3):
                    for col in range(1, 3):
                        first = (pH @ C[0] @ (pr2 @ Xa_t + Xa[k + 1] @ pr_t) @ C_t[0] @ pH_t * (R ** -1))
                        second = (pH @ C[0] @ (pr2 @ Xa_t + Xa[k + 1] @ pr_t) @ C_t[col] @ H_t * (R ** -1))
                        third = (H @ C[row] @ (pr2 @ Xa_t + Xa[k + 1] @ pr_t) @ C_t[0] @ pH_t * (R ** -1))
                        fourth = (H @ C[row] @ (pr2 @ Xa_t + Xa[k + 1] @ pr_t) @ C_t[col] @ H_t * (R ** -1))
                        dpM[i, b][row - 1, col - 1] += first + second + third + fourth
        all_k_pr2.append(all_b_pr2)

    for matrix in dpM:
        print(matrix)
    # Возврат первой матрицы (для проверки)
    return dpM
