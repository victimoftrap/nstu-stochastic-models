import numpy as np


def compute_fisher_derivatives(N, s, F, psi, H, R, X0, U, pF, pPsi, pH, pR, x0_derivs):
    C = []
    for i in range(3):
        C.append(i)
        if i == 0:
            C[0] = np.array([[1, 0, 0]])
        elif i == 1:
            C[1] = np.array([[0, 1, 0]])
        else:
            C[2] = np.array([[0, 0, 1]])

    C_t = []
    for i in range(3):
        C_t.append(i)
        if i == 0:
            C_t[0] = C[0].transpose()
        elif i == 1:
            C_t[1] = C[1].transpose()
        else:
            C_t[2] = C[2].transpose()

    Fa = np.zeros((3 * len(F), 3 * len(F)))
    for row in range(len(F)):
        for col in range(len(F)):
            for cell in range(3):
                current_position = cell * len(F)
                Fa[current_position + row, current_position + col] = F[row, col]

                if cell != 0:
                    Fa[current_position + row, col] = pF[cell - 1][row, col]
    # print(f"Fa: \n{Fa}\n")

    # Вектор ПсиА
    psiA = np.zeros((3 * len(psi), len(psi[0])))
    for cell in range(3):
        for row in range(len(psi)):
            for col in range(len(psi[0])):
                if cell == 0:
                    setting_value = psi[row, col]
                else:
                    setting_value = pPsi[cell - 1][row, col]
                psiA[cell * len(psi) + row, col] = setting_value
    # print(f"PSIa: \n{psiA}\n")

    # [u1(t0), u1(t1)]
    # [u2(t0), u2(t1)]
    fisher_derivatives = np.zeros((len(U[0]), N, s, s))
    for signal in range(len(fisher_derivatives)):
        for time in range(len(fisher_derivatives[0])):
            fisher_derivatives[signal, time] = np.zeros((s, s))
    # print(f"Fisher derivatives: \n{fisher_derivatives}")

    Xa = [0]
    for k in range(N):
        # print(f"k: {k}")
        Xa.append(k + 1)
        Xa[k + 1] = np.zeros((3 * len(psi), 1))
        # print(f"Xa(t{k + 1}): \n{Xa[k + 1]}\n")

        # Шаг 3
        u_vector = U[k]
        # print(f"U(t{k}): \n{u_vector}\n")

        # Шаг 4
        if k == 0:
            # Формирование вектора Xa
            for cell in range(3):
                if cell == 0:
                    setting_value = psi.dot(u_vector)
                else:
                    setting_value = pPsi[cell - 1].dot(u_vector)
                # print(f"by cell {cell}: {setting_value}")
                Xa[k + 1][cell] = setting_value
        else:
            Xa[k + 1] = 0 + psiA.dot(u_vector)
        # print(f"Xa(t{k + 1}): \n{Xa[k + 1]}\n")

        pUa = np.array([[[1], [0]],
                        [[0], [1]]])
        # Шаг 5
        for b in range(N):
            # print(f"beta: {b}")
            # Шаг 6
            # print(f"pU: \n{pUa[k]}\n")

            # Соотношение (36)
            all_pr1 = []
            for i in range(N):
                if b == k:
                    pr1 = 0 + psiA.dot(pUa[i])
                else:
                    pr1 = np.zeros((len(psiA), 1))
                # print(f"PSIa * du(t{k})/du{i + 1}(t{b}): \n{pr1}\n")
                all_pr1.append(pr1)

            # Соотношение (35)
            all_pr2 = []
            for i in range(N):
                if k == 0:
                    pr2 = all_pr1[i]
                else:
                    pr2 = 0 + all_pr1[i]
                # print(f"dXa(t{k + 1})/du{i + 1}(t{b}): \n{pr2}\n")
                all_pr2.append(pr2)

            for i in range(len(all_pr2)):
                pr2 = all_pr2[i]

                # Транспонирование матриц и векторов
                Xa_t = Xa[k + 1].transpose()
                pH_t = pH.transpose()
                H_t = H.transpose()
                pr2_t = pr2.transpose()

                R_inv = np.linalg.inv(R)
                # Вычисление дельта производной ИМФ
                for row in range(1, 3):
                    for col in range(1, 3):
                        first = pH @ C[0] @ (pr2 @ Xa_t + Xa[k + 1] @ pr2_t) @ C_t[0] @ pH_t @ R_inv
                        second = pH @ C[0] @ (pr2 @ Xa_t + Xa[k + 1] @ pr2_t) @ C_t[col] @ H_t * R_inv
                        third = H @ C[row] @ (pr2 @ Xa_t + Xa[k + 1] @ pr2_t) @ C_t[0] @ pH_t * R_inv
                        fourth = (H @ C[row] @ (pr2 @ Xa_t + Xa[k + 1] @ pr2_t) @ C_t[col] @ H_t @ R_inv)
                        fisher_derivatives[i, b][row - 1, col - 1] += first + second + third + fourth
    return fisher_derivatives
