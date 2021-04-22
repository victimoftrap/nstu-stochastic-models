import numpy as np


def __compute_initial_extended_x__(cells, f_capital, psi_capital, x0, u_t0, f_derivs, psi_derivs, x0_derivs):
    xs_extended = np.zeros((cells * len(f_capital), 1))

    xs_extended[0:len(f_capital)] = f_capital @ x0 + psi_capital * u_t0
    for alpha in range(len(f_derivs)):
        xs_extended[(alpha + 1) * len(f_derivs):(alpha + 2) * len(f_derivs)] = \
            f_derivs[alpha] @ x0 + f_capital @ x0_derivs[alpha] + psi_derivs[alpha] * u_t0
    return xs_extended


def __compute_extended_x__(f_ext, x_ext, psi_ext, u_tk):
    return f_ext @ x_ext + psi_ext * u_tk


def __compute_all_c__(cells, n):
    """Вычислить все матрицы C.
    Каждая матрица C имеет вид:
    Ci = [ O, ..., O, I, O, ..., O], где
             i-раз
    O - нулевая матрица размера nxn
    I - единичная матрица размера nxn

    :param cells: количество клеток (1 + количество переменных тетта)
    :param n: размер матрицы F
    :return: массив матриц C размера (1 + s)
    """
    cs = np.zeros((cells, n, n * cells))
    for cell_idx in range(cells):
        c = np.zeros((n, n * cells))
        for i in range(n):
            c[i, (n * cell_idx) + i] = 1
        cs[cell_idx] = c
    return cs


def __fisher_information_element__(i, j, n_capital, cs, x_ext, h_cap, r_cap, h_derivs, r_derivs):
    r_inv = np.linalg.inv(r_cap)
    x_ext_transposed = np.transpose(x_ext)
    h_transposed = np.transpose(h_cap)
    h_der_j_transposed = np.transpose(h_derivs[j])
    c0_transposed = np.transpose(cs[0])

    first = h_derivs[i] @ cs[0] @ x_ext @ x_ext_transposed @ c0_transposed @ h_der_j_transposed * r_inv[0, 0]
    second = h_derivs[i] @ cs[0] @ x_ext @ x_ext_transposed @ np.transpose(cs[j]) @ h_transposed * r_inv[0, 0]
    third = h_cap @ cs[i] @ x_ext @ x_ext_transposed @ c0_transposed @ h_der_j_transposed * r_inv[0, 0]
    fourth = h_cap @ cs[i] @ x_ext @ x_ext_transposed @ np.transpose(cs[j]) @ h_transposed * r_inv[0, 0]
    return first + second + third + fourth + (n_capital / 2) * np.trace(r_derivs[i] @ r_inv @ r_derivs[j] @ r_inv)


def compute_fisher_information(n_capital, s_theta_number, f_capital, psi_capital, h_capital, r_capital, x0, u_capital,
                               f_derivs, psi_derivs, h_derivs, r_derivs, x0_derivs):
    cells_number = 1 + s_theta_number
    # 1 cell for real value and s numbers for every theta-param

    f_extended = np.zeros((cells_number * len(f_capital), cells_number * len(f_capital)))
    for row in range(len(f_capital)):
        for col in range(len(f_capital)):
            for cell in range(cells_number):
                current_position = cell * len(f_capital)
                f_extended[current_position + row, current_position + col] = f_capital[row, col]

                if cell != 0:
                    f_extended[current_position + row, col] = f_derivs[cell - 1][row, col]
    # print(f"F_A:\n {f_extended}\n")

    psi_extended = np.zeros((cells_number * len(psi_capital), 1))
    for cell in range(cells_number):
        for row in range(len(psi_capital)):
            if cell == 0:
                setting_value = psi_capital[row, 0]
            else:
                setting_value = psi_derivs[cell - 1][row, 0]
            psi_extended[cell * len(psi_capital) + row, 0] = setting_value

    cs = __compute_all_c__(cells_number, len(f_capital))

    fisher_info = np.zeros((s_theta_number, s_theta_number))
    multiplier = n_capital / 2
    r_inverted = np.linalg.inv(r_capital)
    for i in range(s_theta_number):
        for j in range(s_theta_number):
            fisher_info[i, j] = multiplier * np.trace(r_derivs[i] @ r_inverted @ r_derivs[j] @ r_inverted)

    x_extended_prev = []
    for k in range(n_capital):
        u_tk = u_capital[k]
        if k == 0:
            x_extended = __compute_initial_extended_x__(
                cells_number, f_capital, psi_capital, x0, u_tk, f_derivs, psi_derivs, x0_derivs
            )
        else:
            x_extended = __compute_extended_x__(f_extended, x_extended_prev, psi_extended, u_tk)
        x_extended_prev = x_extended

        delta_fisher = np.zeros((s_theta_number, s_theta_number))
        for i in range(s_theta_number):
            for j in range(s_theta_number):
                delta_fisher[i, j] = __fisher_information_element__(
                    i, j, n_capital, cs, x_extended, h_capital, r_capital, h_derivs, r_derivs
                )
        fisher_info += delta_fisher
    return fisher_info
