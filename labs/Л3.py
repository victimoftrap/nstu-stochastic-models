import numpy as np

def pIMF():
    # Нулевая матрица 2 на 2
    zMatrix = np.array([[0, 0], [0, 0]])
    
    # Объявление массива векторов Xa
    Xa = []
    # Добавление нулевого элемента в массив Xa
    Xa.append(0)
    
    # Объвление массива производной по u * psi
    pr = []
    
    # Пустой массив
    tmpList = []
    
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
    
    # Шаг 1
    # Инициализация заданных переменных
    p1 = -1
    p2 = 0.5
    F = np.array([[-0.8, 1], [p1, 0]])
    psi = np.array([[p2], [1]])
    H = np.array([1, 0])
    R = 0.1
    Xt0 = np.array([[0], [0]])
    
    # Инициализация производных по переменным
    pF = [np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 0]])]
    pPsi = [np.array([[0], [0]]), np.array([[1], [0]])]
    pH = np.array([0, 0])
    pR = 0
    pXt0 = Xt0
    
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
    
    # Шаг 2
    N = 3
    
    # Производная информационной матрицы Фишера до начала алгоритма
    dpM = []
    for i in range(N):
        dpM.append(np.zeros((2, 2)))
    
    for k in range(N):
        Xa.append(k+1)
        Xa[k+1] = np.zeros((3*len(psi), 1))
        pr.append(list(tmpList))
        # Шаг 3
        u = 1
        # Шаг 4
        if k == 0: 
            # Формирование вектора Xa
            for cell in range(3):
                for row in range(len(psi)):
                    if cell == 0:
                        setting_value = F[row].dot(Xt0) + psi[row] * u
                    else:
                        setting_value = pF[cell - 1][row].dot(Xt0) + F[row].dot(pXt0) + pPsi[cell - 1][row] * u
                    Xa[k+1][cell * len(psi) + row, 0] = setting_value
        else:
            Xa[k+1] = Fa.dot(Xa[k]) + psiA * u
        #print(Xa[k+1])
        # Шаг 5
        for b in range(N):
            pr[k].append(list(tmpList))
            # Шаг 6
            # Производная u всегда равна 1, т.к. у нас скаляр
            pUa = 1
            # Соотношение (36)
            if b == k:
                pr36 = psiA * pUa
            else:
                pr36 = np.zeros((len(psiA), 1))
            # Соотношение (35)
            if k == 0:
                pr[k][b] = pr36
            else: 
                 pr[k][b] = Fa @ pr[k-1][b] + pr36
            # Шаг 7
            # Транспонирование матриц и векторов
            Xa_t = Xa[k+1].transpose()
            pH_t = pH.transpose()
            H_t = H.transpose()
            pr_t = pr[k][b].transpose()
            #print("PR:")
            #print(pr[k][b])
            # Вычисление дельта производной ИМФ
            
            for row in range(1, 3):
                for col in range(1, 3):
                    dpM[b][row-1, col-1] += (pH @ C[0] @ (pr[k][b] @ Xa_t + Xa[k+1] @ pr_t) @ C_t[0] @ pH_t * (R ** -1)) + \
                        + (pH @ C[0] @ (pr[k][b] @ Xa_t + Xa[k+1] @ pr_t) @ C_t[col] @ H_t * (R ** -1)) + \
                            + (H @ C[row] @ (pr[k][b] @ Xa_t + Xa[k+1] @ pr_t) @ C_t[0] @ pH_t * (R ** -1)) + \
                                + (H @ C[row] @ (pr[k][b] @ Xa_t + Xa[k+1] @ pr_t) @ C_t[col] @ H_t * (R ** -1))
            
    for i in range(N):
        print(dpM[i])
    # Возврат первой матрицы (для проверки)
    return dpM[0]






