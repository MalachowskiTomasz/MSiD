# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np


def mean_squared_error(x, y, w):
    '''
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    '''
    result = 0
    for i in range(x.size):
        equationResult = 0
        for j in range(w.size):
            equationResult += (x[i][0] ** j) * w[j][0]
        result += ((y[i][0] - equationResult) ** 2)
    result /= x.size

    return result


def design_matrix(x_train, M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''
    'print(x_train)'

    array = np.ndarray(shape=(x_train.size, M + 1), dtype=float)

    for x in range(x_train.size):
        for y in range(M + 1):
            array[x][y] = x_train[x][0] ** y

    return array


def least_squares(x_train, y_train, M):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''
    fi = design_matrix(x_train, M)

    w = np.dot(np.dot(np.linalg.inv(np.dot(fi.transpose(), fi)), fi.transpose()), y_train)

    return w, mean_squared_error(x_train, y_train, w)


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''
    fi = design_matrix(x_train, M)

    w = np.dot(np.dot(np.linalg.inv(np.dot(fi.transpose(), fi) + np.dot(np.identity(M + 1), regularization_lambda)),
                      fi.transpose()), y_train)

    return w, mean_squared_error(x_train, y_train, w)


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''
    tabl = []
    for m in M_values:
        tabl.append(least_squares(x_train, y_train, m))

    train_err = 300
    val_err = 300
    bestw = 0

    # c, c_err = tabl[0]
    # print(c, c_err)
    # r, r_err = least_squares(x_val, y_val, tabl[0])
    # print(r, r_err)

    for w, w_err in tabl:
        r_val = mean_squared_error(x_val, y_val, w)
        if r_val < val_err:
            bestw = w
            val_err = r_val
            train_err = w_err

    return bestw, train_err, val_err


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''

    tabl = []
    for lam in lambda_values:
        tabl.append((regularized_least_squares(x_train, y_train, M, lam), lam))

    train_err = 300
    val_err = 300
    best_lam = 300
    best_w = 0

    for (w, w_err), lam in tabl:
        r_val = mean_squared_error(x_val, y_val, w)
        if r_val < val_err:
            best_lam = lam
            best_w = w
            val_err = r_val
            train_err = w_err

    return best_w, train_err, val_err, best_lam
