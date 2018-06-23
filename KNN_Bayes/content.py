# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division

import numpy as np


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. Odleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    def hamming(vector_x):
        return np.sum(np.logical_xor(vector_x, X_train.toarray()), axis=1)

    return np.apply_along_axis(hamming, axis=1, arr=X.toarray())


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    w = Dist.argsort(kind='mergesort')
    return y[w]


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    numberofclasses = 4

    output = np.delete(y, range(k, y.shape[1]), axis=1)

    array = np.full((y.shape[0], numberofclasses), 0)
    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            index = output[x][y]
            array[x][index] += 1
    output = array

    output = np.delete(output, range(numberofclasses, output.shape[1]), axis=1)
    output = np.divide(output, k)

    return output


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    array = np.ndarray(shape=y_true.shape, dtype=int)
    for x in range(p_y_x.shape[0]):
        maxIndex = 0
        maxValue = 0
        for y in range(p_y_x.shape[1]):
            if p_y_x[x][y] >= maxValue:
                maxIndex = y
                maxValue = p_y_x[x][y]
        array[x] = maxIndex

    output = 0
    for i in range(array.shape[0]):
        if array[i] != y_true[i]:
            output += 1
    output = output / array.shape[0]
    return output


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """

    # distancearray = hamming_distance(Xval, Xtrain)
    # labeledarray = sort_train_labels_knn(distancearray, ytrain)
    #
    # minerror, mink, p_k = 1, 0, []
    # for k in k_values:
    #     p_y_x = p_y_x_knn(labeledarray, k)
    #     error = classification_error(p_y_x, yval)
    #     if error < minerror:
    #         minerror = error
    #         mink = k
    #     p_k.append(error)
    #
    # return minerror, mink, p_k
    dist = hamming_distance(Xval, Xtrain)
    y = sort_train_labels_knn(dist, ytrain)

    errors = []
    for k in k_values:
        p_y_x = p_y_x_knn(y, k)
        err = classification_error(p_y_x, yval)
        errors.append(err)

    index = np.argmin(errors)
    best_k = k_values[int(index % len(k_values))]
    best_error = errors[int(index)]

    return best_error, best_k, errors


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    numberofclasses = 4
    dlugosc = ytrain.shape[0]

    output = np.full(numberofclasses, 0)
    for x in range(dlugosc):
        index = ytrain[x]
        output[index] += 1
    output = np.divide(output, dlugosc)

    return output


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """

    Xtrain = Xtrain.toarray()
    numberofclasses = 4
    output = []

    def summ(row, y_equal_k):
        return np.sum(np.bitwise_and(row, y_equal_k))

    for i in range(0, numberofclasses):
        yk = np.equal(ytrain, i)
        yksum = np.sum(yk)
        row = np.apply_along_axis(summ, axis=0, arr=Xtrain, y_equal_k=yk)
        output.append(np.divide(np.add(row, a - 1), yksum + a + b - 2))

    output = np.vstack(output) #Zamiana listy na ndarray
    return output


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    X = X.toarray()

    def p_y_x(x):
        prod = np.prod((p_x_1_y ** x) * (np.subtract(p_x_1_y, 1) ** (1 - x)), axis=1) * p_y
        return prod / np.sum(prod, axis=0)

    return np.apply_along_axis(p_y_x, axis=1, arr=X)


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """
    errors = []
    for a in a_values:
        error = []
        for b in b_values:
            p_y = estimate_a_priori_nb(ytrain)
            p_x_1_y = estimate_p_x_y_nb(Xtrain, ytrain, a, b)

            p_y_x = p_y_x_nb(p_y, p_x_1_y, Xval)
            err = classification_error(p_y_x, yval)

            error.append(err)
        errors.append(error)

    index = np.argmin(errors)
    index_a = int(index / len(b_values))
    index_b = int(index % len(a_values))

    error_best = errors[index_a][index_b]
    best_a = a_values[index_a]
    best_b = b_values[index_b]

    return error_best, best_a, best_b, errors
