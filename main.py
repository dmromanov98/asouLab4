import numpy as np


def create_matrix():
    m_0 = np.matrix([[1, 1 / 2, 5],
                     [2, 1, 9],
                     [1 / 5, 1 / 9, 1]])

    m_1 = np.matrix([[1, 60 / 78, 60 / 64, 60 / 52],
                     [78 / 60, 1, 78 / 64, 78 / 52],
                     [64 / 60, 64 / 78, 1, 64 / 52],
                     [52 / 60, 52 / 78, 52 / 64, 1]])

    m_2 = np.matrix([[1, 71 / 73, 71 / 74, 71 / 69],
                     [73 / 71, 1, 73 / 74, 73 / 69],
                     [74 / 71, 74 / 73, 1, 74 / 69],
                     [69 / 71, 69 / 73, 69 / 74, 1]])

    m_3 = np.matrix([[1, 53 / 80, 53 / 71, 53 / 77],
                     [80 / 53, 1, 80 / 71, 80 / 77],
                     [71 / 53, 71 / 80, 1, 71 / 77],
                     [77 / 53, 77 / 80, 77 / 71, 1]])

    return m_0, m_1, m_2, m_3


def eigen_vector(w_0, a):
    return np.sqrt((a * w_0) / (np.transpose(a) * (1 / w_0)))


def change_comparison_matrix(m_0, i, j, val):
    m = m_0.copy()
    m[i, j] = val
    m[j, i] = 1 / val
    return m


def matching_matrix(matrix):
    m = matrix.copy()
    n = np.size(matrix, 0)
    w = eigen_vector(np.transpose(np.matrix(np.diagonal(np.ones((n, n))))), m)
    while consistency_relation(w, m) > 0.01:
        alpha = find_best_changes(w, m)
        print(m)
        for k in range(0, n):
            i = alpha[0, k]
            j = alpha[1, k]
            print("To reconcile need to change(", i, j, "). If If you will not change the value, enter \'n\'.")
            inp = input()
            if inp != 'n':
                m = change_comparison_matrix(m, i, j, float(inp))
                break

        w = eigen_vector(w, m)

    w = w / np.sum(w)
    return m, w


def consistency_relation(w, a):
    average_consistency_index = [0.00, 0.00, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57,
                                 1.59]
    n = np.size(a, 0)
    lmb = np.matrix(np.diagonal(np.ones((n, n)))) * a * (w / np.sum(w))
    uc = (lmb - n) / (n - 1)
    oc = uc / average_consistency_index[n]
    return oc


def find_best_changes(w, a):
    n = np.size(a, 0)
    alpha = np.matrix(
        [[i for i in range(1, n * n + 1)], [i for i in range(1, n * n + 1)], [0 for i in range(1, n * n + 1)]])
    print(alpha)
    for i in range(0, n):
        for j in range(i, n):
            l = n * (i - 1) + j
            alpha[0, l] = i
            alpha[1, l] = j
            alpha[2, l] = (a[i, j] * w[j]) / w[i] + w[i] / (a[i, j] * w[j])
    l = -np.sort(-alpha[2, :])
    print(l)
    alpha[0, :] = alpha[0, l - 1]
    alpha[1, :] = alpha[1, l - 1]
    alpha[2, :] = alpha[2, l - 1]
    return alpha


def main():
    c_0, c_1, c_2, c_3 = create_matrix()
    print('C0 = ', c_0)
    print('C1 = ', c_1)
    print('C2 = ', c_2)
    print('C3 = ', c_3)
    print("************************************************")
    c_0, w_0 = matching_matrix(c_0)
    c_1, w_1 = matching_matrix(c_1)
    c_2, w_2 = matching_matrix(c_2)
    c_3, w_3 = matching_matrix(c_3)
    print('C0 = ', c_0)
    print('C1 = ', c_1)
    print('C2 = ', c_2)
    print('C3 = ', c_3)

    w = np.transpose(np.concatenate((np.transpose(w_1), np.transpose(w_2), np.transpose(w_3))))
    print('W0 = ', w_0)
    print('W = ', w)
    res = w * w_0
    print('Result = ', res)


if __name__ == "__main__":
    main()
