import numpy as np


def create_matrix():
    m_0 = np.matrix([[1, 1 / 1.31, 1.4],
                     [1.31, 1, 1.25],
                     [1 / 1.4, 1 / 1.25, 1]])

    m_1 = np.matrix([[1, 1 / 1.31, 1, 1.25],
                     [1.31, 1, 1.25, 1.51],
                     [1, 1 / 1.25, 1, 1.25],
                     [1 / 1.25, 1 / 1.51, 1 / 1.25, 1]])

    m_2 = np.matrix([[1, 1, 1, 1],
                     [1, 1, 1, 1.25],
                     [1, 1, 1, 1.31],
                     [1, 1 / 1.25, 1 / 1.31, 1]])

    m_3 = np.matrix([[1, 1 / 1.51, 1 / 1.31, 1 / 1.51],
                     [1.51, 1, 1.25, 1],
                     [1.31, 1 / 1.25, 1, 1 / 1.25],
                     [1.51, 1, 1.25, 1]])

    return m_0, m_1, m_2, m_3


# maximum eigenvector of the matrix
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
        [[i for i in range(0, n * n)],
         [i for i in range(0, n * n)],
         [0 for i in range(0, n * n)]])
    print(alpha)
    for i in range(0, n):
        for j in range(i, n):
            l = n * (i - 1) + j
            alpha[0, l] = i
            alpha[1, l] = j
            alpha[2, l] = (a[i, j] * w[j]) / w[i] + w[i] / (a[i, j] * w[j])
    sort(alpha)
    return alpha


def sort(matrix):
    x_size = np.size(matrix, 1)
    for i in range(0, x_size - 1):
        for j in range(i + 1, x_size):
            if matrix[2, i] < matrix[2, j]:
                matrix[2, i], matrix[1, i], matrix[0, i], matrix[2, j], matrix[1, j], matrix[0, j] = \
                    matrix[2, j], matrix[1, j], matrix[0, j], matrix[2, i], matrix[1, i], matrix[0, i]
    return matrix


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
    res = w.dot(w_0)
    print('Result = ', res)


if __name__ == "__main__":
    main()
