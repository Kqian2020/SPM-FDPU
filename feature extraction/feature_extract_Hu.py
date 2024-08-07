import numpy as np
import scipy.io as sio


def order_moment(image, p, q, x_average: float = 0.0, y_average: float = 0.0):
    x_matrix = np.mat(np.arange(1, image.shape[0] + 1)).T - x_average
    y_matrix = np.mat(np.arange(1, image.shape[1] + 1)) - y_average
    xy_matrix = np.dot(np.power(x_matrix, p), np.power(y_matrix, q))
    sum_moment = np.sum(np.multiply(xy_matrix, image))
    # sumMoment = 0
    # for x in range(1, image.shape[0]+1):
    #     for y in range(1, image.shape[1]+1):
    #         sumMoment = sumMoment + np.power(x, p)*np.power(y, q)*image[x-1, y-1]
    return sum_moment


def central_moment(image, p, q):
    m_00 = order_moment(X_png[i].T, 0, 0)
    x_average = order_moment(image, 1, 0) / m_00
    y_average = order_moment(image, 0, 1) / m_00
    return order_moment(image, p, q, x_average, y_average)


def normalized_central_moment(image, p, q):
    mu_00 = central_moment(image, 0, 0)
    eta = central_moment(image, p, q) / mu_00 ** ((p + q) / 2)
    return eta


if __name__ == '__main__':
    indicator = sio.loadmat('E:/QK/Code/My/indicator/indicator_diagram.mat')
    X_png = indicator['X_png']
    X_moment = np.zeros((len(X_png), 7))

    for i in range(len(X_png)):
        # a = order_moment(X_png[i].T, 2, 0)
        # print('(2+0)-order moment:', a)
        # b = central_moment(X_png[i].T, 2, 0)
        # print('(2+0)-order central moment:', b)
        # c = normalized_central_moment(X_png[i].T, 2, 0)
        # print('(2+0)-order normalized central moment:', c)
        temp_image = X_png[i].T
        # temp_image[temp_image > 0] = 1

        eta_20 = normalized_central_moment(temp_image, 2, 0)
        eta_02 = normalized_central_moment(temp_image, 0, 2)
        eta_11 = normalized_central_moment(temp_image, 1, 1)

        eta_30 = normalized_central_moment(temp_image, 3, 0)
        eta_03 = normalized_central_moment(temp_image, 0, 3)
        eta_21 = normalized_central_moment(temp_image, 2, 1)
        eta_12 = normalized_central_moment(temp_image, 1, 2)

        X_moment[i, 0] = eta_20 + eta_02
        X_moment[i, 1] = (eta_20 - eta_02) ** 2 + 4 * eta_11 ** 2
        X_moment[i, 2] = (eta_30 - 3 * eta_12) ** 2 + (3 * eta_21 - eta_03) ** 2
        X_moment[i, 3] = (eta_30 - eta_12) ** 2 + (eta_21 + eta_03) ** 2
        X_moment[i, 4] = (eta_30 - 3 * eta_12)*(eta_30 + eta_12)*((eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2) + \
                         (3 * eta_21 - eta_03)*(eta_21 + eta_03)*((3 * eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2)
        X_moment[i, 5] = (eta_20 - 3 * eta_02)*((eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2) + \
                         4*eta_11 * (eta_30 + eta_12) * (eta_21 + eta_03)
        X_moment[i, 6] = (3*eta_21 - eta_03)*(eta_30 + eta_12)*((eta_30 + eta_12)**2 - (eta_21 + eta_03)**2) + \
                         (3*eta_12-eta_30)*(eta_21 + eta_03)*((3*eta_30 + eta_12)**2 - (eta_21 + eta_03)**2)
    X_moment_normalized = (X_moment - X_moment.min(axis=0)) / (X_moment.max(axis=0) - X_moment.min(axis=0))
    sio.savemat('./data/indicator_moment_1.mat', {'X_moment': X_moment, 'X_moment_normalized': X_moment_normalized, 'Y': indicator['Y']})
