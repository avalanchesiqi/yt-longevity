import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Hawkes Intensity Process model


def get_C(k, alpha=2.016, beta=0.1):
    """
    Get parameter capital C
    :param k: scaling factor for video quality
    :param alpha: power-law exponent of user influence distribution
    :param beta: user influence component
    :return: parameter capital C
    """
    return k*(alpha-1)/(alpha-beta-1)


def rand_initialize_weights(n):
    """
    Initialize multiple sets of random weights for theta
    :param n: number of sets of random weights
    :return: n+3 sets of random vectors, in the order of mu, theta, C, c, gamma, eta
    """
    # 3 sets of fixed weights
    a = np.array([0.2, 10, get_C(3), 4.5, 10, 100])
    b = np.array([1, 3.35, get_C(0.0024), 0.0001, 100, 10])
    c = np.array([5, 0.00001, get_C(50), 1.5, 10000, 1000])
    ret = [a, b, c]
    for _ in xrange(n):
        rand_mu = np.random.uniform(0, 505.90, 1)
        rand_theta = np.random.uniform(2.3, 67.7, 1)
        rand_C = np.random.uniform(get_C(0), get_C(52.9), 1)
        rand_c = np.random.uniform(np.finfo(float).eps, 4, 1)
        rand_gamma = np.random.uniform(0, 9947, 1)
        rand_eta = np.random.uniform(0, 289.2, 1)
        ret.append(np.array([rand_mu, rand_theta, rand_C, rand_c, rand_gamma, rand_eta]))
    return ret


def rep(i, c):
    """
    Replacement for sequence (tau + c)
    :param i: tau value
    :param c: c value
    :return: abbreviated presentation
    """
    return np.arange(1, i+1)[::-1]+c


def cost_function(params, x, y, num_split=None):
    """
    Non-regularized cost function for HIP model
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount
    :param y: observed viewcount
    :param num_split: number of test set
    :return: cost function value
    """
    view_predict = predict(params, x)
    cost_vector = view_predict - y
    if num_split is not None:
        cost_vector = cost_vector[-num_split:]
    cost = np.sum(cost_vector ** 2) / 2
    return cost


def reg_cost_function(params, x, y, params0):
    """
    Regularized cost function for HIP model
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount
    :param y: observed viewcount
    :param params0: reference values from non-regularized model
    :return: cost function value
    """
    mu, theta, C, c, gamma, eta = params
    mu0, C0, gamma0, eta0, w = params0
    view_predict = predict(params, x)
    cost_vector = view_predict - y
    cost = np.sum(cost_vector ** 2) / 2
    return cost + w/2*((mu/mu0)**2+(C/C0)**2+(gamma/gamma0)**2+(eta/eta0)**2)


def grad_descent(params, x, y):
    """
    Non-regularized gradient function for HIP model
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount
    :param y: observed viewcount
    :return: cost function value
    """
    mu, theta, C, c, gamma, eta = params
    view_predict = predict(params, x)
    n = len(x)
    # partial derivative for mu
    grad_mu_vector = np.zeros(n)
    grad_mu_vector[0] = x[0]
    for i in xrange(1, n):
        grad_mu_vector[i] = x[i] + C*np.sum(grad_mu_vector[:i] * (rep(i, c)**(-1-theta)))
    grad_mu = np.sum((view_predict-y)*grad_mu_vector)/n
    # partial derivative for theta
    grad_theta_vector = np.zeros(n)
    grad_theta_vector[0] = 0
    for i in xrange(1, n):
        grad_theta_vector[i] = C*np.sum((grad_theta_vector[:i]-view_predict[:i]*np.log(rep(i, c))) * (rep(i, c)**(-1-theta)))
    grad_theta = np.sum((view_predict-y)*grad_theta_vector)/n
    # partial derivative for C
    grad_C_vector = np.zeros(n)
    grad_C_vector[0] = 0
    for i in xrange(1, n):
        grad_C_vector[i] = np.sum((C*grad_C_vector[:i]+view_predict[:i]) * (rep(i, c)**(-1-theta)))
    grad_C = np.sum((view_predict-y)*grad_C_vector)/n
    # partial derivative for c
    grad_c_vector = np.zeros(n)
    grad_c_vector[0] = 0
    for i in xrange(1, n):
        grad_c_vector[i] = C*np.sum((grad_c_vector[:i]-(1+theta)*view_predict[:i]/rep(i, c)) * (rep(i, c)**(-1-theta)))
    grad_c = np.sum((view_predict-y)*grad_c_vector)/n
    # partial derivative for gamma
    grad_gamma_vector = np.zeros(n)
    grad_gamma_vector[0] = 1
    for i in xrange(1, n):
        grad_gamma_vector[i] = C*np.sum(grad_gamma_vector[:i] * (rep(i, c)**(-1-theta)))
    grad_gamma = np.sum((view_predict-y)*grad_gamma_vector)/n
    # partial derivative for eta
    grad_eta_vector = np.zeros(n)
    grad_eta_vector[0] = 0
    for i in xrange(1, n):
        grad_eta_vector[i] = 1 + C*np.sum(grad_eta_vector[:i] * (rep(i, c)**(-1-theta)))
    grad_eta = np.sum((view_predict-y)*grad_eta_vector)/n
    return np.array([grad_mu, grad_theta, grad_C, grad_c, grad_gamma, grad_eta])


def reg_grad_descent(params, x, y, params0):
    """
    Regularized gradient function for HIP model
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount
    :param y: observed viewcount
    :param params0: reference values from non-regularized model
    :return: cost function value
    """
    mu, theta, C, c, gamma, eta = params
    mu0, C0, gamma0, eta0, w = params0
    view_predict = predict(params, x)
    n = len(x)
    # partial derivative for mu
    grad_mu_vector = np.zeros(n)
    grad_mu_vector[0] = x[0]
    for i in xrange(1, n):
        grad_mu_vector[i] = x[i] + C*np.sum(grad_mu_vector[:i] * (rep(i, c)**(-1-theta)))
    grad_mu = (np.sum((view_predict-y)*grad_mu_vector) + w*mu/mu0/mu0)/n
    # partial derivative for theta
    grad_theta_vector = np.zeros(n)
    grad_theta_vector[0] = 0
    for i in xrange(1, n):
        grad_theta_vector[i] = C*np.sum((grad_theta_vector[:i]-view_predict[:i]*np.log(rep(i, c))) * (rep(i, c)**(-1-theta)))
    grad_theta = np.sum((view_predict-y)*grad_theta_vector)/n
    # partial derivative for C
    grad_C_vector = np.zeros(n)
    grad_C_vector[0] = 0
    for i in xrange(1, n):
        grad_C_vector[i] = np.sum((C*grad_C_vector[:i]+view_predict[:i]) * (rep(i, c)**(-1-theta)))
    grad_C = (np.sum((view_predict-y)*grad_C_vector) + w*C/C0/C0)/n
    # partial derivative for c
    grad_c_vector = np.zeros(n)
    grad_c_vector[0] = 0
    for i in xrange(1, n):
        grad_c_vector[i] = C*np.sum((grad_c_vector[:i]-(1+theta)*view_predict[:i]/rep(i, c)) * (rep(i, c)**(-1-theta)))
    grad_c = np.sum((view_predict-y)*grad_c_vector)/n
    # partial derivative for gamma
    grad_gamma_vector = np.zeros(n)
    grad_gamma_vector[0] = 1
    for i in xrange(1, n):
        grad_gamma_vector[i] = C*np.sum(grad_gamma_vector[:i] * (rep(i, c)**(-1-theta)))
    grad_gamma = (np.sum((view_predict-y)*grad_gamma_vector) + w*gamma/gamma0/gamma0)/n
    # partial derivative for eta
    grad_eta_vector = np.zeros(n)
    grad_eta_vector[0] = 0
    for i in xrange(1, n):
        grad_eta_vector[i] = 1 + C*np.sum(grad_eta_vector[:i] * (rep(i, c)**(-1-theta)))
    grad_eta = (np.sum((view_predict-y)*grad_eta_vector) + w*eta/eta0/eta0)/n
    return np.array([grad_mu, grad_theta, grad_C, grad_c, grad_gamma, grad_eta])


def predict(params, x):
    """
    Predict viewcount with sharecount sequence x
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount sequence from beginning
    :return: predict value
    """
    mu, theta, C, c, gamma, eta = params
    n = len(x)
    x_predict = np.zeros(len(x))
    for i in xrange(n):
        if i == 0:
            x_predict[0] = gamma + mu * x[0]
        else:
            x_predict[i] = eta + mu*x[i] + C*np.sum(x_predict[:i]*((np.arange(1, i+1)[::-1]+c)**(-1-theta)))
    return x_predict


def test_predict(params, x, y, title, idx, pred_params=None):
    """
    Test predict function
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount
    :param y: observed viewcount
    :param title: figure title, YoutubeID
    :param idx: subplot index
    :param pred_params: fitted parameters
    :return: 
    """
    # visualise sample data
    ax1 = fig.add_subplot(221+idx)
    ax2 = ax1.twinx()
    ax1.plot(np.arange(1, 121), y, 'k--', label='observed #views')
    ax2.plot(np.arange(1, 121), x, 'r-', label='#share')

    ax1.set_ylim(ymin=max(0, ax1.get_ylim()[0]))
    ax2.set_ylim(ymax=3*max(x))
    ax1.set_xlabel('video age (day)')
    ax1.set_ylabel('Number of views', color='k')
    ax1.tick_params('y', colors='k')
    ax2.set_ylabel('Number of shares', color='r')
    ax2.tick_params('y', colors='r')

    mu, theta, C, c, gamma, eta = params
    ax2.text(0.03, 0.80, 'WWW\n$\mu$={0:.2e}, $\\theta$={1:.2e}\nC={2:.2e}, c={3:.2e}\n$\gamma$={4:.2e}, $\eta$={5:.2e}'
             .format(mu, theta, C, c, gamma, eta), transform=ax1.transAxes)

    x_www = predict(params, x)
    ax1.plot(np.arange(1, 121), x_www, 'b-', label='WWW popularity')
    ax1.set_title(title, fontdict={'fontsize': 15})

    if pred_params is not None:
        pred_mu, pred_theta, pred_C, pred_c, pred_gamma, pred_eta = pred_params
        ax2.text(0.55, 0.80, 'HIP\n$\mu$={0:.2e}, $\\theta$={1:.2e}\nC={2:.2e}, c={3:.2e}\n$\gamma$={4:.2e}, $\eta$={5:.2e}'
                 .format(pred_mu, pred_theta, pred_C, pred_c, pred_gamma, pred_eta), transform=ax1.transAxes)
        x_predict = predict(pred_params, x)
        ax1.plot(np.arange(1, 121), x_predict, 'g-', label='HIP popularity')


if __name__ == '__main__':
    fig = plt.figure(figsize=(14, 10))

    # == == == == == == == == Part 1: Generate test cases == == == == == == == == #
    # test case 1: YoutubeID bUORBT9iFKc
    dailyshare1 = [54, 40, 24, 26, 25, 12, 8, 36, 36, 32, 55, 31, 19, 22, 26, 29, 31, 45, 31, 24, 24, 26, 23, 33, 49,
                   39, 28, 28, 36, 40, 31, 57, 35, 39, 40, 42, 33, 58, 50, 31, 29, 27, 26, 36, 34, 52, 28, 23, 17, 11,
                   16, 17, 45, 21, 19, 28, 32, 26, 28, 31, 17, 19, 30, 19, 15, 21, 51, 33, 21, 16, 13, 19, 21, 35, 38,
                   18, 17, 25, 18, 23, 37, 18, 22, 14, 19, 23, 14, 18, 15, 21, 18, 18, 24, 17, 29, 14, 12, 10, 13, 12,
                   15, 28, 17, 15, 11, 14, 22, 12, 12, 15, 9, 15, 8, 10, 9, 18, 13, 7, 11, 14, 12, 9, 13, 7, 6, 19, 10,
                   7, 9, 15, 9, 7, 13, 22, 6, 4, 10, 9, 3, 13, 13, 9, 6, 9, 11, 7, 9, 10, 0, 11, 19, 9, 10]
    dailyview1 = [7903, 9842, 9985, 8687, 6902, 4709, 5846, 14451, 15425, 13040, 14285, 10695, 11635, 12792, 11064,
                  10767, 11136, 12910, 12566, 10264, 11190, 10027, 11689, 12128, 13275, 13374, 10155, 13200, 12686,
                  13940, 11717, 15163, 16111, 11827, 12688, 12652, 14080, 16067, 18019, 16479, 12593, 13565, 12566,
                  13418, 13885, 15320, 13552, 10732, 10525, 9584, 11972, 11983, 14046, 13043, 9858, 14198, 13271, 13520,
                  13250, 14149, 14055, 10690, 12624, 13705, 13208, 14130, 16700, 18319, 12262, 12854, 12155, 11497,
                  13203, 15245, 14111, 11582, 12625, 10184, 12805, 13197, 13465, 11992, 9007, 10439, 10457, 11160,
                  12304, 12100, 11017, 10985, 11837, 11626, 12776, 11987, 13501, 12229, 8684, 10010, 10538, 10596,
                  10903, 11606, 11610, 9101, 10441, 10694, 10298, 10291, 12208, 10462, 8595, 9629, 9750, 10173, 10331,
                  11315, 10424, 8424, 10639, 11632, 11022, 11832, 11366, 10212, 9124, 10277, 12155, 12918, 7436, 9996,
                  12580, 11867, 14391, 15402, 14171, 8807, 12615, 12650, 12025, 14176, 13717, 13442, 11590, 13511,
                  12917, 9576, 10847, 11229, 12049, 12609, 12686, 11398, 8765]
    # trim to the first 120 days
    dailyshare1 = np.array(dailyshare1[:120])
    dailyview1 = np.array(dailyview1[:120])
    test_params1 = [7.985706e+01, 5.371356922, 8.298009e-03, -4.691019e-01, 3.015634e+02, 4.336e+03]

    # test case 2: YoutubeID WKJoBeeSWhc
    dailyshare2 = [10, 7, 10, 8, 14, 20, 21, 12, 11, 6, 3, 3, 8, 3, 4, 3, 0, 3, 4, 0, 1, 0, 2, 0, 3, 1, 2, 4, 3, 11, 1,
                   1, 3, 2, 2, 2, 1, 1, 0, 2, 3, 3, 6, 1, 4, 6, 6, 13, 8, 8, 12, 9, 5, 5, 11, 10, 19, 11, 26, 5, 5, 5,
                   2, 16, 12, 6, 8, 4, 15, 5, 13, 20, 19, 23, 15, 10, 16, 11, 17, 7, 15, 5, 13, 14, 11, 15, 20, 8, 19,
                   10, 14, 26, 29, 13, 28, 10, 10, 16, 18, 14, 14, 11, 12, 13, 12, 14, 6, 16, 9, 10, 11, 16, 16, 20, 13,
                   19, 7, 11, 12, 15, 14, 10, 19, 26, 20, 19, 8, 3, 9, 9, 9, 7, 8, 6, 7, 4, 5, 12, 3, 10, 15, 9, 16, 12,
                   11, 6, 3, 16, 12, 15, 8, 5, 2, 13, 15, 15, 10, 9, 13, 8, 15, 22, 17, 17, 10, 13, 9, 13, 17, 6, 15,
                   13, 6, 13, 11, 30, 19, 6, 9, 3, 23, 26, 11, 9, 14, 8, 9, 16, 13, 13, 9, 9, 6, 10, 10, 11, 22, 12, 1,
                   5, 7, 10, 9, 10, 20, 11, 10, 12, 17]
    dailyview2 = [918, 937, 835, 998, 635, 903, 624, 592, 481, 430, 293, 279, 309, 476, 395, 382, 270, 284, 290, 257,
                  291, 291, 224, 165, 178, 157, 221, 205, 310, 291, 215, 213, 372, 293, 373, 164, 169, 306, 349, 410,
                  234, 284, 489, 675, 745, 727, 832, 781, 943, 1405, 1155, 963, 992, 1022, 1280, 1165, 1391, 1781, 2687,
                  1271, 1323, 990, 1120, 1421, 1321, 1553, 1589, 1610, 1420, 1267, 1563, 1573, 2743, 3201, 3365, 1896,
                  2136, 2341, 2622, 2353, 2257, 2416, 2513, 2790, 3167, 2566, 2236, 2426, 2514, 2435, 2305, 3288, 4083,
                  2981, 2294, 2398, 2171, 2717, 3319, 2750, 2449, 2514, 2568, 2720, 2854, 2880, 2356, 1931, 2428, 2545,
                  2405, 2525, 2973, 2978, 2894, 2698, 2284, 2267, 2189, 2877, 2702, 2540, 2547, 3150, 3117, 2861, 2928,
                  2462, 2373, 2315, 2077, 2340, 2540, 2525, 2158, 2190, 2275, 2213, 2252, 2501, 3036, 2629, 2438, 2607,
                  2674, 2298, 2717, 2931, 2845, 2638, 2597, 2385, 2182, 2805, 3190, 2880, 2551, 2761, 2430, 2356, 3073,
                  3703, 2734, 2780, 2634, 2693, 2999, 3105, 2840, 2593, 2401, 3079, 3138, 3012, 3408, 3735, 3006, 2869,
                  2286, 2296, 2978, 3292, 3723, 2957, 2849, 2804, 2331, 2415, 2989, 3244, 2615, 2351, 2541, 2828, 3045,
                  3284, 3487, 2449, 2592, 2009, 2383, 3359, 3127, 3518, 3381, 3434, 3407, 2980, 3485]
    # trim to the first 120 days
    dailyshare2 = np.array(dailyshare2[:120])
    dailyview2 = np.array(dailyview2[:120])
    test_params2 = [4.285222e+01, 0.413187407, 9.593957e-01, 3.267503e+00, 1.316225e+01, 4.359545e-10]

    # test case 3: YoutubeID 3UTHH8GRuQY
    dailyshare3 = [45, 201, 237, 175, 77, 47, 39, 20, 18, 8, 4, 5, 20, 17, 12, 8, 4, 12, 12, 10, 6, 9, 7, 10, 8, 6, 4,
                   10, 3, 8, 10, 7, 4, 6, 14, 9, 5, 6, 2, 10, 8, 9, 6, 10, 5, 3, 13, 8, 12, 12, 12, 14, 11, 5, 17, 14,
                   11, 8, 5, 7, 7, 4, 8, 16, 11, 13, 14, 3, 14, 14, 21, 17, 10, 10, 21, 17, 17, 22, 14, 8, 9, 12, 13,
                   10, 10, 9, 8, 15, 6, 5, 11, 8, 12, 10, 5, 7, 10, 13, 16, 15, 9, 11, 10, 12, 9, 6, 13, 16, 9, 5, 6,
                   18, 14, 10, 11, 17, 6, 17, 14, 19, 15, 21, 6, 19, 11, 17, 26, 28, 26, 21, 22, 13, 20, 38, 25, 25, 19,
                   8, 22, 34, 35, 42, 43, 44, 22, 31, 52, 44, 37, 26, 28, 24, 41, 32, 41, 34, 27, 26, 15, 25, 25, 34,
                   30, 28, 24, 15, 22, 15, 19, 22, 29, 10, 11, 19, 14, 16, 13, 16, 12, 6, 10, 16, 6, 16, 12, 21, 9, 5,
                   7, 8, 3, 2, 5, 5, 7, 5, 3, 3, 3, 6, 5, 4, 10, 2, 3, 0, 5, 1, 3]
    dailyview3 = [4079, 58295, 44519, 51725, 26461, 18558, 12439, 5853, 3199, 2088, 1822, 2252, 3381, 3166, 2107, 1743,
                  1710, 3287, 1905, 1952, 2226, 2314, 1972, 2466, 2025, 1920, 2074, 1948, 1962, 1915, 1573, 1759, 1587,
                  2073, 2414, 2080, 1576, 1175, 1319, 1179, 1398, 1814, 1072, 1110, 658, 598, 660, 829, 713, 806, 688,
                  686, 605, 524, 567, 720, 726, 656, 675, 683, 541, 630, 771, 713, 782, 728, 673, 655, 822, 1043, 1056,
                  1164, 1023, 888, 816, 916, 1066, 931, 960, 813, 773, 579, 622, 708, 594, 564, 471, 497, 444, 561, 601,
                  482, 509, 445, 541, 456, 463, 540, 500, 762, 531, 363, 410, 491, 666, 669, 618, 671, 549, 604, 618,
                  787, 772, 757, 794, 722, 684, 809, 954, 904, 945, 941, 729, 695, 829, 1055, 1159, 1273, 1446, 1098,
                  1110, 1137, 1679, 2366, 1420, 1338, 1219, 1206, 1480, 1611, 1629, 1764, 1506, 1404, 1358, 1647, 1563,
                  2022, 1707, 1521, 1501, 1547, 1954, 1857, 1615, 1575, 1290, 1087, 1051, 1434, 1401, 1411, 1456, 1378,
                  1091, 921, 1115, 1449, 1176, 985, 1053, 867, 688, 859, 901, 856, 750, 757, 583, 520, 1280, 1787, 1585,
                  694, 536, 527, 548, 530, 588, 370, 230, 246, 400, 351, 491, 395, 330, 214, 219, 274, 314, 287, 270,
                  328, 242, 278, 209, 270, 248]
    # trim to the first 120 days
    dailyshare3 = np.array(dailyshare3[:120])
    dailyview3 = np.array(dailyview3[:120])
    test_params3 = [5.208709e+01, 21.795834351, 6.111318e-01, 4.848469e+00, 2.131529e-13, 3.995517e-01]

    # test case 4: YoutubeID 1PuvXpv0yDM
    dailyshare4 = [0, 0, 0, 0, 1, 0, 1, 0, 0, 6, 11, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 27, 68, 17, 20, 7, 5, 7, 14, 12, 10, 8, 5, 0, 17, 8, 2, 0, 2, 3, 1, 0, 4, 4, 0, 0, 2, 1, 0,
                   1, 0, 0, 0, 1, 18, 5, 0, 3, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 1, 0, 0, 0, 0,
                   2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 16, 6, 7]
    dailyview4 = [196, 120, 11, 5, 49, 189, 40, 19, 12, 695, 1587, 563, 190, 52, 60, 62, 92, 43, 58, 66, 59, 57, 77, 60,
                  78, 57, 66, 69, 76, 77, 134, 122, 105, 110, 140, 132, 115, 125, 115, 136, 131, 133, 148, 139, 145,
                  155, 142, 142, 208, 183, 167, 163, 300, 173, 125, 159, 124, 171, 195, 201, 166, 152, 315, 187, 201,
                  198, 194, 172, 216, 190, 197, 199, 206, 208, 174, 191, 222, 239, 243, 224, 212, 165, 169, 209, 220,
                  256, 254, 297, 277, 211, 216, 246, 259, 259, 237, 255, 206, 231, 264, 227, 244, 247, 221, 205, 415431,
                  426721, 148240, 190181, 90269, 64850, 106321, 309221, 244289, 95555, 37582, 32771, 22207, 138428,
                  74847, 21808, 11179, 20855, 10961, 4854, 4281, 6584, 5285, 3282, 2658, 2544, 3106, 6826, 4083, 3088,
                  2660, 2337, 1465, 14098, 10888, 9537, 4304, 2473, 2018, 1505, 1515, 1841, 1855, 1420, 1361, 1326,
                  1256, 1287, 1654, 1345, 1327, 1611, 1740, 1053, 1004, 957, 1502, 1724, 1643, 1490, 2296, 1227, 931,
                  930, 2744, 646, 477, 499, 815, 758, 691, 598, 922, 483, 716, 1179, 1519, 1033, 2357, 21912, 293011,
                  206816, 7505]
    # trim to the first 120 days
    dailyshare4 = np.array(dailyshare4[:120])
    dailyview4 = np.array(dailyview4[:120])
    test_params4 = [1.190261e+02, -1.552484832, 1.850818e-03, 5.000000e+00, 7.150544e+01, 3.771896e+01]

    test_cases = {'bUORBT9iFKc': [test_params1, dailyshare1, dailyview1],
                  'WKJoBeeSWhc': [test_params2, dailyshare2, dailyview2],
                  '3UTHH8GRuQY': [test_params3, dailyshare3, dailyview3],
                  '1PuvXpv0yDM': [test_params4, dailyshare4, dailyview4]}

    # # == == == == == == == == Part 2: Test predict function == == == == == == == == #
    # for idx, vid in enumerate(test_cases.keys()):
    #     test_params, dailyshare, dailyview = test_cases[vid]
    #     test_predict(test_params, dailyshare, dailyview, vid, idx)

    # == == == == == == == == Part 3: Test cost and grad function == == == == == == == == #
    # setting parameters
    iteration = 100
    num_train = 75
    num_cv = 15
    num_test = 30
    bounds = [(0, None), (np.finfo(float).eps, 100), (0, None), (np.finfo(float).eps, 5), (0, None), (0, None)]
    for idx, vid in enumerate(test_cases.keys()):
        test_params, dailyshare, dailyview = test_cases[vid]

        x_train = dailyshare[: num_train]
        y_train = dailyview[: num_train]
        x_cv = dailyshare[: num_train+num_cv]
        y_cv = dailyview[: num_train+num_cv]
        x_test = dailyshare[: num_train+num_cv+num_test]
        y_test = dailyview[: num_train+num_cv+num_test]

        # initialize weights
        # 3 sets of fixed params and k sets of random
        k = 5
        initial_theta_sets = rand_initialize_weights(k)

        best_reg_J0_params = None
        best_reg_param0 = None
        best_cost = np.inf
        # find best optimizers within those k+3 sets of params
        print 'Test case vid: {0}'.format(vid)
        print '\ttarget training cost: {0:>6.4e}'.format(cost_function(test_params, x_train, y_train))
        for epoch_idx, initial_theta in enumerate(initial_theta_sets):
            # perform regularize linear regression with l-bfgs
            optimizer = optimize.minimize(cost_function, initial_theta, jac=grad_descent, method='L-BFGS-B',
                                          args=(x_train, y_train), bounds=bounds,
                                          options={'disp': None, 'maxiter': iteration})

            print '\tepoch{0}: non-regularized cost: {1:>6.4e}'.format(epoch_idx, optimizer.fun)

            # == == == == == == == == Part 4: Test regularized cost and grad function == == == == == == == == #
            mu0, theta0, C0, c0, gamma0, eta0 = optimizer.x
            J0 = optimizer.fun
            for w in np.arange(np.log(10**-4*J0), np.log(10*J0)):
                w0 = np.exp(w)
                reg_param0 = np.array([mu0, C0, gamma0, eta0, w0])
                reg_optimizer = optimize.minimize(reg_cost_function, optimizer.x, jac=reg_grad_descent, method='L-BFGS-B',
                                                  args=(x_train, y_train, reg_param0), bounds=bounds,
                                                  options={'disp': None, 'maxiter': iteration})
                # cross validate by using cv dataset
                x_predict = predict(reg_optimizer.x, x_cv)
                cv_cost = cost_function(reg_optimizer.x, x_cv, y_cv, num_split=num_cv)
                if cv_cost < best_cost:
                    best_reg_J0_params = optimizer.x
                    best_reg_param0 = reg_param0
                    best_cost = cv_cost

        # train with optimal initialization and w
        reg_optimizer = optimize.minimize(reg_cost_function, best_reg_J0_params, jac=reg_grad_descent, method='L-BFGS-B',
                                          args=(x_cv, y_cv, best_reg_param0), bounds=bounds,
                                          options={'disp': None, 'maxiter': iteration})
        print '+'*79
        print '+      target test cost for {0}: {1:>6.4e}'.format(vid, cost_function(test_params, x_test, y_test, num_split=num_test))
        print '+ regularized test cost for {0}: {1:>6.4e} @best w: {2:>6.4e}'.format(vid, cost_function(reg_optimizer.x, x_test, y_test, num_split=num_test), best_reg_param0[-1])
        print '+'*79
        test_predict(test_params, dailyshare, dailyview, vid, idx, pred_params=reg_optimizer.x)

    # plt.legend()
    plt.tight_layout()
    plt.show()
