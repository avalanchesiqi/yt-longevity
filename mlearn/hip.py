import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Hawkes Intensity Process model


def rand_initialize_weights(n):
    """
    Initialize random weights for theta
    :param n: number of features
    :return: random vector with all values bounded within epsilon
    """
    # return 10*np.random.rand(n)
    return np.random.uniform(0, sys.maxint, n)


def rep(i, c):
    """
    Replacement for sequence (tau + c)
    :param i: tau value
    :param c: c value
    :return: abbreviated presentation
    """
    return np.arange(1, i+1)[::-1]+c


def cost_function(params, x, y):
    """
    Regularized cost function for HIP model
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount
    :param y: observed viewcount
    # :param w: regularization strength
    :return: cost function value
    """
    view_predict = predict(params, x)
    cost_vector = view_predict - y
    cost = np.sum(cost_vector ** 2) / 2
    return cost


def grad_descent(params, x, y):
    """
    Gradient function for HIP model
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount
    :param y: observed viewcount
    # :param w: regularization strength
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
    ax1 = fig.add_subplot(121+idx)
    ax2 = ax1.twinx()
    ax1.plot(np.arange(1, 121), y, 'k--', label='observed #views')
    ax2.plot(np.arange(1, 121), x, 'r-', label='#share')

    ax2.set_ylim(ymax=3*max(x))
    ax1.set_xlabel('video age (day)')
    ax1.set_ylabel('Number of views', color='k')
    ax1.tick_params('y', colors='k')
    ax2.set_ylabel('Number of shares', color='r')
    ax2.tick_params('y', colors='r')

    mu, theta, C, c, gamma, eta = params
    ax2.text(0.03, 0.82, 'WWW\n$\mu$={0:.2f}, $\\theta$={1:.2f}\nC={2:.3f}, c={3:.2f}\n$\gamma$={4:.2f}, $\eta$={5}'
             .format(mu, theta, C, c, gamma, eta), transform=ax1.transAxes)

    x_www = predict(params, x)
    ax1.plot(np.arange(1, 121), x_www, 'b-', label='WWW popularity')
    ax1.set_title(title, fontdict={'fontsize': 15})

    if pred_params is not None:
        pred_mu, pred_theta, pred_C, pred_c, pred_gamma, pred_eta = pred_params
        ax2.text(0.63, 0.82, 'HIP\n$\mu$={0:.2f}, $\\theta$={1:.2e}\nC={2:.2e}, c={3:.2e}\n$\gamma$={4:.2f}, $\eta$={5:.2e}'
                 .format(pred_mu, pred_theta, pred_C, pred_c, pred_gamma, pred_eta), transform=ax1.transAxes)
        x_predict = predict(pred_params, x)
        ax1.plot(np.arange(1, 121), x_predict, 'g-', label='HIP popularity')


if __name__ == '__main__':
    fig = plt.figure(figsize=(16, 6))

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

    test_cases = {'bUORBT9iFKc': [test_params1, dailyshare1, dailyview1],
                  'WKJoBeeSWhc': [test_params2, dailyshare2, dailyview2]}

    # # == == == == == == == == Part 2: Test predict function == == == == == == == == #
    # for idx, vid in enumerate(test_cases.keys()):
    #     test_params, dailyshare, dailyview = test_cases[vid]
    #     test_predict(test_params, dailyshare, dailyview, vid, idx)

    # == == == == == == == == Part 3: Test cost and grad function == == == == == == == == #
    for idx, vid in enumerate(test_cases.keys()):
        test_params, dailyshare, dailyview = test_cases[vid]
        # setting parameters
        iteration = 100
        num_train = 90

        # initialize weights
        initial_theta = rand_initialize_weights(6)

        # perform regularize linear regression with l-bfgs
        optimizer = optimize.minimize(cost_function, initial_theta, jac=grad_descent, method='L-BFGS-B',
                                      args=(dailyshare[:num_train], dailyview[:num_train]),
                                      options={'disp': None, 'maxiter': iteration})

        pred_view = predict(optimizer.x, dailyshare[:num_train])
        print 'target cost value for test case {0}: {1}'.format(idx+1, cost_function(test_params, dailyshare[:num_train], dailyview[:num_train]))
        print ' final cost value for test case {0}: {1}'.format(idx+1, cost_function(optimizer.x, dailyshare[:num_train], dailyview[:num_train]))
        test_predict(test_params, dailyshare, dailyview, vid, idx, pred_params=optimizer.x)

    # plt.legend()
    plt.tight_layout()
    plt.show()
