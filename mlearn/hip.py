import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Hawkes Intensity Process model

cost_list = []
def rand_initialize_weights(n):
    """
    Initialize random weights for theta
    :param n: number of features
    :return: random vector with all values bounded within epsilon
    """
    return 10*np.random.rand(n)


def cost_function(params, x, y):
    """
    Regularized cost function for HIP model
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount
    :param y: observed viewcount
    # :param w: regularization strength
    :return: cost function value
    """
    mu, theta, C, c, gamma, eta = params
    n = len(x)
    cost_vector = np.zeros(n)
    # when t equals to 0
    cost_vector[0] = (gamma + mu*x[0] - y[0])
    # when t is greater than 0
    for i in xrange(1, n):
        cost_vector[i] = (eta + mu*x[i] + C*np.sum(view_predict[:i]*((np.arange(1, i+1)[::-1]+c)**(-1-theta))) - y[i])
    cost = np.sum(cost_vector**2)/2
    cost_list.append(cost)
    return cost


def rep(i, c):
    """
    Replacement for sequence (tau + c)
    :param i: tau value
    :param c: c value
    :return: abbreviated presentation
    """
    return np.arange(1, i+1)[::-1]+c


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
    n = len(x)
    # partial derivative for mu
    grad_mu_vector = np.zeros(n)
    grad_mu_vector[0] = (view_predict[0]-y[0])*x[0]
    if n > 1:
        for i in xrange(1, n):
            grad_mu_vector[i] = (view_predict[i] - y[i])*(
                x[i] + C*np.sum(grad_mu_vector[:i] * (rep(i, c)**(-1-theta))))
    # partial derivative for theta
    grad_theta_vector = np.zeros(n)
    grad_theta_vector[0] = 0
    if n > 1:
        for i in xrange(1, n):
            grad_theta_vector[i] = (view_predict[i] - y[i]) * (
                C*np.sum((grad_theta_vector[:i] - view_predict[:i]*(np.log(rep(i, c)))) * (rep(i, c)**(-1-theta))))
    # partial derivative for C
    grad_C_vector = np.zeros(n)
    grad_C_vector[0] = 0
    if n > 1:
        for i in xrange(1, n):
            grad_C_vector[i] = (view_predict[i] - y[i]) * (
                np.sum((C*grad_C_vector[:i] + view_predict[:i]) * (rep(i, c)**(-1-theta))))
    # partial derivative for c
    grad_c_vector = np.zeros(n)
    grad_c_vector[0] = 0
    if n > 1:
        for i in xrange(1, n):
            grad_c_vector[i] = (view_predict[i] - y[i]) * (
                C*np.sum((grad_c_vector[:i] - (1+theta)*view_predict[:i]/(rep(i, c))) * (rep(i, c)**(-1-theta))))
    # partial derivative for gamma
    grad_gamma_vector = np.zeros(n)
    grad_gamma_vector[0] = 1
    if n > 1:
        for i in xrange(1, n):
            grad_gamma_vector[i] = (view_predict[i] - y[i]) * (
                C*np.sum(grad_gamma_vector[:i] * (rep(i, c)**(-1-theta))))
    # partial derivative for eta
    grad_eta_vector = np.zeros(n)
    grad_eta_vector[0] = 0
    if n > 1:
        for i in xrange(1, n):
            grad_eta_vector[i] = (view_predict[i] - y[i]) * (
                1 + C*np.sum(grad_eta_vector[:i] * (rep(i, c)**(-1-theta))))
    return np.array([grad_mu_vector[-1], grad_theta_vector[-1], grad_C_vector[-1], grad_c_vector[-1],
                     grad_gamma_vector[-1], grad_eta_vector[-1]])


def predict(params, x):
    """
    Predict viewcount with sharecount sequence x
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount sequence from beginning
    :return: predict value
    """
    mu, theta, C, c, gamma, eta = params
    n = len(x)
    for i in xrange(n):
        if i == 0:
            view_predict[0] = gamma + mu * x[0]
        else:
            view_predict[i] = eta + mu*x[i] + C*np.sum(view_predict[:i]*((np.arange(1, i+1)[::-1]+c)**(-1-theta)))
    viewcount = view_predict[n-1]
    return viewcount, view_predict


if __name__ == '__main__':
    # generate sample data
    dailyshare = [54,40,24,26,25,12,8,36,36,32,55,31,19,22,26,29,31,45,31,24,24,26,23,33,49,39,28,28,36,40,31,57,35,39,40,42,33,58,50,31,29,27,26,36,34,52,28,23,17,11,16,17,45,21,19,28,32,26,28,31,17,19,30,19,15,21,51,33,21,16,13,19,21,35,38,18,17,25,18,23,37,18,22,14,19,23,14,18,15,21,18,18,24,17,29,14,12,10,13,12,15,28,17,15,11,14,22,12,12,15,9,15,8,10,9,18,13,7,11,14,12,9,13,7,6,19,10,7,9,15,9,7,13,22,6,4,10,9,3,13,13,9,6,9,11,7,9,10,0,11,19,9,10]
    dailyview = [7903,9842,9985,8687,6902,4709,5846,14451,15425,13040,14285,10695,11635,12792,11064,10767,11136,12910,12566,10264,11190,10027,11689,12128,13275,13374,10155,13200,12686,13940,11717,15163,16111,11827,12688,12652,14080,16067,18019,16479,12593,13565,12566,13418,13885,15320,13552,10732,10525,9584,11972,11983,14046,13043,9858,14198,13271,13520,13250,14149,14055,10690,12624,13705,13208,14130,16700,18319,12262,12854,12155,11497,13203,15245,14111,11582,12625,10184,12805,13197,13465,11992,9007,10439,10457,11160,12304,12100,11017,10985,11837,11626,12776,11987,13501,12229,8684,10010,10538,10596,10903,11606,11610,9101,10441,10694,10298,10291,12208,10462,8595,9629,9750,10173,10331,11315,10424,8424,10639,11632,11022,11832,11366,10212,9124,10277,12155,12918,7436,9996,12580,11867,14391,15402,14171,8807,12615,12650,12025,14176,13717,13442,11590,13511,12917,9576,10847,11229,12049,12609,12686,11398,8765]
    # trim to the first 120 days
    dailyshare = np.array(dailyshare[:120])
    dailyview = np.array(dailyview[:120])
    view_predict = np.zeros(120)

    # training examples: 75
    # cross validation examples: 15
    # test examples: 30
    num_train = 75
    num_cv = 15
    num_test = 30

    # initialize weights
    initial_theta = rand_initialize_weights(6)

    # initialize historical predict value
    _, view_predict = predict(initial_theta, dailyshare[:num_train])
    # print 'view predict'
    # print view_predict[:75]

    ci = 0
    gi = 0

    # perform regularize linear regression with l-bfgs
    iteration = 25
    optimizer = optimize.minimize(cost_function, initial_theta, jac=grad_descent, method='L-BFGS-B',
                                  args=(dailyshare[:num_train], dailyview[:num_train]),
                                  options={'disp': None, 'maxiter': iteration})

    _, x_pred = predict(optimizer.x, dailyshare[:num_train])

    # visualise sample data, YoutubeID: bUORBT9iFKc
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax2 = ax1.twinx()
    # ax1.plot(np.arange(1, 121), dailyview, 'k--', label='observed #views')
    # ax2.plot(np.arange(1, 121), dailyshare, 'r-', label='#share')
    # ax2.set_ylim(ymax=180)

    plt.plot(np.arange(1, 1+len(cost_list)), cost_list)

    # plt.legend()
    plt.show()
