import numpy as np
import autograd.numpy as np
import cPickle as pickle
from scipy import optimize
from autograd import grad, jacobian


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
    a = np.array([1, 0.2, get_C(0.024), 0.001, 100, 100])
    b = np.array([0.2, 10, get_C(3), 4.5, 10, 100])
    c = np.array([1, 3.35, get_C(0.0024), 0.0001, 100, 10])
    ret = [a, b, c]
    for _ in xrange(n):
        rand_mu = np.random.uniform(10, 165)
        rand_theta = np.random.uniform(3.042, 19.370)
        rand_C = np.random.uniform(get_C(0), get_C(5))
        rand_c = np.random.uniform(0, 1.9550)
        rand_gamma = np.random.uniform(0, 739)
        rand_eta = np.random.uniform(0.12, 119.10)
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
    n = len(x)
    view_predict = predict(params, x)
    cost_vector = view_predict - y
    if num_split is not None:
        cost_vector = cost_vector[-num_split:]
    cost = np.sum(cost_vector ** 2) / 2
    return cost/n


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
    grad_mu = np.sum((view_predict-y)*grad_mu_vector)
    # partial derivative for theta
    grad_theta_vector = np.zeros(n)
    grad_theta_vector[0] = 0
    for i in xrange(1, n):
        grad_theta_vector[i] = C*np.sum((grad_theta_vector[:i]-view_predict[:i]*np.log(rep(i, c))) * (rep(i, c)**(-1-theta)))
    grad_theta = np.sum((view_predict-y)*grad_theta_vector)
    # partial derivative for C
    grad_C_vector = np.zeros(n)
    grad_C_vector[0] = 0
    for i in xrange(1, n):
        grad_C_vector[i] = np.sum((C*grad_C_vector[:i]+view_predict[:i]) * (rep(i, c)**(-1-theta)))
    grad_C = np.sum((view_predict-y)*grad_C_vector)
    # partial derivative for c
    grad_c_vector = np.zeros(n)
    grad_c_vector[0] = 0
    for i in xrange(1, n):
        grad_c_vector[i] = C*np.sum((grad_c_vector[:i]-(1+theta)*view_predict[:i]/rep(i, c)) * (rep(i, c)**(-1-theta)))
    grad_c = np.sum((view_predict-y)*grad_c_vector)
    # partial derivative for gamma
    grad_gamma_vector = np.zeros(n)
    grad_gamma_vector[0] = 1
    for i in xrange(1, n):
        grad_gamma_vector[i] = C*np.sum(grad_gamma_vector[:i] * (rep(i, c)**(-1-theta)))
    grad_gamma = np.sum((view_predict-y)*grad_gamma_vector)
    # partial derivative for eta
    grad_eta_vector = np.zeros(n)
    grad_eta_vector[0] = 0
    for i in xrange(1, n):
        grad_eta_vector[i] = 1 + C*np.sum(grad_eta_vector[:i] * (rep(i, c)**(-1-theta)))
    grad_eta = np.sum((view_predict-y)*grad_eta_vector)
    return np.array([grad_mu, grad_theta, grad_C, grad_c, grad_gamma, grad_eta])/n
    # return grad_mu/n


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
            try:
                x_predict[0] = gamma + mu * x[0]
            except Exception as e:
                print 'x[0]-------------------'
                print str(e)
                print mu, theta, C, c, gamma, eta
        else:
            try:
                x_predict[i] = eta + mu*x[i] + C*np.sum(x_predict[:i]*((np.arange(1, i+1)[::-1]+c)**(-1-theta)))
            except Exception as e:
                print 'x[{0}]-------------------'.format(i)
                print str(e)
                print mu, theta, C, c, gamma, eta
    return x_predict


def close_enough(prime, fprime, tol=10e-2):
    """
    Function that determines whether values in two numpy arrays are close enough
    :param prime: numpy array, gradient value calculated by manually derivative
    :param fprime: numpy array, approximate value returned by optimize.approx_fprime function
    :param tol: tolerance of difference
    :return: (close)true or (not close)false
    """
    ret = 0
    helper = []
    for i, j in zip(prime, fprime):
        diff = abs(i-j)
        if diff < tol:
            ret += 0
            helper.append(0)
        else:
            relative_diff = diff/i
            ret += relative_diff**2
            helper.append(relative_diff)
    if not np.sqrt(ret) < tol:
        print prime
        print fprime
        print helper
    return np.sqrt(ret) < tol


if __name__ == '__main__':
    # == == == == == == == == Part 1: Generate test cases == == == == == == == == #
    test_cases = pickle.load(open('active_pars.p', 'rb'))
    # random select 4 videos
    test_videos = np.array(test_cases.keys())
    random_index = np.random.randint(0, len(test_videos), 4)
    test_vids = test_videos[random_index]

    # == == == == == == == == Part 3: Test cost and grad function == == == == == == == == #
    # setting parameters
    age = 120
    iteration = 100
    num_train = 75
    num_cv = 15
    num_test = 30
    eps = np.finfo(float).eps

    # Define a function that returns gradients of training loss using autograd.

    for tc_idx, vid in enumerate(test_vids):
        test_params, dailyshare, dailyview = test_cases[vid]
        dailyshare = dailyshare[:age]
        dailyview = dailyview[:age]

        x_train = dailyshare[: num_train]
        y_train = dailyview[: num_train]
        x_cv = dailyshare[: num_train + num_cv]
        y_cv = dailyview[: num_train + num_cv]
        x_test = dailyshare[: num_train + num_cv + num_test]
        y_test = dailyview[: num_train + num_cv + num_test]

        # initialize weights
        # 3 sets of fixed params and k sets of random
        k = 5
        initial_theta_sets = rand_initialize_weights(k)
        initial_theta_sets.insert(0, test_params)

        print 'err value for test case {0}'.format(tc_idx)
        for init_idx, initial_theta in enumerate(initial_theta_sets):

            prime = grad_descent(initial_theta, x_train, y_train)
            fprime = optimize.approx_fprime(initial_theta, cost_function, np.sqrt(np.finfo(float).eps), x_train, y_train)

            training_gradient_fun = grad(cost_function)
            auto_diff = training_gradient_fun(np.array(initial_theta), x_train, y_train)
            # print auto_diff

            print close_enough(prime, fprime)

            # print training_gradient_fun(np.array(initial_theta), x_train, y_train)
            print '----------------------'
