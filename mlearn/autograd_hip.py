from __future__ import print_function, division
import sys
import cPickle as pickle
import autograd.numpy as np
from autograd import grad
# import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings('error')

# Hawkes Intensity Process model in Python, grad by auto grad


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
    :return: n+4 sets of random vectors, in the order of mu, theta, C, c, gamma, eta
    """

    # 4 sets of fixed weights
    a = np.array([1, 0.2, get_C(0.024), 0.001, 100, 100])
    b = np.array([0.2, 10, get_C(3), 4.5, 10, 100])
    c = np.array([1, 3.35, get_C(0.0024), 0.0001, 100, 10])
    d = np.array([0.00001, 5, get_C(50), 1.5, 10000, 1000])
    ret = [a, b, c, d]
    for _ in xrange(n):
        rand_mu = np.random.uniform(0, 505.90)
        rand_theta = np.random.uniform(2.3, 67.7)
        rand_C = get_C(np.random.uniform(0, 52.9))
        rand_c = np.random.uniform(0, 4)
        rand_gamma = np.random.uniform(0, 9947)
        rand_eta = np.random.uniform(0, 289.2)
        ret.append(np.array([rand_mu, rand_theta, rand_C, rand_c, rand_gamma, rand_eta]))
    return ret


def time_decay(i, c):
    """
    Time decay part for series (tau + c)
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
    return cost/len(cost_vector)


def reg_cost_function(params, x, y, params0):
    """
    Regularized cost function for HIP model
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount
    :param y: observed viewcount
    :param params0: reference values from non-regularized model
    :return: cost function value
    """
    n = len(x)
    mu, theta, C, c, gamma, eta = params
    # handle refer parameters equal to zero
    for i in xrange(4):
        if params0[i] == 0:
            params0[i] = 1
    mu0, C0, gamma0, eta0, w = params0
    view_predict = predict(params, x)
    cost_vector = view_predict - y
    cost = np.sum(cost_vector**2) / 2
    cost += w/2*((mu/mu0)**2+(C/C0)**2+(gamma/gamma0)**2+(eta/eta0)**2)
    return cost/n


def predict(params, x):
    """
    Predict viewcount with sharecount sequence x, rewritten for auto grad tools
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount sequence from beginning
    :return: predict value
    """
    mu, theta, C, c, gamma, eta = params
    n = len(x)
    x_predict = None
    for i in xrange(n):
        if i == 0:
            x_predict = np.array([gamma + mu*x[0]])
        else:
            prev_predict = x_predict
            curr_predict = np.array([eta + mu*x[i] + C*np.sum(x_predict[:i]*(time_decay(i, c)**(-1-theta)))])
            x_predict = np.concatenate([prev_predict, curr_predict], axis=0)
    return x_predict


def test_predict(params, x, y, title, idx, init_idx=None, pred_params=None):
    """
    Test predict function
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount
    :param y: observed viewcount
    :param title: figure title, YoutubeID
    :param idx: subplot index
    :param init_idx: best initial set index
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
    ax2.text(0.03, 0.75, 'WWW\n$\mu$={0:.2e}, $\\theta$={1:.2e}\nC={2:.2e}, c={3:.2e}\n$\gamma$={4:.2e}, $\eta$={5:.2e}\nobj value={6:.2e}'
             .format(mu, theta, C, c, gamma, eta, cost_function(params, x, y)), transform=ax1.transAxes)

    x_www = predict(params, x)
    ax1.plot(np.arange(1, 121), x_www, 'b-', label='WWW popularity')
    ax1.set_title(title, fontdict={'fontsize': 15})

    if pred_params is not None:
        pred_mu, pred_theta, pred_C, pred_c, pred_gamma, pred_eta = pred_params
        ax2.text(0.55, 0.75, 'HIP\n$\mu$={0:.2e}, $\\theta$={1:.2e}\nC={2:.2e}, c={3:.2e}\n$\gamma$={4:.2e}, $\eta$={5:.2e}\nobj value={6:.2e} @init{7}'
                 .format(pred_mu, pred_theta, pred_C, pred_c, pred_gamma, pred_eta, cost_function(pred_params, x, y), init_idx), transform=ax1.transAxes)
        x_predict = predict(pred_params, x)
        ax1.plot(np.arange(1, 121), x_predict, 'g-', label='HIP popularity')


if __name__ == '__main__':
    fig = plt.figure(figsize=(14, 10))

    # == == == == == == == == Part 1: Generate test cases == == == == == == == == #
    test_cases = pickle.load(open('active_pars.p', 'rb'))
    # random select 4 videos
    test_videos = np.array(test_cases.keys())
    random_index = np.random.randint(0, len(test_videos), 4)
    test_vids = test_videos[random_index]
    # or select 4 videos manually
    # test_vids = ['0VuncLRnRlw', '4IlZLjmPA2k', 'ddUDCug_nVA', '0yq9h88X5Xg']

    # # == == == == == == == == Part 2: Test predict function == == == == == == == == #
    # for tc_idx, vid in enumerate(test_vids):
    #     test_params, dailyshare, dailyview = test_cases[vid]
    #     test_predict(test_params, dailyshare, dailyview, vid, idx)

    # == == == == == == == == Part 3: Test gradient function == == == == == == == == #
    # for tc_idx, vid in enumerate(test_vids):
    #     test_params, dailyshare, dailyview = test_cases[vid]
    #     print('err value for test case {0}: {1}'.format(tc_idx, optimize.check_grad(cost_function, grad_descent, test_params, dailyshare, dailyview)))

    # == == == == == == == == Part 4: Test cost and grad function == == == == == == == == #
    # setting parameters
    age = 120
    iteration = 200
    num_train = 75
    num_cv = 15
    num_test = 30
    eps = np.finfo(float).eps
    bounds = [(0, None), (0, 100), (0, None), (0, 5), (0, None), (0, None)]

    # define auto grad function
    auto_grad_func = grad(cost_function)
    reg_auto_grad_func = grad(reg_cost_function)

    for tc_idx, vid in enumerate(test_vids):
        test_params, dailyshare, dailyview = test_cases[vid]
        dailyshare = dailyshare[:age]
        dailyview = dailyview[:age]

        # if vid == '0VuncLRnRlw':
        #     test_params = [218.9131, 24.36634, get_C(22.27494), 0.1545296, 2869.518, 242.8026]
        # if vid == '4IlZLjmPA2k':
        #     test_params = [20.38386, 1.55089, get_C(0.5068971), 2.220446E-16, 1688.062, 44.46518]

        x_train = dailyshare[: num_train]
        y_train = dailyview[: num_train]
        x_cv = dailyshare[: num_train+num_cv]
        y_cv = dailyview[: num_train+num_cv]
        x_test = dailyshare[: num_train+num_cv+num_test]
        y_test = dailyview[: num_train+num_cv+num_test]

        # initialize weights
        # 4 sets of fixed params and k sets of random
        k = 5
        initial_theta_sets = rand_initialize_weights(k)
        initial_theta_sets.insert(0, test_params)

        best_reg_params = None
        best_reg_params0 = None
        best_cost = np.inf
        best_init_idx = None
        # find best optimizers within those k+5 sets of params
        print('Test case vid: {0}'.format(vid))
        print('\ttarget training cost: {0:>6.4e}'.format(cost_function(test_params, x_train, y_train)))
        for init_idx, initial_theta in enumerate(initial_theta_sets):
            # perform non-regularized optimization with l-bfgs
            optimizer = optimize.minimize(cost_function, initial_theta, jac=auto_grad_func, method='L-BFGS-B',
                                          args=(x_train, y_train), bounds=bounds,
                                          options={'disp': None, 'maxiter': iteration})
            print('\tinitial set{0}:  non-regularized training cost: {1:>6.4e}'.format(init_idx, optimizer.fun))

            # == == == == == == == == Part 5: Test regularized cost and grad function == == == == == == == == #
            mu0, theta0, C0, c0, gamma0, eta0 = optimizer.x
            J0 = optimizer.fun
            for w in np.arange(np.log(10**-4*J0), np.log(10*J0), 1):
                w0 = np.exp(w)
                reg_params0 = np.array([mu0, C0, gamma0, eta0, w0])
                reg_optimizer = optimize.minimize(reg_cost_function, optimizer.x, jac=reg_auto_grad_func, method='L-BFGS-B',
                                                  args=(x_train, y_train, reg_params0), bounds=bounds,
                                                  options={'disp': None, 'maxiter': iteration})
                # model selection by using cv dataset
                selection_cost = cost_function(reg_optimizer.x, x_cv, y_cv)
                if selection_cost < best_cost:
                    best_reg_params = reg_optimizer.x
                    best_reg_params0 = reg_params0
                    best_cost = selection_cost
                    best_init_idx = init_idx
            print('\tinitial set{0}: best regularized training cost: {1:>6.4e}'.format(init_idx, cost_function(best_reg_params, x_train, y_train)))

        best_reg_optimizer = optimize.minimize(reg_cost_function, best_reg_params, jac=reg_auto_grad_func, method='L-BFGS-B',
                                               args=(x_cv, y_cv, best_reg_params0), bounds=bounds,
                                               options={'disp': None, 'maxiter': iteration})

        print('+'*79)
        print('+      target test cost for {0}: {1:>6.4e}'.format(vid, cost_function(test_params, x_test, y_test)))
        print('+ regularized test cost for {0}: {1:>6.4e} @best initial set: {2}'.format(vid, cost_function(best_reg_params, x_test, y_test), best_init_idx))
        print('+'*79)
        test_predict(test_params, dailyshare, dailyview, vid, tc_idx, init_idx=best_init_idx, pred_params=best_reg_params)

    # plt.legend()
    plt.tight_layout()
    plt.show()
