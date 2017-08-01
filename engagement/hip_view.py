from __future__ import print_function, division
import sys
import os
import bz2
import json
import isodate
import cPickle as pickle
import autograd.numpy as np
from autograd import grad
# import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Hawkes Intensity Process model in Python


def read_as_float_array(content, truncated=None, delimiter=None):
    """
    Read input as a float array.
    :param content: string input
    :param truncated: head number of elements extracted
    :param delimiter: delimiter string
    :return: a numpy float array
    """
    if truncated is None:
        return np.array(map(float, content.split(delimiter)), dtype=np.float64)
    else:
        return np.array(map(float, content.split(delimiter)[:truncated]), dtype=np.float64)


def get_C(k, alpha=2.016, beta=0.1):
    """
    Get parameter capital C.
    :param k: scaling factor for video quality
    :param alpha: power-law exponent of user influence distribution
    :param beta: user influence component
    :return: parameter capital C
    """
    return k*(alpha-1)/(alpha-beta-1)


def rand_initialize_weights(n):
    """
    Initialize multiple sets of random weights for theta.
    :param n: number of sets of random weights
    :return: n sets of random vectors, in the order of mu, theta, C, c, gamma, eta
    """
    ret = []
    for _ in xrange(n):
        rand_mu = np.random.uniform(0, 505.90)
        rand_theta = np.random.uniform(2.3, 7.7)
        rand_C = get_C(np.random.uniform(0, 52.9))
        rand_c = np.random.uniform(0, 4)
        rand_gamma = np.random.uniform(0, 9947)
        rand_eta = np.random.uniform(0, 289.2)
        ret.append(np.array([rand_mu, rand_theta, rand_C, rand_c, rand_gamma, rand_eta]))
    return ret


def time_decay(i, c):
    """
    Time decay part for series (tau + c).
    :param i: tau value
    :param c: c value
    :return: abbreviated presentation
    """
    return np.arange(1, i+1)[::-1]+c


def predict(params, x):
    """
    Predict viewcount with sharecount sequence x.
    Comments are for vector operation style
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount sequence from beginning
    :return: predict value
    """
    mu, theta, C, c, gamma, eta = params
    n = len(x)
    x_predict = None
    # x_predict = np.zeros(len(x))
    for i in xrange(n):
        if i == 0:
            x_predict = np.array([gamma + mu * x[0]])
            # x_predict[0] = gamma + mu*x[0]
        else:
            prev_predict = x_predict
            curr_predict = np.array([eta + mu * x[i] + C * np.sum(x_predict[:i] * (time_decay(i, c) ** (-1 - theta)))])
            x_predict = np.concatenate([prev_predict, curr_predict], axis=0)
            # x_predict[i] = eta + mu*x[i] + C*np.sum(x_predict[:i]*(time_decay(i, c)**(-1-theta)))
    return x_predict


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
        grad_mu_vector[i] = x[i] + C*np.sum(grad_mu_vector[:i] * (time_decay(i, c)**(-1-theta)))
    grad_mu = np.sum((view_predict-y)*grad_mu_vector)
    # partial derivative for theta
    grad_theta_vector = np.zeros(n)
    grad_theta_vector[0] = 0
    for i in xrange(1, n):
        grad_theta_vector[i] = C*np.sum((grad_theta_vector[:i]-view_predict[:i]*np.log(time_decay(i, c))) * (time_decay(i, c)**(-1-theta)))
    grad_theta = np.sum((view_predict-y)*grad_theta_vector)
    # partial derivative for C
    grad_C_vector = np.zeros(n)
    grad_C_vector[0] = 0
    for i in xrange(1, n):
        grad_C_vector[i] = np.sum((C*grad_C_vector[:i]+view_predict[:i]) * (time_decay(i, c)**(-1-theta)))
    grad_C = np.sum((view_predict-y)*grad_C_vector)
    # partial derivative for c
    grad_c_vector = np.zeros(n)
    grad_c_vector[0] = 0
    for i in xrange(1, n):
        grad_c_vector[i] = C*np.sum((grad_c_vector[:i]-(1+theta)*view_predict[:i]/time_decay(i, c)) * (time_decay(i, c)**(-1-theta)))
    grad_c = np.sum((view_predict-y)*grad_c_vector)
    # partial derivative for gamma
    grad_gamma_vector = np.zeros(n)
    grad_gamma_vector[0] = 1
    for i in xrange(1, n):
        grad_gamma_vector[i] = C*np.sum(grad_gamma_vector[:i] * (time_decay(i, c)**(-1-theta)))
    grad_gamma = np.sum((view_predict-y)*grad_gamma_vector)
    # partial derivative for eta
    grad_eta_vector = np.zeros(n)
    grad_eta_vector[0] = 0
    for i in xrange(1, n):
        grad_eta_vector[i] = 1 + C*np.sum(grad_eta_vector[:i] * (time_decay(i, c)**(-1-theta)))
    grad_eta = np.sum((view_predict-y)*grad_eta_vector)
    return np.array([grad_mu, grad_theta, grad_C, grad_c, grad_gamma, grad_eta])/n


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
    # handle refer parameters equal to zero
    for i in xrange(4):
        if params0[i] == 0:
            params0[i] = 1
    mu0, C0, gamma0, eta0, w = params0
    view_predict = predict(params, x)
    n = len(x)
    # partial derivative for mu
    grad_mu_vector = np.zeros(n)
    grad_mu_vector[0] = x[0]
    for i in xrange(1, n):
        grad_mu_vector[i] = x[i] + C*np.sum(grad_mu_vector[:i] * (time_decay(i, c)**(-1-theta)))
    grad_mu = (np.sum((view_predict-y)*grad_mu_vector) + w*mu/mu0/mu0)
    # partial derivative for theta
    grad_theta_vector = np.zeros(n)
    grad_theta_vector[0] = 0
    for i in xrange(1, n):
        grad_theta_vector[i] = C*np.sum((grad_theta_vector[:i]-view_predict[:i]*np.log(time_decay(i, c))) * (time_decay(i, c)**(-1-theta)))
    grad_theta = np.sum((view_predict-y)*grad_theta_vector)
    # partial derivative for C
    grad_C_vector = np.zeros(n)
    grad_C_vector[0] = 0
    for i in xrange(1, n):
        grad_C_vector[i] = np.sum((C*grad_C_vector[:i]+view_predict[:i]) * (time_decay(i, c)**(-1-theta)))
    grad_C = (np.sum((view_predict-y)*grad_C_vector) + w*C/C0/C0)
    # partial derivative for c
    grad_c_vector = np.zeros(n)
    grad_c_vector[0] = 0
    for i in xrange(1, n):
        grad_c_vector[i] = C*np.sum((grad_c_vector[:i]-(1+theta)*view_predict[:i]/time_decay(i, c)) * (time_decay(i, c)**(-1-theta)))
    grad_c = np.sum((view_predict-y)*grad_c_vector)
    # partial derivative for gamma
    grad_gamma_vector = np.zeros(n)
    grad_gamma_vector[0] = 1
    for i in xrange(1, n):
        grad_gamma_vector[i] = C*np.sum(grad_gamma_vector[:i] * (time_decay(i, c)**(-1-theta)))
    grad_gamma = np.sum((view_predict-y)*grad_gamma_vector) + w*gamma/gamma0/gamma0
    # partial derivative for eta
    grad_eta_vector = np.zeros(n)
    grad_eta_vector[0] = 0
    for i in xrange(1, n):
        grad_eta_vector[i] = 1 + C*np.sum(grad_eta_vector[:i] * (time_decay(i, c)**(-1-theta)))
    grad_eta = np.sum((view_predict-y)*grad_eta_vector) + w*eta/eta0/eta0
    return np.array([grad_mu, grad_theta, grad_C, grad_c, grad_gamma, grad_eta])/n


def train_process(x, y, initial_weights_sets):
    """
    Train HIP with BFGS optimization tool
    :param x: observed sharecount
    :param y: observed viewcount
    :param initial_weights_sets: sets of random initial weights
    :return: best optimization parameters
    """
    best_cost = np.inf
    whether_reg = False
    autograd = False

    if autograd:
        grad_func = autograd_func
        reg_grad_func = reg_autograd_func
    else:
        grad_func = grad_descent
        reg_grad_func = reg_grad_descent

    if whether_reg:
        best_reg_params = None
        best_reg_params0 = None
        x_train = x[:-num_cv]
        y_train = y[:-num_cv]
    else:
        best_params = None
        x_train = x
        y_train = y

    for init_idx, initial_weight in enumerate(initial_weights_sets):
        # perform non-regularized optimization with l-bfgs
        optimizer = optimize.minimize(cost_function, initial_weight, jac=grad_func, method='L-BFGS-B',
                                      args=(x_train, y_train), bounds=bounds)

        if whether_reg:
            mu0, theta0, C0, c0, gamma0, eta0 = optimizer.x
            J0 = optimizer.fun
            # line search in logspace (10e-4*J0, 10*J0)
            for w in np.arange(np.log(10 ** -4 * J0), np.log(10 * J0), 1):
                w0 = np.exp(w)
                reg_params0 = np.array([mu0, C0, gamma0, eta0, w0])
                reg_optimizer = optimize.minimize(reg_cost_function, optimizer.x, jac=reg_grad_func,
                                                  method='L-BFGS-B', args=(x_train, y_train, reg_params0),
                                                  bounds=bounds, options={'disp': None, 'maxiter': iteration})
                # model selection by using cv dataset
                cv_cost = cost_function(reg_optimizer.x, x, y, num_split=num_cv)
                if cv_cost < best_cost:
                    best_reg_params = reg_optimizer.x
                    best_reg_params0 = reg_params0
                    best_cost = cv_cost
        else:
            cv_cost = cost_function(optimizer.x, x, y)
            if cv_cost < best_cost:
                best_cost = cv_cost
                best_params = optimizer.x

    if whether_reg:
        best_reg_optimizer = optimize.minimize(reg_cost_function, best_reg_params, jac=reg_grad_func,
                                               method='L-BFGS-B', args=(x, y, best_reg_params0),
                                               bounds=bounds, options={'disp': None, 'maxiter': iteration})
        return best_reg_optimizer.x
    else:
        best_optimizer = optimize.minimize(cost_function, best_params, jac=grad_func,
                                           method='L-BFGS-B', args=(x, y),
                                           bounds=bounds, options={'disp': None, 'maxiter': iteration})
        return best_optimizer.x


def plot_func(params, x, y, title, idx):
    """
    Plot trend from R-HIP, PY-HIP and AUTO-HIP parameters
    :param params: model parameters, mu, theta, C, c, gamma, eta
    :param x: observed sharecount
    :param y: observed viewcount
    :param title: figure title, YoutubeID
    :param idx: subplot index
    :return:
    """
    # visualise sample data
    ax1 = fig.add_subplot(121+idx)
    # ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(np.arange(1, age+1), y, 'k--', label='observed watch time')
    ax2.plot(np.arange(1, age+1), x, 'r-', label='#share')
    ax1.plot((num_train, num_train), (ax1.get_ylim()[0], ax1.get_ylim()[1]), 'k--')

    ax1.set_ylim(ymin=max(0, ax1.get_ylim()[0]))
    ax2.set_ylim(ymax=3*max(x))
    ax1.set_xlabel('video age (day)')
    ax1.set_ylabel('Time of watches', color='k')
    ax1.tick_params('y', colors='k')
    ax2.set_ylabel('Number of shares', color='r')
    ax2.tick_params('y', colors='r')

    mu, theta, C, c, gamma, eta = params
    ax2.text(0.03, 0.85, '$\mu$={0:.2f}, $\\theta$={1:.2f}\nC={2:.2f}, c={3:.2f}\n$\gamma$={4:.2f}, $\eta$={5:.2f}'
             .format(mu, theta, C, c, gamma, eta), transform=ax1.transAxes)
    ax1.set_title(title)

    predidt_x = predict(params, x)
    ax1.plot(np.arange(1, num_train+1), predidt_x[:num_train], 'b-', label='HIP fit')
    ax1.plot(np.arange(num_train+1, age+1), predidt_x[num_train:age], 'm-', label='HIP forecast')
    if vid in mlr_predict_dict:
        ax1.plot(np.arange(90+1, age+1), mlr_predict_dict[vid], 'g-', label='MLR forecast')


def strify(iterable_struct):
    """
    Convert an iterable structure to comma separated string
    :param iterable_struct: an iterable structure
    :return: a string with comma separated
    """
    return ','.join(map(str, iterable_struct))


if __name__ == '__main__':
    # == == == == == == == == Part 1: Load ACTIVE dataset == == == == == == == == #
    # First time it gets loaded from the JSON format and writes essential fields into a pickle binary file.
    # check if the binary exists
    if not os.path.exists('./active-dataset.p'):
        print('>>> Converting ACTIVE dataset from JSON format to pickle... might take a while!')
        test_cases = {}
        with bz2.BZ2File('./active-dataset.json.bz2') as f:
            dataset = json.loads(f.readline())
            for video in dataset:
                if not video['duration'] == 'NA':
                    test_cases[video['YoutubeID']] = (video['numShare'], video['dailyViewcount'], video['watchTime'], video['duration'])
        pickle.dump(test_cases, open('./active-dataset.p', 'wb'))

    print('>>> Loading the ACTIVE dataset from pickle...')
    test_cases = pickle.load(open('./active-dataset.p', 'rb'))

    # Loading MLR result
    mlr_predict_dict = {}
    with open('mlr_daily_forecast_watch.log', 'r') as f:
        for line in f:
            vid, observed, daily_predict = line.rstrip().split(',', 2)
            daily_predict = read_as_float_array(daily_predict, delimiter=',')
            mlr_predict_dict[vid] = daily_predict

    # select 2 videos from paper
    test_vids = ['X1kGausCXM4', '6hFjCvNRNlw']
    # or random select 2 videos
    # test_videos = np.array(test_cases.keys())
    # random_index = np.random.randint(0, len(test_videos), 2)
    # test_vids = test_videos[random_index]
    # test_vids = test_cases.keys()

    # == == == == == == == == Part 2: Set up experiment parameters == == == == == == == == #
    # setting parameters
    fig = plt.figure(figsize=(14, 5))
    age = 120
    num_train = 30
    num_cv = 15
    num_test = age - num_train
    k = 5
    bounds = [(0, None), (0, 10), (0, None), (0, 5), (0, None), (0, None)]
    iteration = 100

    # define auto grad function
    autograd_func = grad(cost_function)
    reg_autograd_func = grad(reg_cost_function)

    true_predict_tuple = []

    # fout = open('training_watch_reg_tune.log', 'w')
    for tc_idx, vid in enumerate(test_vids):
        print('fitting and forecasting for video: {0}'.format(vid))
        dailyshare, dailyview, watchtime, duration_txt = test_cases[vid]
        duration = isodate.parse_duration(duration_txt).seconds

        # first 120 days
        dailyshare = dailyshare[:age]
        dailyview = dailyview[:age]
        watchtime = watchtime[:age]

        x_whole = dailyshare
        # select view count or watch time as dependent variable
        y_whole = watchtime

        # first 90 days
        x_train = x_whole[: num_train]
        y_train = y_whole[: num_train]

        # initialize weights
        # k sets of random params
        initial_weights_sets = rand_initialize_weights(k)

        # == == == == == == == == Part 3: Train with closed form gradient == == == == == == == == #
        best_fitted_params = train_process(x_train, y_train, initial_weights_sets)

        # == == == == == == == == Part 4: Forecast for the next 30 days == == == == == == == == #
        y_predict = predict(best_fitted_params, x_whole)
        ground_truth = np.sum(y_whole)
        predict_sum = np.sum(y_whole[:-num_test]) + np.sum(y_predict[-num_test:])
        if vid in mlr_predict_dict:
            mlr_predict = np.sum(y_whole[:-num_test]) + np.sum(mlr_predict_dict[vid])
        else:
            mlr_predict = 'NA'
        print('True: {0}; Predict: {1}; MLR: {2}'.format(ground_truth, predict_sum, mlr_predict))

        # true_predict_tuple.append((ground_truth, predict_sum))

        # fout.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n'.format(vid, best_fitted_params[0],
        #                                                                    best_fitted_params[1], best_fitted_params[2],
        #                                                                    best_fitted_params[3], best_fitted_params[4],
        #                                                                    best_fitted_params[5], strify(x_whole),
        #                                                                    strify(y_whole), ground_truth, predict_sum))

        # # == == == == == == == == Part 4: Plot fitting and forecast result == == == == == == == == #
        plot_func(best_fitted_params, x_whole, y_whole, vid+' - '+duration_txt, tc_idx)

    print('Plot done')
    plt.show()
    # fout.close()
