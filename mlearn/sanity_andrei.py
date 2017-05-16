import sys
import numpy as np
from scipy import optimize
import cPickle as pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('error')

# Hawkes Intensity Process model


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
            try:
                x_predict[i] = eta + mu*x[i] + C*np.sum(x_predict[:i]*(np.abs(np.arange(1, i+1)[::-1]+c)**(-1-theta)))
            except Warning:
                print 'warning at {0}'.format(i)
                print mu, theta, C, c, gamma, eta
                print x_predict[:i]
                print 'base', np.abs(np.arange(1, i+1)[::-1]+c)
                print 'c:', c, '\ttheta:', theta
                print (np.arange(1, i+1)[::-1]+c)**(-1-theta)
                print sys.exit(1)
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
    test_cases = pickle.load(open('active_pars.p', 'rb'))
    # random select 4 videos
    test_videos = np.array(test_cases.keys())
    random_index = np.random.randint(0, len(test_videos), 4)
    test_vids = test_videos[random_index]
    # or select 4 videos manually
    # test_vids = ['GblgGI26XpY', '_CTpnvgatHQ', '_lMwQkgzDuk', 'aOaEnZdXByk']
    test_vids = ['X7hjr2gnS4A', '0VuncLRnRlw', '7EVdFDVafYA', '4IlZLjmPA2k']

    age = 120
    # # == == == == == == == == Part 2: Test predict function == == == == == == == == #
    for tc_idx, vid in enumerate(test_vids):
        test_params, dailyshare, dailyview = test_cases[vid]
        dailyshare = dailyshare[:age]
        dailyview = dailyview[:age]
        test_predict(test_params, dailyshare, dailyview, vid, tc_idx)

    plt.tight_layout()
    plt.show()
