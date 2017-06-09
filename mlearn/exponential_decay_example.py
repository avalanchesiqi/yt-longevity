import json
import isodate
from datetime import datetime
from scipy import optimize
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt


def read_as_int_array(content, truncated=None):
    if truncated is None:
        return np.array(map(int, content.split(',')), dtype=np.uint32)
    else:
        return np.array(map(int, content.split(',')), dtype=np.uint32)[:truncated]


def read_as_float_array(content, truncated=None):
    if truncated is None:
        return np.array(map(float, content.split(',')), dtype=np.float64)
    else:
        return np.array(map(float, content.split(',')), dtype=np.float64)[:truncated]


def predict(params, x):
    theta, C = params
    x_predict = None
    for i in x:
        if i == 0:
            x_predict = np.array([C])
        else:
            curr_predict = np.array([C*(i**(-theta))])
            x_predict = np.concatenate([x_predict, curr_predict], axis=0)
    return x_predict


def cost_function(params, x, y):
    wp_predict = predict(params, x)
    cost_vector = wp_predict - y
    cost = np.sum(cost_vector ** 2) / 2
    return cost/len(cost_vector)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    fig = plt.figure(figsize=(10, 10))
    cnt = 0
    autograd_func = grad(cost_function)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    category_id = '29'
    data_loc = '../../data/production_data/random_dataset/{0}.json'.format(category_id)

    with open(data_loc) as fin:
        for line in fin:
            video = json.loads(line.rstrip())
            duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            published_at = video['snippet']['publishedAt'][:10]
            start_date = video['insights']['startDate']
            time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days

            days = read_as_int_array(video['insights']['days']) + time_diff
            if len(days) < 90:
                continue
            daily_view = read_as_int_array(video['insights']['dailyView'], truncated=len(days))

            # pre filter
            if len(days[daily_view < 100]) > 1/4*len(days):
                continue

            daily_watch = read_as_float_array(video['insights']['dailyWatch'], truncated=len(days))

            daily_wp = np.divide(daily_watch * 60, daily_view * duration, where=(daily_view != 0))
            daily_wp[daily_wp > 1] = 1

            optimizer = optimize.minimize(cost_function, np.array([1, 0.5]), jac=autograd_func, method='L-BFGS-B',
                                          args=(days, daily_wp), bounds=[(None, None), (0, 1)],
                                          options={'disp': None})
            print optimizer.fun
            print 'initial wp: {1}, theta: {0}'.format(*optimizer.x)

            ax1 = fig.add_subplot(321+cnt)
            ax2 = ax1.twinx()
            ax1.plot(days, daily_view, 'r-', label='observed #views')
            ax2.plot(days, daily_wp, 'o-', c='b', ms=2, mfc='None', mec='b', mew=1, label='daily watch percentage')
            ax2.plot(days, predict(optimizer.x, days), 'o-', c='g', ms=2, mfc='None', mec='g', mew=1, label='fitted watch percentage')

            ax1.set_xlim(xmax=365)
            ax1.set_ylim(ymin=max(0, ax1.get_ylim()[0]))
            ax2.set_xlim(xmax=365)
            ax2.set_ylim(ymin=0)
            ax2.set_ylim(ymax=1)
            ax1.set_xlabel('video age (day)')
            ax1.set_ylabel('Number of views', color='k')
            ax1.tick_params('y', colors='k')
            ax2.set_ylabel('Portion of watch', color='r')
            ax2.tick_params('y', colors='r')
            cnt += 1

            if cnt > 5:
                break

    plt.tight_layout()
    plt.show()
