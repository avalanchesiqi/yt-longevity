import os
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def thousands(x, pos):
    return '%.1fK' % (x*1e-3)


def plot_scatter(arr, ax):
    """ Scatter plot of track value in rate limit messages. """
    xaxis = np.arange(1, len(arr)+1)
    ax.scatter(xaxis, arr, marker='x', s=1, c='b')
    ax1.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_title('Scatter plot of rate limit message value wrt incoming order')
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    ax.set_xlabel('Incoming order')
    ax.set_ylabel('Rate limit message value')


def get_non_disconnect_window(arr, width):
    """ Get non-disconnect windows from rate limit messages. """
    box = deque(maxlen=width)
    windows = []
    for val in arr:
        if len(box) == 0:
            windows.append([val])
        elif len(box) < width and (min(box)-val) > 2500 and val < 300:
            # detect drop 1, not full, decrease sharply, hard threshold
            print 'detect drop #1: {0} {1} {2} {3}'.format(val, len(box), min(box), box)
            # reset box
            box = deque(maxlen=width)
            windows.append([val])
        elif len(box) == width and val < min(box) and val < 300:
            # detect drop 2, full, smaller than min in box, hard threshold
            print 'detect drop #2: {0} {1} {2} {3}'.format(val, len(box), min(box), box)
            # reset box
            box = deque(maxlen=width)
            windows.append([val])
        else:
            windows[-1].append(val)
        box.append(val)

    n1 = len(windows)
    # filter out connection with less than 100 points
    # windows = [w for w in windows if len(w) > 100]
    n2 = len(windows)
    print 'number of connections after filter:', n2, 'short segment:', n1-n2
    return windows


def plot_width_distribution(bound_orders, bound_lines, ax):
    lower_bound_order = bound_orders[0]
    upper_bound_order = bound_orders[1]
    lower_bound_line = bound_lines[0]
    upper_bound_line = bound_lines[1]
    project_gaps = []
    n = len(upper_bound_order)
    for i in xrange(1, n):
        order_incr = upper_bound_order[i] - upper_bound_order[i-1]
        track_incr = upper_bound_line[i] - upper_bound_line[i-1]
        tan_theta = 1.0*order_incr/track_incr
        vertical_gap = upper_bound_line[i] - lower_bound_line[i]
        # sin(theta) = tan(theta)/sqrt(tan(theta)**2 +1)
        project_gap = vertical_gap*tan_theta/math.sqrt(tan_theta**2 + 1)
        project_gaps.append(project_gap)

    ax.plot(lower_bound_order[1:], project_gaps)
    ax.set_title('Scatter plot of rate limit message value wrt incoming order')
    ax.set_xlabel('Incoming order')
    ax.set_ylabel('Projected width')


def get_and_plot_simplified_lines(orders_arr, tracks_arr, ax1, ax2):
    # get base (lowest) line
    t = len(orders_arr)

    baseline_order = orders_arr[-1]
    baseline = tracks_arr[-1]
    num_of_points = len(baseline)
    ratio = 0.05
    num_of_simplified_points = int(math.floor(num_of_points * ratio))
    simplified_base_order = []
    simplified_base_line = []
    for i in xrange(num_of_simplified_points):
        simplified_base_order.append(baseline_order[int(i*1/ratio)])
        simplified_base_line.append(baseline[int(i*1/ratio)])

    # bound lines, the highest one and the lowest one
    bound_orders = [simplified_base_order]
    bound_lines = [simplified_base_line]

    # get closest 3 top lines
    for i in xrange(t - 1):
        current_order = orders_arr[i]
        current_line = tracks_arr[i]
        n = len(current_order)
        simplified_current_order = []
        simplified_current_line = []
        s = 0
        for j in xrange(n - 1):
            if s == len(simplified_base_order):
                break
            # find two points enclose point in simplified baseline order
            if current_order[j] < simplified_base_order[s] < current_order[j + 1]:
                if current_order[j] - simplified_base_order[s] < simplified_base_order[s] - current_order[j + 1]:
                    simplified_current_order.append(current_order[j])
                    simplified_current_line.append(current_line[j])
                else:
                    simplified_current_order.append(current_order[j + 1])
                    simplified_current_line.append(current_line[j + 1])
                s += 1
        ax1.plot(simplified_current_order, simplified_current_line, '-', ms=2, marker='x')
        if i == 0:
            bound_orders.append(simplified_current_order)
            bound_lines.append(simplified_current_line)

    ax1.plot(simplified_base_order, simplified_base_line, '-', ms=2, marker='x')
    ax1.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax1.set_title('Simulation result of simplified lines')
    ax1.set_xlabel('Incoming order')

    plot_width_distribution(bound_orders, bound_lines, ax2)
    return


def fit_and_plot_multiple_lines(arr, ax1, ax2, ax3, head=0, prev=0):
    # initialize order array and value array
    orders_arr = [[prev+1]]
    tracks_arr = [[arr[0]]]

    n = len(arr)
    order = prev+2
    for i in xrange(1, n):
        new_comer = arr[i]
        m = len(tracks_arr)
        last_elms = np.array([tracks_arr[j][-1] for j in xrange(m)])
        if new_comer <= min(last_elms):
            # start a new line only if this is the head window or absolute value less than 100
            if (head == 0 and order < 50) or new_comer < 100:
                orders_arr.append([order])
                tracks_arr.append([new_comer])
        else:
            # assign to the least incremental line
            last_elms[new_comer <= last_elms] = 0
            k = np.argmin(new_comer-last_elms)
            orders_arr[k].append(order)
            tracks_arr[k].append(new_comer)
        order += 1

    # filter out segment with less than 25 points
    # orders_arr = [orders for orders in orders_arr if len(orders) > 25]
    # tracks_arr = [tracks for tracks in tracks_arr if len(tracks) > 25]

    t = len(tracks_arr)
    print 'number of inner segments:', t

    if head == 0:
        window_miss = sum([(tracks_arr[i][-1] - tracks_arr[i][0]) for i in xrange(t)])
    else:
        window_miss = sum([(tracks_arr[i][-1]) for i in xrange(t)])

    # plot simulation result of fitting multiple lines
    for order, seg in zip(orders_arr, tracks_arr):
        ax1.plot(order, seg, '-', ms=2, marker='x')
    ax1.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax1.set_title('Simulation result of fitting multiple lines')
    ax1.set_xlabel('Incoming order')

    # get and plot a simplified line by using 1% points
    # get_and_plot_simplified_lines(orders_arr, tracks_arr, ax2, ax3)

    return orders_arr, tracks_arr, window_miss


if __name__ == '__main__':
    file_loc = '../../data/rate_limit'
    filename = 'rate_2015-02-27.bz2.txt'
    # 2015-11-17, noise input
    date = filename[5:15]
    filepath = os.path.join(file_loc, filename)

    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1), sharex=ax1, sharey=ax1)
    ax3 = plt.subplot2grid((2, 3), (0, 2), sharex=ax1, sharey=ax1)
    ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3, sharex=ax1)

    # read rate limit track value from file
    values = []
    with open(filepath, 'r') as filedata:
        for line in filedata:
            values.append(int(line.rstrip()))

    # plot scatter of rate limit track value wrt incoming order
    plot_scatter(values, ax1)

    # get non-disconnect windows from values
    width = 20
    non_disconnect_windows = get_non_disconnect_window(values, width)

    # fit multiple lines to each non-disconnect window and get miss tweets number
    prev_msgs = 0
    total_miss = 0
    for (cnt, window) in enumerate(non_disconnect_windows):
        # fit multiple lines and plot lines, simplified lines, width distribution
        orders_arr, tracks_arr, window_miss = fit_and_plot_multiple_lines(window, ax2, ax3, ax4, head=cnt, prev=prev_msgs)
        # number of messages in previous windows
        prev_msgs += len(window)
        total_miss += window_miss
    print 'On date {0} {1} tweets have been missed due to sampling policy on Twitter'.format(date, total_miss)

    plt.show()
