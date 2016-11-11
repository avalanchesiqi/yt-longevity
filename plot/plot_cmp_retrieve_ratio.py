import os
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import cPickle as pickle


def get_retrieve_num(path):
    retrieve_dict = {}
    with open(path, 'r') as data:
        filename = data.readline()
        while filename:
            date = filename.rstrip().rsplit(':', 1)[1][-14:-4]
            retrieve_num = int(data.readline().rstrip().rsplit(':', 1)[1])
            data.readline()
            data.readline()
            data.readline()
            data.readline()
            retrieve_dict[date] = retrieve_num
            filename = data.readline()
    return retrieve_dict


def fit_multiple_lines(lst, head=0, prev=0):
    lines = [[lst[0]]]
    orders = [[1+prev]]
    n = len(lst)
    cnt = 2+prev
    for i in xrange(1, n):
        comer = lst[i]
        m = len(lines)
        last_elms = np.array([lines[j][-1] for j in xrange(m)])
        if comer <= min(last_elms):
            if (head == 0 and cnt < 20) or comer < 100:
                lines.append([comer])
                orders.append([cnt])
        else:
            last_elms[comer <= last_elms] = 0
            k = np.argmin(comer-last_elms)
            lines[k].append(comer)
            orders[k].append(cnt)
        cnt += 1

    # filter out segment with less than 25 points
    orders = [order for order in orders if len(order) > 25]
    lines = [line for line in lines if len(line) > 25]

    t = len(lines)
    print 'number of inner segments:', t, len(orders)
    if head == 0:
        miss_num = sum([(lines[i][-1] - lines[i][0]) for i in xrange(t)])
    else:
        miss_num = sum([(lines[i][-1]) for i in xrange(t)])
    return orders, lines, miss_num


def get_miss_num_max(lst, head=0):
    if head == 0:
        miss_num = max(lst) - min(lst)
    else:
        miss_num = max(lst)
    return miss_num


def get_miss_num_controller(arrs):
    prev = 0
    total_miss_sum = 0
    total_miss_max = 0
    for (cnt, arr) in enumerate(arrs):
        total_miss_max += get_miss_num_max(arr, head=cnt)

        orders, _, segs_miss_num = fit_multiple_lines(arr, head=cnt, prev=prev)
        prev += sum([len(order) for order in orders])
        total_miss_sum += segs_miss_num
    return total_miss_max, total_miss_sum


def get_disconnect_window(arr, width):
    box = deque(maxlen=width)
    windows = []
    for val in arr:
        if len(box) == 0:
            windows.append([val])
        elif len(box) < width and (min(box)-val) > 2500 and val < 100:
            # detect drop 1, not full, decrease sharply, hard threshold
            # print 'detect drop #1: {0} {1} {2} {3}'.format(val, len(box), min(box), box)
            # reset box
            box = deque(maxlen=width)
            windows.append([val])
        elif len(box) == width and min(box) > val and val < 100:
            # detect drop 2, full, smaller than min in box, hard threshold
            # print 'detect drop #2: {0} {1} {2} {3}'.format(val, len(box), min(box), box)
            # reset box
            box = deque(maxlen=width)
            windows.append([val])
        else:
            windows[-1].append(val)
        box.append(val)

    windows = [w for w in windows if len(w) > 100]
    n = len(windows)
    print 'number of connections:', n
    return windows


def get_miss_num(loc):
    miss_max_dict = {}
    miss_sum_dict = {}

    for subdir, _, files in os.walk(loc):
        for f in sorted(files):
            path = os.path.join(subdir, f)
            tracks = []
            with open(path, 'r') as data:
                for line in data:
                    track = int(line.rstrip())
                    tracks.append(track)

            date = path.rstrip()[-18:-8]
            print '-----------------'
            print date
            width = 20
            non_disconnect_windows = get_disconnect_window(tracks, width)
            total_miss_max, total_miss_sum = get_miss_num_controller(non_disconnect_windows)

            miss_max_dict[date] = total_miss_max
            miss_sum_dict[date] = total_miss_sum
    return miss_max_dict, miss_sum_dict


def date2str(date_format):
    return date_format.strftime("%Y-%m-%d")


def millions(x, pos):
    return '%1.1fM' % (x*1e-6)


def get_stats(arr):
    arr = np.array(arr)
    return np.mean(arr), np.std(arr), np.min(arr), np.percentile(arr, 25), np.median(arr), np.percentile(arr, 75), np.max(arr)


def plot_stats_controller(ax):
    after = [datetime(2015, 2, 7)+timedelta(days=i) for i in xrange(150)]
    before = [datetime(2015, 2, 5)+timedelta(days=-i) for i in xrange(150)]
    retrieve_before = [retrieve_map[date2str(d)] for d in before if date2str(d) in retrieve_map]
    max_before = [retrieve_map[date2str(d)] + miss_max_map[date2str(d)] for d in before if date2str(d) in retrieve_map and date2str(d) in miss_max_map]
    sum_before = [retrieve_map[date2str(d)] + miss_sum_map[date2str(d)] for d in before if date2str(d) in retrieve_map and date2str(d) in miss_sum_map]
    retrieve_after = [retrieve_map[date2str(d)] for d in after if date2str(d) in retrieve_map]
    max_after = [retrieve_map[date2str(d)] + miss_max_map[date2str(d)] for d in after if date2str(d) in retrieve_map and date2str(d) in miss_max_map]
    sum_after = [retrieve_map[date2str(d)] + miss_sum_map[date2str(d)] for d in after if date2str(d) in retrieve_map and date2str(d) in miss_sum_map]
    print get_stats(retrieve_before)
    print get_stats(max_before)
    print get_stats(sum_before)
    print get_stats(retrieve_after)
    print get_stats(max_after)
    print get_stats(sum_after)


if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(2, 1)

    file_loc1 = '../../data/test_sampling'
    file_loc2 = '../../data/rate_limit'
    filepath1 = os.path.join(file_loc1, 'sample_ratio.log')

    retrieve_map = get_retrieve_num(filepath1)
    miss_max_map, miss_sum_map = get_miss_num(file_loc2)

    with open(os.path.join(file_loc1, r'retrieve_map.pickle'), 'wb') as output1:
        pickle.dump(retrieve_map, output1)
    with open(os.path.join(file_loc1, r'miss_max_map.pickle'), 'wb') as output2:
        pickle.dump(miss_max_map, output2)
    with open(os.path.join(file_loc1, r'miss_sum_map.pickle'), 'wb') as output3:
        pickle.dump(miss_sum_map, output3)

    with open(os.path.join(file_loc1, r'retrieve_map.pickle'), 'rb') as input1:
        retrieve_map = pickle.load(input1)
    with open(os.path.join(file_loc1, r'miss_max_map.pickle'), 'rb') as input2:
        miss_max_map = pickle.load(input2)
    with open(os.path.join(file_loc1, r'miss_sum_map.pickle'), 'rb') as input3:
        miss_sum_map = pickle.load(input3)

    # del retrieve_map['2014-05-28']
    del retrieve_map['2014-12-26']
    date_axis = [datetime(2014, 6, 1)+timedelta(days=i) for i in xrange(len(retrieve_map)+115)]
    retrieve_axis = [retrieve_map[date2str(d)] if date2str(d) in retrieve_map else np.nan for d in date_axis]
    restore_axis1 = [retrieve_map[date2str(d)]+miss_max_map[date2str(d)] if date2str(d) in retrieve_map and date2str(d) in miss_max_map else np.nan for d in date_axis]
    restore_axis2 = [retrieve_map[date2str(d)]+miss_sum_map[date2str(d)] if date2str(d) in retrieve_map and date2str(d) in miss_sum_map else np.nan for d in date_axis]
    ax1.plot_date(date_axis, restore_axis1, '-', ms=2, color='b', label='assume max difference')
    ax1.plot_date(date_axis, restore_axis2, '-', ms=2, color='r', label='assume multiple lines sum up')
    ax1.plot_date(date_axis, retrieve_axis, '-', ms=2, color='g', label='ground-truth retrieve number')
    ax1.bar(date_axis, np.isnan(retrieve_axis)*ax1.get_ylim()[1], color=(.9, .9, .9))
    ax1.yaxis.set_major_formatter(FuncFormatter(millions))
    ax1.plot_date((datetime(2015,2,6), datetime(2015,2,6)), (0, ax1.get_ylim()[1]), 'm-')

    # plot mean, std, five points summary
    plot_stats_controller(ax1)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Retrieve tweets number')
    ax1.legend(loc='lower left')

    sampling_axis1 = [100.0*retrieve_map[date2str(d)]/(retrieve_map[date2str(d)]+miss_max_map[date2str(d)]) if date2str(d) in retrieve_map and date2str(d) in miss_max_map else np.nan for d in date_axis]
    sampling_axis2 = [100.0*retrieve_map[date2str(d)]/(retrieve_map[date2str(d)]+miss_sum_map[date2str(d)]) if date2str(d) in retrieve_map and date2str(d) in miss_max_map else np.nan for d in date_axis]
    ax2.plot_date(date_axis, sampling_axis1, '-', ms=2, color='b', label='assume max difference')
    ax2.plot_date(date_axis, sampling_axis2, '-', ms=2, color='r', label='assume multiple lines sum up')
    ax2.bar(date_axis, np.isnan(retrieve_axis) * ax2.get_ylim()[1], color=(.9, .9, .9))
    ax2.plot_date((datetime(2015, 2, 6), datetime(2015, 2, 6)), (0, ax2.get_ylim()[1]), 'm-')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Retrieve ratio')
    ax2.legend(loc='lower left')

    # plt.tight_layout()
    plt.show()

