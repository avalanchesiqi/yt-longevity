import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import cPickle as pickle
import logging

file_loc = '../../data/test_sampling'


def lst2num(lst):
    return map(int, lst.split(', '))


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


def get_miss_num_max(path):
    miss_dict = {}
    with open(path, 'r') as data:
        for line in data:
            _, _, date, segments = line.rstrip().split(':')
            print '------------------------'
            print date
            segments = segments.split('-')[:-1]
            n = len(segments)
            miss_num = max(lst2num(segments[0])) - min(lst2num(segments[0]))
            if n > 1:
                i = 1
                while i < n:
                    miss_num += max(lst2num(segments[i]))
                    i += 1
            print 'total miss', miss_num
            miss_dict[date] = miss_num
    return miss_dict


def fit_multiple_lines(lst, truncated=False, prev=0):
    lst = map(int, lst.split(','))
    lines = [[lst[0]]]
    orders = [[1]]
    n = len(lst)
    cnt = 2+prev
    for i in xrange(1, n):
        m = len(lines)
        last_elms = np.array([lines[j][-1] for j in xrange(m)])
        if lst[i] <= min(last_elms):
            lines.append([lst[i]])
            orders.append([cnt])
        else:
            last_elms[lst[i] <= last_elms] = min(last_elms)-1
            k = np.argmin(lst[i]-last_elms)
            lines[k].append(lst[i])
            orders[k].append(cnt)
        cnt += 1
    # print 'number of lines: {0}'.format(len(lines))
    # print lines
    t = len(lines)
    print 'inner segments', t, len(orders)
    return orders, lines
    # logging.debug(t)
    # if not t == 4:
    #     logging.debug(lines)
    # res = 0
    # # print 'first: {0}, last: {1}, min: {2}, max: {3}'.format(lines[0][0], lines[0][-1], min(lines[0]), max(lines[0]))
    # for i in xrange(t):
    #     if truncated:
    #         res += (lines[i][-1] - lines[i][0])
    #         # print lines[i][-1], lines[i][0]
    #     else:
    #         res += lines[i][-1]
    #         # print lines[i][-1]
    #     # print 'last: {0}, max: {1}'.format(lines[0][-1], max(lines[0]))
    # return res


def get_miss_num_sum(path):
    miss_dict = {}
    with open(path, 'r') as data:
        data.readline()
        ts_track = data.readline()
        segments = ts_track.rstrip().rsplit(':', 1)[1]
        prev = 0
        cnt = 0
        orders, lines = fit_multiple_lines(segments, prev=prev)
        for i in xrange(len(orders)):
            ax2.plot(orders[i], lines[i], '-', marker='x', ms=2)
        prev = sum([len(order) for order in orders])
        # for line in data:
        #     _, _, date, segments = line.rstrip().split(':')
        #     if date == '2016-06-09':
        #         segments = segments.split('-')[:-1]
        #         prev = 0
        #         cnt = 0
        #         for seg in segments:
        #             # seg = map(int, seg.split(','))
        #             # xaxis = [prev + i for i in xrange(1, len(seg)+1)]
        #             # print len(xaxis)
        #             # print len(seg)
        #             # ax2.scatter(xaxis, seg, marker='x', s=1, color='rgb'[cnt])
        #             # prev += len(seg)
        #             # cnt += 1
        #             orders, lines = fit_multiple_lines(seg, prev=prev)
        #             for i in xrange(len(orders)):
        #                 ax2.plot(orders[i], lines[i], '-', marker='x', ms=2)
        #             prev = sum([len(order) for order in orders])
            # print '------------------------'
            # print date
            # segments = segments.split('-')[:-1]
            # miss_num = 0
            # print 'number of connection', len(segments)
            # n = len(segments)
            # miss_num += fit_multiple_lines(segments[0], True)
            # for i in xrange(1, n):
            #     miss_num += fit_multiple_lines(segments[i], False)
            # print 'total miss', miss_num
            # miss_dict[date] = miss_num
    # return miss_dict


def date2str(date_format):
    return date_format.strftime("%Y-%m-%d")


def millions(x):
    return '%1.1fM' % (x*1e-6)


if __name__ == '__main__':
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig, ax2 = plt.subplots(1, 1)

    # logging.basicConfig(filename='../../data/test_sampling/inner_segments.log', level=logging.DEBUG)
    #
    # filepath1 = os.path.join(file_loc, 'sample_ratio.log')
    filepath2 = os.path.join(file_loc, 'tweet_rate_ts.log')
    get_miss_num_sum(filepath2)

    #
    # retrieve_map = get_retrieve_num(filepath1)
    # miss_max_map = get_miss_num_max(filepath2)
    # miss_sum_map = get_miss_num_sum(filepath2)
    #
    # with open(os.path.join(file_loc, r'retrieve_map.pickle'), 'wb') as output1:
    #     pickle.dump(retrieve_map, output1)
    # with open(os.path.join(file_loc, r'miss_max_map.pickle'), 'wb') as output2:
    #     pickle.dump(miss_max_map, output2)
    # with open(os.path.join(file_loc, r'miss_sum_map.pickle'), 'wb') as output3:
    #     pickle.dump(miss_sum_map, output3)

    # with open(os.path.join(file_loc, r'retrieve_map.pickle'), 'rb') as input1:
    #     retrieve_map = pickle.load(input1)
    # with open(os.path.join(file_loc, r'miss_max_map.pickle'), 'rb') as input2:
    #     miss_max_map = pickle.load(input2)
    # with open(os.path.join(file_loc, r'miss_sum_map.pickle'), 'rb') as input3:
    #     miss_sum_map = pickle.load(input3)
    #
    # del retrieve_map['2016-04-03']
    # date_axis = [datetime(2014, 6, 1)+timedelta(days=i) for i in xrange(len(retrieve_map))]
    # retrieve_axis = [retrieve_map[date2str(d)] if date2str(d) in retrieve_map else np.nan for d in date_axis]
    # restore_axis1 = [retrieve_map[date2str(d)]+miss_max_map[date2str(d)] if date2str(d) in retrieve_map and date2str(d) in miss_max_map else np.nan for d in date_axis]
    # restore_axis2 = [retrieve_map[date2str(d)]+miss_sum_map[date2str(d)] if date2str(d) in retrieve_map and date2str(d) in miss_max_map else np.nan for d in date_axis]
    # ax1.plot_date(date_axis, restore_axis1, '-', ms=2, color='b', label='assume max difference')
    # ax1.plot_date(date_axis, restore_axis2, '-', ms=2, color='r', label='assume multiple lines sum up')
    # ax1.plot_date(date_axis, retrieve_axis, '-', ms=2, color='g', label='practical retrieve number')
    # ax1.bar(date_axis, np.isnan(retrieve_axis)*ax1.get_ylim()[1], color=(.9, .9, .9))
    # ax1.yaxis.set_major_formatter(FuncFormatter(millions))
    # ax1.plot_date((datetime(2015,2,6), datetime(2015,2,6)), (0, ax1.get_ylim()[1]), 'm-')
    # ax1.set_xlabel('Date')
    # ax1.set_ylabel('Retrieve tweets number')
    # ax1.legend(loc='lower left')
    #
    # sampling_axis1 = [100.0*retrieve_map[date2str(d)]/(retrieve_map[date2str(d)]+miss_max_map[date2str(d)]) if date2str(d) in retrieve_map and date2str(d) in miss_max_map else np.nan for d in date_axis]
    # sampling_axis2 = [100.0*retrieve_map[date2str(d)]/(retrieve_map[date2str(d)]+miss_sum_map[date2str(d)]) if date2str(d) in retrieve_map and date2str(d) in miss_max_map else np.nan for d in date_axis]
    # ax2.plot_date(date_axis, sampling_axis1, '-', ms=2, color='b', label='assume max difference')
    # ax2.plot_date(date_axis, sampling_axis2, '-', ms=2, color='r', label='assume multiple lines sum up')
    # ax2.bar(date_axis, np.isnan(retrieve_axis) * ax2.get_ylim()[1], color=(.9, .9, .9))
    # ax2.plot_date((datetime(2015, 2, 6), datetime(2015, 2, 6)), (0, ax2.get_ylim()[1]), 'm-')
    # ax2.set_xlabel('Date')
    # ax2.set_ylabel('Retrieve ratio')
    # ax2.legend(loc='lower left')

    # plt.tight_layout()
    plt.show()

