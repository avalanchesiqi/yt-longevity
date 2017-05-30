#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from itertools import islice
import matplotlib.pyplot as plt


def get_ranking(r, py, auto):
    if r > max(py, auto):
        if py > auto:
            return 0
        else:
            return 1
    elif py > max(r, auto):
        if r > auto:
            return 2
        else:
            return 3
    else:
        if r > py:
            return 4
        else:
            return 5


if __name__ == '__main__':
    data_loc = '../mlearn/pyhip.log'
    n = 9

    fig, (ax1, ax2) = plt.subplots(2, 1)

    efficiency = []
    test_precision = []
    entire_precision = []
    # rpa, rap, pra, par, arp, apr
    test_ranking = [0, 0, 0, 0, 0, 0]
    entire_ranking = [0, 0, 0, 0, 0, 0]

    with open(data_loc, 'r') as f:
        while True:
            next_n_lines = list(islice(f, n))
            if not next_n_lines:
                break
            else:
                vid, r_test, r_entire, py_test, py_entire, py_time, auto_test, auto_entire, auto_time = next_n_lines
                vid = vid[-12:]
                py_time = float(py_time.rsplit(None, 1)[1][:-1])
                auto_time = float(auto_time.rsplit(None, 1)[1][:-1])
                r_test = float(r_test.rsplit(None, 1)[1])
                r_entire = float(r_entire.rsplit(None, 1)[1])
                py_test = float(py_test.split()[3])
                py_entire = float(py_entire.split()[3])
                auto_test = float(auto_test.split()[3])
                auto_entire = float(auto_entire.split()[3])

                efficiency.append(auto_time/py_time)
                test_precision.append(abs(py_test-auto_test)/py_test*100)
                entire_precision.append(abs(py_entire-auto_entire)/py_entire*100)

                test_ranking[get_ranking(r_test, py_test, auto_test)] += 1
                entire_ranking[get_ranking(r_entire, py_entire, auto_entire)] += 1

    print('|-------------------------- Test -----------------------|')
    print('|----- R-HIP -------|----- PY-HIP -----|--- AUTO-HIP ---|')
    print('| PY-HIP | AUTO-HIP | R-HIP | AUTO-HIP | R-HIP | PY-HIP |')
    print('|{0:8d}|{1:10d}|{2:7d}|{3:10d}|{4:7d}|{5:8d}|'.format(*test_ranking))
    print('|-------------------------------------------------------|')
    print()
    print('|------------------------- Entire ----------------------|')
    print('|----- R-HIP -------|----- PY-HIP -----|--- AUTO-HIP ---|')
    print('| PY-HIP | AUTO-HIP | R-HIP | AUTO-HIP | R-HIP | PY-HIP |')
    print('|{0:8d}|{1:10d}|{2:7d}|{3:10d}|{4:7d}|{5:8d}|'.format(*entire_ranking))
    print('|-------------------------------------------------------|')

    ax1.hist(efficiency, bins=20)
    ax1.set_xlabel('Time(auto-hip)/Time(py-hip)')
    ax1.set_ylabel('number of videos')
    ax1.set_title('Efficiency of auto grad')

    ax2.boxplot([test_precision, entire_precision], labels=['test relative error', 'entire relative error'], showfliers=False)
    ax2.set_ylabel('relative error')
    ax2.set_title('Precision of auto grad')

    plt.tight_layout()
    plt.show()
