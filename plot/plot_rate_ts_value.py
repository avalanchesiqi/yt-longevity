import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import seaborn as sns

file_loc = '../../data/test_sampling'


def isincrease(lst):
    n = len(lst)
    cnt = 0
    print n
    for i in xrange(n-1):
        for j in xrange(i+1, n):
            if lst[j] < lst[i]:
                print i-j
                print lst[i] - lst[j]
                cnt += 1
                print '----------------'
            # return False
    print cnt
    return


def read_data(filepath):
    skip = 16
    with open(filepath, 'r') as filedata:
        for _ in xrange(skip):
            filedata.readline()
            filedata.readline()
        values = filedata.readline().rstrip().split(':')[-1]
        values = map(int, values.split(','))
        tss = filedata.readline().rstrip().split(':')[-1]
        # print isincrease(map(int, tss.split(',')))
        tss = map(int, tss.split(','))
        # tss = np.array(map(int, tss.split(',')))/1000
        # tss = mdate.epoch2num(tss)
    df = pd.DataFrame({'timestamp': tss[:1000], 'value': values[:1000]})
    return df

if __name__ == '__main__':
    filepath = os.path.join(file_loc, 'maximum_invert_ts.log')
    df1 = read_data(filepath)
    # df2 = read_data(filepath2)
    sns.jointplot(x='timestamp', y='value', data=df1, kind='reg', joint_kws={'line_kws': {'color': 'red'}})
    # sns.jointplot(x='timestamp', y='value', data=df1)
    # sns.jointplot(x='index', y='timestamp', data=df2, kind='reg', joint_kws={'line_kws': {'color': 'red'}})
    # sns.regplot(x='index', y='timestamp', data=df2, fit_reg=True, ax=ax2)
    sns.despine()
    sns.plt.show()
