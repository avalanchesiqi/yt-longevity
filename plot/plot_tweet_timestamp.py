import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import seaborn as sns

file_loc = '../../data/yt_twitter_crawler'


def read_data(filepath):
    indexes = []
    timestamps = []
    with open(filepath, 'r') as filedata:
        for line in filedata:
            idx, ts = map(int, line.rstrip().split())
            indexes.append(idx)
            timestamps.append(ts)
    df = pd.DataFrame({'index': indexes, 'timestamp': timestamps})
    return df

if __name__ == '__main__':
    filepath1 = os.path.join(file_loc, 'test.txt')
    filepath2 = os.path.join(file_loc, '2014_06_28_out_of_order.log')
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    df1 = read_data(filepath1)
    # df2 = read_data(filepath2)
    sns.jointplot(x='index', y='timestamp', data=df1, kind='reg', joint_kws={'line_kws': {'color': 'red'}})
    # sns.jointplot(x='index', y='timestamp', data=df2, kind='reg', joint_kws={'line_kws': {'color': 'red'}})
    # sns.regplot(x='index', y='timestamp', data=df2, fit_reg=True, ax=ax2)
    sns.despine()
    sns.plt.show()
