import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    file_loc = '../../data/vevo'
    num_videos = []

    for subdir, _, files in os.walk(file_loc):
        for f in files:
            filepath = os.path.join(subdir, f)
            cnt = 0
            with open(filepath, 'r') as filedata:
                for line in filedata:
                    if line.rstrip():
                        cnt += 1
            num_videos.append(cnt)

    fig, ax1 = plt.subplots(1, 1)

    num_videos_dict = defaultdict(int)
    for num in num_videos:
        num_videos_dict[num] += 1

    x_axis = num_videos_dict.keys()
    y_axis = [num_videos_dict[k] for k in x_axis]

    ax1.scatter(x_axis, y_axis, marker='x', s=5, c='b')

    median_value = np.median(num_videos)
    mean_value = np.mean(num_videos)

    ax1.axvline(x=median_value)
    ax1.text(median_value+0.2, 0.8*ax1.get_ylim()[1], 'Median={0:.2f}'.format(median_value))

    ax1.axvline(x=mean_value)
    ax1.text(mean_value+1, 0.7*ax1.get_ylim()[1], 'Mean={0:.2f}'.format(mean_value))

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(xmin=1)
    ax1.set_ylim(ymin=1)
    ax1.set_xlabel('Number of videos published')
    ax1.set_ylabel('Number of VEVO artists')
    ax1.set_title('Figure 1: Distribution of video per artist.')

    plt.show()
