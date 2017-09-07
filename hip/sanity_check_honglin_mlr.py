import numpy as np
import bz2, json
from scipy import stats
import matplotlib.pyplot as plt


def lookup_percentile(mapping, query):
    return stats.percentileofscore(mapping, query)


def _load_data(file1, file2):
    ret_dict = {}
    with open(file1, 'r') as fin1:
        fin1.readline()
        for line in fin1:
            vid, predicts = line.rstrip().split('\t', 1)
            ret_dict[vid] = np.sum(map(float, predicts.split()))
    with open(file2, 'r') as fin2:
        fin2.readline()
        for line in fin2:
            vid, predicts = line.rstrip().split('\t', 1)
            ret_dict[vid] = np.sum(map(float, predicts.split()))
    return ret_dict


if __name__ == '__main__':
    # == == == == == == == == Part 1: Load target video ids == == == == == == == == #
    target_vids = map(lambda x: x.rstrip(), open('./output/14k_vids.txt', 'r').readlines())
    print('>>> Construct percentile map on {0} videos'.format(len(target_vids)))

    # == == == == == == == == Part 2: Load true view gain == == == == == == == == #
    true_gain = {}
    with bz2.BZ2File('./active-dataset.json.bz2') as f:
        dataset = json.loads(f.readline())
        for video in dataset:
            true_gain[str(video['YoutubeID'])] = np.sum(video['dailyViewcount'][90:120])

    # == == == == == == == == Part 3: Load Honglin's view gain == == == == == == == == #
    prefix_dir = './output/Honglin-baselines/'
    history_files = [prefix_dir+'active-non-outliers.csv', prefix_dir+'active-outlier.csv']
    share_files = [prefix_dir+'active-non-outlier-shares.csv', prefix_dir+'active-outlier-shares.csv']
    tweet_files = [prefix_dir+'active-non-outlier-tweets.csv', prefix_dir+'active-outlier-tweets.csv']
    history_predicted = _load_data(*history_files)
    share_predicted = _load_data(*share_files)
    tweet_predicted = _load_data(*tweet_files)

    # == == == == == == == == Part 4: Calculate absolute percentile error == == == == == == == == #
    percentile_map = np.array([true_gain[vid] for vid in target_vids])
    true_percentile = [lookup_percentile(percentile_map, true_gain[vid]) for vid in target_vids]
    history_percentile = [lookup_percentile(percentile_map, history_predicted[vid]) for vid in target_vids]
    share_percentile = [lookup_percentile(percentile_map, share_predicted[vid]) for vid in target_vids]
    tweet_percentile = [lookup_percentile(percentile_map, tweet_predicted[vid]) for vid in target_vids]

    ape_matrix = []
    ape_matrix.append([abs(x - y) for x, y in zip(true_percentile, history_percentile)])
    ape_matrix.append([abs(x - y) for x, y in zip(true_percentile, share_percentile)])
    ape_matrix.append([abs(x - y) for x, y in zip(true_percentile, tweet_percentile)])

    # == == == == == == == == Part 4: Plot absolute percentile error == == == == == == == == #
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.boxplot(ape_matrix, labels=['History', 'Share', 'Tweet'], showfliers=False, showmeans=True, widths=0.75)

    means = [np.mean(x) for x in ape_matrix]
    means_labels = ['{0:.2f}%'.format(s) for s in means]
    pos = range(len(means))
    for tick, label in zip(pos, ax1.get_xticklabels()):
        ax1.text(pos[tick] + 1, means[tick] + 0.2, means_labels[tick], horizontalalignment='center', size=16, color='k')
    ax1.set_ylabel('absolute percentile error', fontsize=16)
    ax1.tick_params(axis='y', which='major', labelsize=16)
    ax1.tick_params(axis='x', which='major', labelsize=16)

    # remove upper and right edges
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    plt.show()
