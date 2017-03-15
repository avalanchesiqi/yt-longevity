# running in central machine, yt-longevity/output folder

import os
from collections import defaultdict
import numpy as np
from datetime import datetime


def get_id_label_dict(filepath):
    res_dict = {}
    with open(filepath, 'r') as filedata:
        # skip headline
        filedata.readline()
        for line in filedata:
            id, label, _ = line.rstrip().split(',')
            res_dict[int(id)] = label
    return res_dict


def get_centroid_neighbor_dict(filepath):
    res_dict = defaultdict(list)
    with open(filepath, 'r') as filedata:
        # skip headline
        filedata.readline()
        for line in filedata:
            source, target = map(int, line.rstrip().split(','))
            if source in id_label_dict and target in id_label_dict:
                res_dict[id_label_dict[target]].append(id_label_dict[source])
    return res_dict


def create_neighbor_matrix(source_id):
    header = []
    header.append(source_id)
    filepath = os.path.join(insight_data_loc, source_id[0], source_id[1], source_id)
    if os.path.isfile(filepath):
        with open(filepath, 'r') as insight:
            content = insight.readline().rstrip()
            date, days, dailyviews, totalview, dailyshares, totalshare, dailywatches, avgwatch, dailysubscribers, totalsubscriber = content.split()
            days = map(int, days.split(','))
            dailyviews = map(int, dailyviews.split(','))
            startdate = datetime(*map(int, date.split('-')))
            duration = days[-1] + 1
            matrix = np.array([0] * duration)
            for cnt, day in enumerate(days):
                matrix[day] = dailyviews[cnt]

        neighbor_vids = centroid_neighbor_dict[source_id]

        for vid in neighbor_vids:
            header.append(vid)
            filepath = os.path.join(insight_data_loc, vid[0], vid[1], vid)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as insight:
                    content = insight.readline().rstrip()
                    date, days, dailyviews, totalview, dailyshares, totalshare, dailywatches, avgwatch, dailysubscribers, totalsubscriber = content.split()
                    days = map(int, days.split(','))
                    dailyviews = map(int, dailyviews.split(','))
                    date = datetime(*map(int, date.split('-')))
                    time_gap = (startdate - date).days
                    trainings = np.array([0] * duration)
                    for cnt, day in enumerate(days):
                        if time_gap <= day < duration + time_gap:
                            trainings[day - time_gap] = dailyviews[cnt]
                    matrix = np.vstack((matrix, trainings))

        matrix = matrix.transpose()

        np.savetxt('ego_networks/ego_network_{0}.txt'.format(source_id), matrix, header=','.join(header), delimiter=',', fmt='%d')


if __name__ == '__main__':
    id_label_path = 'input/vevo_id_label_weight.txt'
    encoded_graph_path = 'input/encoded_graph.csv'
    insight_data_loc = '/mnt/data/vevo/vevo_insight_data/raw_data'

    if not os.path.exists('ego_networks'):
        os.makedirs('ego_networks')

    id_label_dict = get_id_label_dict(id_label_path)

    centroid_neighbor_dict = get_centroid_neighbor_dict(encoded_graph_path)

    for vid in centroid_neighbor_dict:
        create_neighbor_matrix(vid)
