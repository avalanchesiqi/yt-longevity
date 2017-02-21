def encode_vids(path):
    ret = {}
    idx = 1
    with open(path, 'r') as data:
        for line in data:
            ret[line.rstrip()] = idx
            idx += 1
    return ret


def _get_exponential_vids(graph_path, videos):
    vids = set()
    with open(graph_path, 'r') as graph:
        for line in graph:
            source, target = line.rstrip().split(',')
            if source in videos:
                if target in label_id_dict.values():
                    vids.add(target)
    vids.update(videos)
    return vids


def get_exponential_vid(graph_path, video):
    vids = set()
    vids.add(video)
    with open(graph_path, 'r') as graph:
        for line in graph:
            source, target = line.rstrip().split(',')
            if source == video:
                if target in label_id_dict.values():
                    vids.add(target)
    # second layer
    vids_2nd = _get_exponential_vids(graph_path, vids)
    # third layer
    vids_3rd = _get_exponential_vids(graph_path, vids_2nd)
    # fourth layer
    vids_4th = _get_exponential_vids(graph_path, vids_3rd)
    return vids_3rd, vids_4th


if __name__ == '__main__':

    vid = 'e-ORhEE9VVg'
    graph_path = '../data/encoded_graph_20.csv'
    node_path = '../data/vevo_id_label_weight.txt'

    label_id_dict = {}
    with open(node_path, 'r') as node_list:
        for line in node_list:
            id, label, weight = line.rstrip().split(',')
            label_id_dict[label] = id

    chosen_videos_set, chosen_videos_set_outer = get_exponential_vid(graph_path, label_id_dict[vid])

    with open('../data/chosen_nodes_{0}.csv'.format(vid), 'w') as f1:
        f1.write('Id,Label,Weight\n')
        with open(node_path, 'r') as data:
            for line in data:
                if line.rstrip().split(',')[0] in chosen_videos_set_outer:
                    f1.write(line)

    with open('../data/chosen_edges_{0}.csv'.format(vid), 'w') as f2:
        f2.write('Source,Target\n')
        with open(graph_path, 'r') as data:
            for line in data:
                if line.rstrip().split(',')[0] in chosen_videos_set:
                    f2.write(line)

    print len(chosen_videos_set)
