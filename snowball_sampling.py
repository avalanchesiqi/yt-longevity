import json


def encode_vids(path):
    ret = {}
    idx = 1
    with open(path, 'r') as data:
        for line in data:
            ret[line.rstrip()] = idx
            idx += 1
    return ret


def extract_node_label(path, out_path):
    output_data = open(out_path, 'w')
    output_data.write('Id,Label\n')
    with open(path, 'r') as data:
        for line in data:
            metadata = json.loads(line.rstrip())
            output_data.write('{0},{1};{2}'.format(metadata['id'], metadata['statistics']['viewCount'], metadata['snippet']['title']))
        output_data.close()


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


def get_exponential_vid(graph_path, video, cap=None):
    vids = set()
    vids.add(video)
    with open(graph_path, 'r') as graph:
        for line in graph:
            source, target = line.rstrip().split(',')
            if source == video:
                if target in label_id_dict.values():
                    vids.add(target)
    # second layer
    vids = _get_exponential_vids(graph_path, vids, cap)
    # third layer
    vids = _get_exponential_vids(graph_path, vids, cap)
    # # fourth layer
    # vids = _get_exponential_vids(graph_path, vids, cap)
    return vids


if __name__ == '__main__':

    vid = 'e-ORhEE9VVg'
    graph_path = '../data/encoded_graph_20.csv'
    node_path = '../data/vevo_id_label_weight.txt'

    label_id_dict = {}
    with open(node_path, 'r') as node_list:
        for line in node_list:
            id, label, weight = line.rstrip().split(',')
            label_id_dict[label] = id

    chosen_videos_set = get_exponential_vid(graph_path, label_id_dict[vid])

    with open('../data/chosen_nodes.csv', 'w') as f1:
        f1.write('Id,Label,Weight\n')
        with open(node_path, 'r') as data:
            for line in data:
                if line.rstrip().split(',')[0] in chosen_videos_set:
                    f1.write(line)

    with open('../data/chosen_edges.csv', 'w') as f2:
        f2.write('Source,Target\n')
        with open(graph_path, 'r') as data:
            for line in data:
                if line.rstrip().split(',')[0] in chosen_videos_set:
                    f2.write(line)

    print len(chosen_videos_set)
