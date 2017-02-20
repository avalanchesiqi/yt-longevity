import argparse
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


def encode_graph(path, output_path, cap=None):
    output_data = open(output_path, 'w')
    output_data.write('Source,Target\n')
    with open(path, 'r') as data:
        for line in data:
            row = line.rstrip().split(',')
            node = row.pop(0)
            if node in encoded_vids_dict:
                neighbours = row[:cap]
                for neigh in neighbours:
                    if neigh in encoded_vids_dict:
                        to_write = '{0},{1}\n'.format(encoded_vids_dict[node], encoded_vids_dict[neigh])
                        output_data.write(to_write)
    output_data.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', help='input path of vid list', required=True)
    parser.add_argument('-s', help='input path of video metadata', required=True)
    parser.add_argument('-g', help='input path of csv format graph', required=True)
    parser.add_argument('-n', help='output path of node table', required=True)
    parser.add_argument('-e', help='output path of edge table', required=True)
    parser.add_argument('-c', '--cap', type=int, default=None, help='cap of top relevant list')
    args = parser.parse_args()

    encoded_vids_dict = encode_vids(args.v)
    extract_node_label(args.s, args.n)
    encode_graph(args.g, args.e, cap=args.cap)
