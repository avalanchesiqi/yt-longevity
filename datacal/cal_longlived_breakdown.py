#!/usr/bin/env python

import os
import json
from collections import defaultdict
from multiprocessing import Process, Queue

BASE_DIR = '../../data/full_jun_data/'
CATEGORY_DICT = json.loads(open('../conf/categorydict.json', 'r').readline().rstrip())


def build_index(year):
    index = {}
    for subdir, _, files in os.walk('{0}/{1}/{2}'.format(BASE_DIR, year, 'metadata')):
        for f in files:
            filepath = os.path.join(subdir, f)
            with open(filepath, 'r') as filedata:
                for line in filedata:
                    metadata = json.loads(line.rstrip())
                    category_id = metadata['snippet']['categoryId']
                    vid = metadata['id']
                    index[vid] = category_id
    print 'finish building index at year {0}'.format(year)
    return index


def query_index(queue):
    while not queue.empty():
        year = queue.get()
        category_index = build_index(year)
        d = defaultdict(int)
        filepath = '{0}/{1}/longlived.txt'.format(BASE_DIR, year)
        with open(filepath, 'r') as filedata:
            for line in filedata:
                try:
                    vid = line.rstrip().split()[0]
                    d[CATEGORY_DICT[category_index[vid]]] += 1
                except:
                    print line
        with open('{0}/{1}/longlived_stat.txt'.format(BASE_DIR, year), 'w') as output:
            for k, v in d.items():
                output.write('{0}\t{1}\n'.format(k, v))
        print 'finish querying at year {0}'.format(year)


if __name__ == '__main__':
    processes = []
    queue = Queue()
    queue.put('2014')
    queue.put('2015')
    queue.put('2016')

    for w in xrange(3):
        p = Process(target=query_index, args=(queue,))
        p.daemon = True
        p.start()
        print 'Process {0} starts'.format(w)
        processes.append(p)

    for p in processes:
        p.join()
