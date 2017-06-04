#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from multiprocessing import Process, Queue
import requests
from bs4 import BeautifulSoup
import logging


def is_official(channel_id):
    featured_page = 'https://www.youtube.com/channel/{0}/featured'.format(channel_id)
    try:
        r = requests.get(featured_page)
        soup = BeautifulSoup(r.text, 'lxml')
        for line in soup.find_all('a'):
            url = line.get('href')
            try:
                if 'support.google.com/youtube/answer/3046484?hl=en' in url:
                    logging.warning(channel_id)
                    return True
            except:
                continue
        return False
    except:
        return is_official(channel_id)


def filter_verified_channel(path, output):
    vids = set()
    with open(path, mode='r') as data:
        for line in data:
            if line.rstrip():
                res = json.loads(line.rstrip())
                try:
                    vid = res['snippet']['id']
                    channel_title = res['snippet']['channelTitle']
                    if vid not in vids:
                        with open('{0}/{1}'.format(output, channel_title), 'a+') as output_file:
                            output_file.write(line)
                        vids.add(vid)
                except Exception:
                    continue
    print 'filter_verified_channel, Done with file', path


def get_vevo_channel(path, output):
    cnt = 0
    with open(path, mode='r') as data:
        for line in data:
            if line.rstrip():
                res = json.loads(line.rstrip())
                try:
                    channel_title = res['snippet']['channelTitle']
                    if 'vevo' in channel_title.lower():
                        channel_id = res['snippet']['channelId']
                        with open('{0}/{1}'.format(output, channel_id), 'a+') as output_file:
                            output_file.write(line)
                        cnt += 1
                except Exception:
                    continue
    print 'get_vevo_channel, Done with file {0}, get vevo videos {1}'.format(path, cnt)


def start_query(queue, output_dir, type):
    while not queue.empty():
        path = queue.get()
        if type == 0:
            get_vevo_channel(path, output_dir)
        if type == 1:
            filter_verified_channel(path, output_dir)


def parallel_query(input_dir, output_dir, type, num_thread=1):
    processes = []
    wqueue = Queue()

    for subdir, _, files in os.walk(input_dir):
        for f in sorted(files):
            if type == 0 or (type == 1 and is_official(f)):
                filepath = os.path.join(subdir, f)
                wqueue.put(filepath)

    for w in xrange(num_thread):
        p = Process(target=start_query, args=(wqueue, output_dir, type))
        p.daemon = True
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    input_dir = 'metadata'
    tmp_dir = 'vevo1'
    output_dir = 'vevo'
    num_thread = 6

    logging.basicConfig(filename='official_vevo_channels.log', level=logging.WARNING)

    # get channels that contain vevo keyword
    parallel_query(input_dir, tmp_dir, type=0, num_thread=num_thread)

    # verify badge
    parallel_query(tmp_dir, output_dir, type=1, num_thread=num_thread)
