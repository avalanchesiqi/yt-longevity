#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
yt-longevity entry program

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import sys
import os
import socket

from yt_longevity.extractor.vidextractor import VideoIdExtractor
from yt_longevity.metadata_crawler.metadata_crawler import MetadataCrawler
from yt_longevity.dailydata_crawler.single_crawler import SingleCrawler
from yt_longevity.combine import combine


def extract(input_dir, output_dir, proc_num, sample_ratio):
    """Extract valid video ids from given folder, output in output_dir."""
    extractor = VideoIdExtractor(input_dir, output_dir)
    extractor.set_proc_num(proc_num)
    extractor.extract(sample_ratio)


def metadata_crawl(input_path, output_dir, idx):
    """Crawl metadata for vids in input_file from YouTube frontend server."""
    metadata_crawler = MetadataCrawler()
    metadata_crawler.set_num_thread(10)
    metadata_crawler.start(input_path, output_dir, idx)


def single_crawl(input_file):
    """Crawl dailydata for vids in input_file from YouTube frontend server in single thread."""
    single_crawler = SingleCrawler()
    single_crawler.start(input_file, 'output')


def batch_crawl():
    """Crawl daily data from YouTube frontend server in multi threads."""
    pass


if __name__ == '__main__':
    # input_dir = '/data2/proj/youtube-twitter-crawl/bz2-files_2015/'
    # output_dir = 'may_2016'
    # proc_num = 12
    # sample_ratio = 0.1
    # extract(input_dir, output_dir, proc_num, sample_ratio)

    idx_path = 'conf/idx.txt'
    if os.path.exists(idx_path):
        with open(idx_path, 'r') as idx_file:
            idx = int(idx_file.readline().rstrip())
    else:
        with open(idx_path, 'w') as idx_file:
            idx_file.write('0')
            idx = 0
    hostname = socket.gethostname()[:-10]

    input_path = 'input/{0}-{1}.txt'.format(hostname, idx)
    if os.path.exists(input_path):
        output_dir = '/mnt/data/'
        metadata_crawl(input_path, output_dir, idx)
    else:
        sys.exit(1)

    # single_crawl('plot/validvids.txt')

    # combine('datasets/metadata', '../dailydata')