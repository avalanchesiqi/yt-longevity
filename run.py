#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
yt-longevity entry program

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

from yt_longevity.extractor.vidextractor import VideoIdExtractor
from yt_longevity.metadata_crawler.metadata_crawler import MetadataCrawler
from yt_longevity.dailydata_crawler.single_crawler import SingleCrawler


def extract(input_dir, output_dir, proc_num):
    """Extract valid video ids from given folder, output in output_dir
    """
    extractor = VideoIdExtractor(input_dir, output_dir)
    extractor.set_proc_num(proc_num)
    extractor.extract()


def metadata_crawl(input_file, developer_key):
    """Crawl metadata for vids in input_file from YouTube frontend server with given developer_key
    """
    metadata_crawler = MetadataCrawler(developer_key)
    metadata_crawler.start(input_file, 'output')


def single_crawl(input_file):
    """Crawl dailydata for vids in input_file from YouTube frontend server in single thread
    """
    single_crawler = SingleCrawler()
    single_crawler.start(input_file, 'output')


def batch_crawl():
    """Crawl daily data from YouTube frontend server in multi threads
    """
    pass


if __name__ == '__main__':
    developer_key = "AIzaSyBxNpscfnZ5-we_4-PfGEB4LIadRYOjs-M"
    metadata_crawl('feb_vids.txt', developer_key)
    # single_crawl('plot/validvids.txt')
    # indir = 'datasets'
    # outdir = 'plot'
    # proc_num = 4
    # main(indir, outdir, proc_num)
