#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
yt-longevity entry program

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import sys
import os
import argparse

from yt_longevity.extractor.vidextractor import VideoIdExtractor
from yt_longevity.v3api_crawler.metadata_crawler import MetadataCrawler
from yt_longevity.v3api_crawler.relevant_crawler import RelevantCrawler
from yt_longevity.insights_crawler.single_crawler import SingleCrawler


def extract(input_dir, output_dir, proc_num, sample_ratio):
    """Extract valid video ids from given folder, output in output_dir."""
    extractor = VideoIdExtractor(input_dir, output_dir)
    extractor.set_proc_num(proc_num)
    extractor.extract(sample_ratio)


def metadata_crawl(input_path, output_dir, thread_num=1):
    """Crawl metadata for vids in input_file from YouTube frontend server."""
    metadata_crawler = MetadataCrawler()
    metadata_crawler.set_num_thread(thread_num)
    metadata_crawler.start(input_path, output_dir)


def relevant_crawl(input_path, output_dir, thread_num=1):
    """Crawl relevant lists for vids in input_file from YouTube V3API server."""
    relevant_crawler = RelevantCrawler()
    relevant_crawler.set_num_thread(thread_num)
    relevant_crawler.start(input_path, output_dir)


def single_crawl(input_path, output_dir):
    """Crawl dailydata for vids in input_file from YouTube frontend server in single thread."""
    single_crawler = SingleCrawler()
    single_crawler.start(input_path, output_dir)


def batch_crawl():
    """Crawl daily data from YouTube frontend server in multi threads."""
    pass


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--function', help='function of conduction', required=True)
    parser.add_argument('-i', '--input', help='input path of tweets or vids', required=True)
    parser.add_argument('-o', '--output', help='output dir of results', required=True)
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output

    # If output directory not exists, create a new one
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.function == 'extract':
        proc_num = 12
        sample_ratio = 1.0
        extract(input_path, output_dir, proc_num, sample_ratio)
    elif args.function == 'metadata':
        thread_num = 1
        metadata_crawl(input_path, output_dir, thread_num=thread_num)
    elif args.function == 'dailydata':
        single_crawl(input_path, output_dir)
    elif args.function == 'relevant':
        thread_num = 1
        relevant_crawl(input_path, output_dir, thread_num=thread_num)
    else:
        print 'You have entered a wrong function!!!'
