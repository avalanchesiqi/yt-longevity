# -*- coding: utf-8 -*-

"""
yt-longevity inner-package entry

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import os

import bz2
import json
import cPickle as pickle

from yt_longevity.helper import YTDict
from yt_longevity.vidextractor import VideoIdExtractor
from crawler.crawler import Crawler


def fetch_vid(indir):
    # ============== Part 1: Create statistics tmp file and video id list ==============
    print "\nStart extracting video ids from tweet bz2 files..."

    yt_dict = YTDict()

    for subdir, _, files in os.walk(indir):
        for f in files:
            filepath = os.path.join(subdir, f)
            datafile = bz2.BZ2File(filepath, mode='r')

            for line in datafile:
                if line.rstrip():
                    try:
                        vid = VideoIdExtractor(json.loads(line)).extract()
                    except:
                        continue
                    if vid:
                        yt_dict.update_tc(vid)

    pickle.dump(yt_dict.getter(), open('tmp/stat_dict', 'wb'))

    with open('tmp/video_ids', 'wb') as vids:
        for vid in yt_dict.keys():
            vids.write('{0}\n'.format(vid))

    print "\nFinish extracting video ids from tweet bz2 files"
    print 'Number of distinct video ids: {0}\n'.format(len(yt_dict))


def batch_crawl():
    # ============== Part 2: Crawl statistics from given video ids file ==============
    crawler = Crawler()
    yt_dict = YTDict(pickle.load(open('tmp/stat_dict', 'rb')))

    print "\nStart batch crawling..."
    # num of thread, corresponding to proxy num and user-agent num
    crawler.set_num_thread(5)
    crawler.batch_crawl("tmp/video_ids", "output", yt_dict)
    print "\nFinish batch crawling"


def main(indir):
    # fetch_vid(indir)
    batch_crawl()
