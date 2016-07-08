# -*- encoding: utf-8 -*-

"""
yt-longevity inner-package entry

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

import time
import bz2
import json
import cPickle as pickle

from yt_longevity.helper import YTDict
from yt_longevity.vid_extractor import VidExtractor
from YTCrawl.crawler import Crawler


def main(indir):
    # ============== Part 1: Create statistics tmp file and video id list ==============
    # yt_dict = YTDict()
    #
    # for subdir, _, files in os.walk(indir):
    #     for f in files:
    #         filepath = os.path.join(subdir, f)
    #         datafile = bz2.BZ2File(filepath, mode='r')
    #
    #         for line in datafile:
    #             if line.rstrip():
    #                 vid = VidExtractor(json.loads(line)).extract()
    #                 if vid:
    #                     yt_dict.update_tc(vid)
    #
    # pickle.dump(yt_dict.getter(), open('tmp/stat_dict', 'wb'))
    #
    # c = 0
    # with open('tmp/video_ids', 'wb') as vids:
    #     for vid in yt_dict.keys():
    #         c += 1
    #         vids.write('{0}\n'.format(vid))
    #
    # vids.close()
    #
    # c1 = len(yt_dict)
    # print 'distinct video id number', c1

    # ============== Part 2: Crawl statistics from given video ids file ==============
    crawler = Crawler()
    yt_dict = YTDict(pickle.load(open('tmp/stat_dict', 'rb')))

    print "\nbatch crawl starts"
    crawler.set_crawl_delay_time(1)
    crawler.batch_crawl("tmp/video_ids", "output", yt_dict)
    print "\nbatch crawler finishes"

