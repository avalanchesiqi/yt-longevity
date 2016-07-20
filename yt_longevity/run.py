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

# import pymongo

from yt_longevity.helper import YTDict
from yt_longevity.vidextractor import VideoIdExtractor
from crawler.crawler import Crawler


def main(indir):
    # # ============== Part 1: Create statistics tmp file and video id list ==============
    # yt_dict = YTDict()
    #
    # fl = []
    # al = []
    # sl = []
    # for subdir, _, files in os.walk(indir):
    #     for f in files:
    #         a = 0
    #         s = set([])
    #         filepath = os.path.join(subdir, f)
    #         datafile = bz2.BZ2File(filepath, mode='r')
    #
    #         for line in datafile:
    #             if line.rstrip():
    #                 vid = VideoIdExtractor(json.loads(line)).extract()
    #                 if vid:
    #                     a += 1
    #                     s.add(vid)
    #                     yt_dict.update_tc(vid)
    #         print f, a, len(s)
    #         fl.append(f)
    #         al.append(a)
    #         sl.append(s)
    #
    # n = len(fl)
    # for i in range(n):
    #     for j in range(i+1, n):
    #         print 'num of intersection between {0} and {1}: {2}'.format(fl[i], fl[j], len(sl[i].intersection(sl[j])))
    #
    # print 'appearance in 3 years'
    # all_three = sl[0].intersection(sl[1]).intersection(sl[2])
    # print len(all_three)
    # for i in all_three:
    #     print i, yt_dict[i]
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
    # crawler.set_crawl_delay_time(0.1)
    crawler.set_num_thread(5)
    crawler.batch_crawl("tmp/video_ids", "output", yt_dict)
    print "\nbatch crawler finishes"

