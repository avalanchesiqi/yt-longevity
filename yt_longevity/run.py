#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
yt-longevity inner-package entry

Author: Siqi Wu
Date last modified: 06/07/2016
"""

import os
import sys
import time

import bz2
import json
import cPickle as pickle

from yt_longevity.helper import YTDict
from yt_longevity.vid_extractor import VidExtractor
from YTCrawl.crawler import Crawler


def main(indir):
    # yt_dict = YTDict()
    crawler = Crawler()
    # c1 = 0
    #
    # for subdir, _, files in os.walk(indir):
    #     for file in files:
    #         filepath = os.path.join(subdir, file)
    #
    #         datafile = bz2.BZ2File(filepath, mode='r')
    #
    #         for line in datafile:
    #             if line.rstrip():
    #                 vid = VidExtractor(json.loads(line)).extract()
    #                 if vid:
    #                     c1 += 1
    #                     yt_dict.update_tc(vid)
    #
    # print 'overall video id number', c1
    # print 'distinct video id number', len(yt_dict)
    #
    # pickle.dump(yt_dict.getter(), open('../tmp/tmp_dict', 'wb'))
    yt_dict = YTDict(pickle.load(open('../tmp/tmp_dict', 'rb')))
    c1 = len(yt_dict)
    print 'distinct video id number', c1

    c2 = 0
    for vid in yt_dict.keys():
        print vid
        try:
            time.sleep(0.1)
            stat = crawler.single_crawl(vid)
            sharecount = sum(stat['numShare'])
            viewcount = sum(stat['dailyViewcount'])
            yt_dict.set_sc(vid, sharecount)
            yt_dict.set_vc(vid, viewcount)
            print yt_dict[vid]
            c2 += 1
            print c2
        except Exception as e:
            print str(e)
            pass

    print 'video with useful info', c2
    print 'available rate: %.4f%%' % (100.0*c2/c1)


if __name__ == '__main__':
    main(sys.argv[1])
