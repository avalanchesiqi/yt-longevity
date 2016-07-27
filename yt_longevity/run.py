# -*- coding: utf-8 -*-

"""
yt-longevity inner-package entry

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import cPickle as pickle

from yt_longevity.extractor.helper import YTDict
from yt_longevity.extractor.vidextractor import VideoIdExtractor
from crawler.crawler import Crawler


# def batch_crawl():
#     # ============== Part 2: Crawl statistics from given video ids file ==============
#     crawler = Crawler()
#     yt_dict = YTDict(pickle.load(open('tmp/stat_dict', 'rb')))
#
#     print "\nStart batch crawling..."
#     # num of thread, corresponding to proxy num and user-agent num
#     crawler.set_num_thread(5)
#     crawler.batch_crawl("tmp/video_ids", "output", yt_dict)
#     print "\nFinish batch crawling"


# ============== Part 1: Create statistics tmp file and video id list ==============
def main(indir, outdir, proc_num):
    extractor = VideoIdExtractor(indir, outdir)
    extractor.set_proc_num(proc_num)
    extractor.extract()

    # crawler = Crawler()
    # crawler.single_crawl("{0}/{1}".format(outdir, "vids.txt"), "{0}/{1}".format(outdir, "output"))
    # batch_crawl()
