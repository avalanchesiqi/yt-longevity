#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
yt-longevity inner-package entry

Author: Siqi Wu
Date last modified: 06/07/2016
"""

import os
import sys

import bz2
import json

from yt_longevity.helper import YTDict
from yt_longevity.vid_extractor import VidExtractor


def main(indir):
    yt_dict = YTDict()
    cnt = 0

    for subdir, _, files in os.walk(indir):
        for file in files:
            filepath = os.path.join(subdir, file)

            datafile = bz2.BZ2File(filepath, mode='r')

            for line in datafile:
                if line.rstrip():
                    vid = VidExtractor(json.loads(line)).extract()
                    if vid:
                        cnt += 1
                        yt_dict.update_tc(vid)

    print cnt
    print len(yt_dict)


if __name__ == '__main__':
    main(sys.argv[1])
