#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
yt-longevity main entry

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

from yt_longevity.run import main


if __name__ == '__main__':
    indir = 'datasets'
    outdir = 'tmp'
    proc_num = 4
    main(indir, outdir, proc_num)
