#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
yt-longevity main entry

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

from yt_longevity.run import main


if __name__ == '__main__':
    indir = 'datasets'
    outdir = 'tmp'
    main(indir, outdir)
