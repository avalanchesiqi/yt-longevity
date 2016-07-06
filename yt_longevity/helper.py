# -*- coding: utf-8 -*-

"""
Helper module, include a custom dictionary to store viewcount, sharecount and tweetcount

Author: Siqi Wu
Date last modified: 06/07/2016
"""

from collections import defaultdict


class YTDict(object):
    """
    YouTube Statistic Dictionary Class.

    For each youtube video, this dictionary is designed to store the statistic data as value with youtube id as key.
    Specifically, {vid: (viewcount, sharecount, tweetcount)}
    """

    def __init__(self):
        self.ytdict = defaultdict(lambda: [0, 0, 0])

    def __getitem__(self, k):
        return self.ytdict[k]

    def __len__(self):
        return len(self.ytdict)

    def update_tc(self, k):
        self.ytdict[k][2] += 1

    def set_vc(self, k, vc):
        self.ytdict[k][0] = vc

    def set_sc(self, k, sc):
        self.ytdict[k][1] = sc
