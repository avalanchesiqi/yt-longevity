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

    def __init__(self, ytdict=None):
        if ytdict:
            self._ytdict = ytdict
        else:
            self._ytdict = defaultdict(lambda: [0, 0, 0])

    def __getitem__(self, k):
        return self._ytdict[k]

    def __len__(self):
        return len(self._ytdict)

    def getter(self):
        return dict(self._ytdict)

    def keys(self):
        return self._ytdict.keys()

    def values(self):
        return self._ytdict.values()

    def items(self):
        return self._ytdict.items()

    def update_tc(self, k):
        self._ytdict[k][2] += 1

    def set_vc(self, k, vc):
        self._ytdict[k][0] = vc

    def set_sc(self, k, sc):
        self._ytdict[k][1] = sc
