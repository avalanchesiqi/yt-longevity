"""Converter for watch percentage and relative engagement """

from __future__ import division
import numpy as np


def to_watch_percentage(lookup_table, duration, re_score, lookup_keys=None):
    """
    Convert relative engagement to watch percentage.
    :param lookup_table: duration ~ watch percentage table, in format of dur: [1st percentile, ..., 1000th percentile]
    :param duration: target input duration
    :param re_score: target input relative engagement score
    :param lookup_keys: pre-computed duration split points, for faster computation
    """
    if lookup_keys is None:
        lookup_keys = lookup_table['duration']
    lookup_keys = np.array(lookup_keys)
    if isinstance(duration, list):
        wp_list = []
        for d, s in zip(duration, re_score):
            wp_list.append(to_watch_percentage(lookup_table, d, s, lookup_keys=lookup_keys))
        return wp_list
    else:
        bin_idx = np.sum(lookup_keys < duration)
        duration_bin = lookup_table[bin_idx]
        wp = duration_bin[int(round(re_score * 1000))]
        return wp


def to_relative_engagement(lookup_table, duration, wp_score, lookup_keys=None):
    """
    Convert watch percentage to relative engagement
    :param lookup_table: duration ~ watch percentage table, in format of dur: [1st percentile, ..., 1000th percentile]
    :param duration: target input duration
    :param wp_score: target input watch percentage score
    :param lookup_keys: pre-computed duration split points, for faster computation
    """
    if lookup_keys is None:
        lookup_keys = lookup_table['duration']
    lookup_keys = np.array(lookup_keys)
    if isinstance(duration, list):
        re_list = []
        for d, s in zip(duration, wp_score):
            re_list.append(to_relative_engagement(lookup_table, d, s, lookup_keys=lookup_keys))
        return re_list
    else:
        bin_idx = np.sum(lookup_keys < duration)
        duration_bin = np.array(lookup_table[bin_idx])
        re = np.sum(duration_bin <= wp_score) / 1000
        # re = (np.sum(duration_bin < wp_score) + np.sum(duration_bin <= wp_score)) / 2000
        return re
