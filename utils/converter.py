"""Converter for watch percentage and relative engagement """

from __future__ import division
import numpy as np


def to_watch_percentage(lookup_table, duration, re_score, lookup_keys=None):
    """Convert relative engagement to watch percentage."""
    if lookup_keys is None:
        lookup_keys = np.array(lookup_table['duration'])
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
    """Convert watch percentage to relative engagement"""
    if lookup_keys is None:
        lookup_keys = np.array(lookup_table['duration'])
    if isinstance(duration, list):
        re_list = []
        for d, s in zip(duration, wp_score):
            re_list.append(to_relative_engagement(lookup_table, d, s, lookup_keys=lookup_keys))
        return re_list
    else:
        bin_idx = np.sum(lookup_keys < duration)
        duration_bin = np.array(lookup_table[bin_idx])
        re = np.sum(duration_bin < wp_score) / 1000
        return re
