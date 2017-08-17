"""Converter for watch percentage and relative engagement """

from __future__ import division
import os
import cPickle as pickle


def write_dict_to_pickle(dict, path):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    pickle.dump(dict, open(path, 'wb'))
