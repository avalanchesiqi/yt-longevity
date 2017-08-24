"""Converter for watch percentage and relative engagement """

from __future__ import division
import os
import cPickle as pickle


def strify(iterable_struct):
    """
    Convert an iterable structure to comma separated string
    :param iterable_struct: an iterable structure
    :return: a string with comma separated
    """
    return ','.join(map(str, iterable_struct))


def write_dict_to_pickle(dict, path):
    """
    Write a dictionary object into pickle file
    :param dict: a dictionary object
    :param path: output pickle file path
    :return: 
    """
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    pickle.dump(dict, open(path, 'wb'))
