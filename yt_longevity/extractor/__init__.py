# -*- coding: utf-8 -*-

"""
Base class for YouTube video id extractor

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""


class Extractor(object):
    """Base class for YouTube extractor. All extractors must inherit from this class.
    """

    def __init__(self):
        self.input_dir = None
        self.output_dir = None
        self.num_thread = 8

    def set_input_dir(self, input_dir):
        """Set input directory used to extract, default is None.
        """
        self.input_dir = input_dir

    def set_output_dir(self, output_dir):
        """Set output directory used to extract, default is None.
        """
        self.output_dir = output_dir

    def set_num_thread(self, n):
        """Set the number of threads used in extracting, default is 8.
        """
        self.num_thread = n

    def _get_num_thread(self):
        """Set the number of threads used in extracting, default is 8.
        """
        return self.num_thread

    def extract(self):
        pass
