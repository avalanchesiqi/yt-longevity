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
        self.proc_num = 8

    def set_input_dir(self, input_dir):
        """Set input directory used to extract, default is None.
        """
        self.input_dir = input_dir

    def set_output_dir(self, output_dir):
        """Set output directory used to extract, default is None.
        """
        self.output_dir = output_dir

    def set_proc_num(self, n):
        """Set the number of processes used in extracting, default is 8.
        """
        self.proc_num = n

    def extract(self):
        pass
