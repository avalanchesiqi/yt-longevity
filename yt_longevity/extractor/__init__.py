# -*- coding: utf-8 -*-

"""
Base class for Tweet extractor

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import os
import logging
import logging.config


class Extractor(object):
    """Base class for Tweet extractor. All extractors must inherit from this class."""
    def __init__(self):
        self.input_dir = None
        self.output_dir = None
        self.proc_num = 8
        self.logger = None

    def set_input_dir(self, input_dir):
        """Set input directory used to extract, default is None."""
        self.input_dir = input_dir

    def set_output_dir(self, output_dir):
        """Set output directory used to extract, default is None."""
        self.output_dir = output_dir

    def set_proc_num(self, n):
        """Set the number of processes used in extracting, default is 8."""
        self.proc_num = n

    def _setup_logger(self, logger_name):
        """Set logger from conf file."""
        log_dir = 'log/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.config.fileConfig('conf/logging.conf')
        self.logger = logging.getLogger(logger_name)

    def extract(self, sampling_ratio=1):
        """Extract data from Tweet folder, default sampling ratio is 1, which means all data."""
        pass
