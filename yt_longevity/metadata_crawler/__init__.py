# -*- coding: utf-8 -*-

"""Base class for YouTube API V3 crawler

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import sys
import os
import logging
import logging.config


class APIV3Crawler(object):
    """Base class for YouTube API V3 crawler."""
    def __init__(self):
        self._keys = None
        self._key_index = 0
        self._api_service = 'youtube'
        self._api_version = 'v3'
        self._num_threads = 1
        self.logger = None

    def set_keys(self, keys):
        """Set YouTube Developer Key list, default is None."""
        self._keys = keys

    def set_key_index(self, key_index):
        """Set YouTube Developer Key index, default is 0."""
        self._key_index = key_index

    def set_num_thread(self, n):
        """Set the number of threads used in crawling, default is 1."""
        self._num_threads = n

    def _setup_logger(self, logger_name):
        """Set logger from conf file."""
        log_dir = 'log/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.config.fileConfig('conf/logging.conf')
        self.logger = logging.getLogger(logger_name)
