# -*- coding: utf-8 -*-

"""Base class for YouTube API V3 crawler

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import os
import logging
import logging.config


class APIV3Crawler(object):
    """Base class for YouTube API V3 crawler.
    """
    def __init__(self):
        self._developer_key = None
        self._api_service = 'youtube'
        self._api_version = 'v3'
        self._num_threads = 1
        self.logger = None

    def set_key(self, key):
        """Set new YouTube Developer Key
        """
        self._developer_key = key

    def set_num_thread(self, n):
        """Set the number of threads used in crawling, default is 1
        """
        self._num_threads = n

    def setup_logger(self, logger_name):
        log_dir = 'log/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.config.fileConfig('conf/logging.conf')
        self.logger = logging.getLogger(logger_name)
