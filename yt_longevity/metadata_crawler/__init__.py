# -*- coding: utf-8 -*-

"""Base class for YouTube API V3 crawler

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""


class APIV3Crawler(object):
    """Base class for YouTube API V3 crawler.
    """
    def __init__(self):
        self._developer_key = None
        self._api_service = 'youtube'
        self._api_version = 'v3'
        self._num_threads = 1

    def set_key(self, key):
        """Set new YouTube Developer Key
        """
        self._developer_key = key

    def set_num_thread(self, n):
        """Set the number of threads used in crawling, default is 1
        """
        self._num_threads = n
