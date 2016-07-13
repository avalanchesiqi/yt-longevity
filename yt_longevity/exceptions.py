# -*- coding: utf-8 -*-

"""
yt-longevity core exceptions

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""


class InvalidVideoIdError(Exception):
    """Indicate YouTube video Id is not valid"""
    def __init__(self, msg):
        self.message = msg
