# -*- coding: utf-8 -*-

"""
UnitTest, YouTube video ID extractor

Author: Siqi Wu
Date last modified: 06/07/2016
"""

import sys
import unittest
import logging

from yt_longevity.vid_extractor import VidExtractor


class TestVidExtractor(unittest.TestCase):
    def setUp(self):
        self.testcases = VidExtractor().get_testcases()

    def tearDown(self):
        self.testcases = None

    def test_extract(self):
        log = logging.getLogger('TestVidExtractor.test_extract')
        for t in self.testcases:
            get = VidExtractor(t[0]).extract()
            expect = t[1]
            log.debug('Get: %r', get)
            log.debug('Expect: %r', expect)
            self.assertEqual(get, expect, msg='Incorrect Video ID')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger('TestVidExtractor.test_extract').setLevel(logging.DEBUG)
    unittest.main()
