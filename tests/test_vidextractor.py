# -*- coding: utf-8 -*-

"""
UnitTest, YouTube video id extractor

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import unittest
import logging
import logging.config

from yt_longevity.extractor.vidextractor import VideoIdExtractor
from yt_longevity.exceptions import InvalidVideoIdError


class TestVidExtractor(unittest.TestCase):
    _PASS_TESTS = [
        # www.youtube.com/watch?...v={vid}...
        ({"entities":
            {"urls":
                [{"url": "http://t.co/Hnysv6NlrP", "expanded_url": "http://www.youtube.com/watch?v=8vByVskwuLg",
                  "display_url": "youtube.com/watch?v=8vByVs\u2026", "indices": [54, 76]}]}}, '8vByVskwuLg'),

        # www.youtube.com/watch?...v={vid}...
        ({"entities":
            {"urls":
                [{"url": "https://t.co/hGMA7iKcMQ", "expanded_url": "https://www.youtube.com/watch?v=OHXIsKbKvVk",
                  "display_url": "youtube.com/watch?v=OHXIsKbKvVk", "indices": [25, 48]}]}}, 'OHXIsKbKvVk'),

        # youtu.be/{vid}...
        ({"entities":
            {"urls":
                [{"url": "http://t.co/4TiHSytsjX", "expanded_url": "http://youtu.be/1XODLHlAGJk",
                  "display_url": "youtu.be/1XODLHlAGJk", "indices": [45, 67]}]}}, '1XODLHlAGJk'),

        # youtu.be/{vid}...
        ({"entities":
            {"urls":
                [{"url": "http://t.co/7QSZcIUwyb", "expanded_url": "http://youtu.be/iv-8-EgPEY0?a",
                  "display_url": "youtu.be/iv-8-EgPEY0?a", "indices": [22, 44]}]}}, 'iv-8-EgPEY0')
    ]

    _EXCEPTION_TESTS = [
        # No entities
        {},

        # No url
        {"entities": {"urls": []}},

        # Invalid url, not related to YouTube video
        {"entities":
            {"urls":
                [{"url": "http://t.co/kj42IV2QBF",
                  "expanded_url": "http://rock-on-tube.blogspot.com/2014/03/charly-sahona-relieved.html?spref=tw",
                  "display_url": "rock-on-tube.blogspot.com/2014/03/charly\u2026", "indices": [33, 55]}]}},

        # Invalid url, not related to a single YouTube video
        {"entities":
            {"urls":
                [{"url": "https://t.co/8GZOrZkZpO", "expanded_url": "https://m.youtube.com/watch?feature=yo",
                  "display_url": "m.youtube.com/watch?feature=\u2026", "indices": [102, 125]}]}},

        # Incomplete url, not equal to 11-digits YouTube video ID
        {"entities":
             {"urls":
                  [{"url": "http://t.co/XM9XqrEoMa", "expanded_url": "http://youtube.com/watch?v=LrcP2Z",
                    "display_url": "youtube.com/watch?v=LrcP2Z", "indices": [91, 113]}]}},

        # Invalid url, ascii encoding issue
        {"entities":
            {"urls":
                [{"url": "https://t.co/BiF0KkhHTE", "expanded_url": "https://youtu.be/çqGQN_5mTIAI",
                  "display_url": "youtu.be/çqGQN_5mTIAI", "indices": [79, 102]}]}}
    ]

    def setUp(self):
        logging.config.fileConfig('../conf/logging.conf')
        self._pass_testcases = self._PASS_TESTS
        self._exc_testcases = self._EXCEPTION_TESTS

    def tearDown(self):
        self._pass_testcases = None
        self._exc_testcases = None

    def test_extract(self):
        _logger = logging.getLogger('TestVidExtractor.test_extract')

        # test pass cases
        for tc in self._pass_testcases:
            expect = tc[1]
            get = VideoIdExtractor(tc[0]).extract()
            _logger.debug('Get    : %r', get)
            _logger.debug('Expect : %r', expect)
            self.assertEqual(get, expect, msg='Incorrect Video ID')

        # test exception cases
        for tc in self._exc_testcases:
            try:
                VideoIdExtractor(tc).extract()
            except Exception as exc:
                self.assertIsInstance(exc, InvalidVideoIdError)
                _logger.debug('Error  : %r', exc.message)


if __name__ == '__main__':
    unittest.main()
