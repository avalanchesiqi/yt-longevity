# -*- coding: utf-8 -*-

"""
UnitTest, YouTube video ID extractor

Author: Siqi Wu
Date last modified: 06/07/2016
"""

import unittest
import logging
import logging.config

from yt_longevity.vid_extractor import VidExtractor


class TestVidExtractor(unittest.TestCase):
    _TESTS = [

        # www.youtube.com/watch?...v={vid}...
        ({"entities":
              {"urls":
                   [{"url": "http://t.co/Hnysv6NlrP", "expanded_url": "http://www.youtube.com/watch?v=8vByVskwuLg",
                     "display_url": "youtube.com/watch?v=8vByVs\u2026", "indices": [54, 76]}]
               }}, '8vByVskwuLg'),

        # www.youtube.com/watch?...v={vid}...
        ({"entities":
              {"urls":
                   [{"url": "https://t.co/hGMA7iKcMQ", "expanded_url": "https://www.youtube.com/watch?v=OHXIsKbKvVk",
                     "display_url": "youtube.com/watch?v=OHXIsKbKvVk", "indices": [25, 48]}]
               }}, 'OHXIsKbKvVk'),

        # youtu.be/{vid}...
        ({"entities":
              {"urls":
                   [{"url": "http://t.co/4TiHSytsjX", "expanded_url": "http://youtu.be/1XODLHlAGJk",
                     "display_url": "youtu.be/1XODLHlAGJk", "indices": [45, 67]}]
               }}, '1XODLHlAGJk'),

        # youtu.be/{vid}...
        ({"entities":
              {"urls":
                   [{"url": "http://t.co/7QSZcIUwyb", "expanded_url": "http://youtu.be/iv-8-EgPEY0?a",
                     "display_url": "youtu.be/iv-8-EgPEY0?a", "indices": [22, 44]}]
               }}, 'iv-8-EgPEY0'),

        # No url
        ({"entities": {"urls": []}}, None),

        # Invalid url, not related to YouTube video
        ({"entities":
              {"urls":
                   [{"url": "http://t.co/kj42IV2QBF",
                     "expanded_url": "http://rock-on-tube.blogspot.com/2014/03/charly-sahona-relieved.html?spref=tw",
                     "display_url": "rock-on-tube.blogspot.com/2014/03/charly\u2026", "indices": [33, 55]}]
               }}, None),

        # Incomplete url, not equal to 11-digits YouTube video ID
        ({"entities":
              {"urls":
                   [{"url": "http://t.co/XM9XqrEoMa", "expanded_url": "http://youtube.com/watch?v=LrcP2Z",
                     "display_url": "youtube.com/watch?v=LrcP2Z", "indices": [91, 113]}]
               }}, None),

        # Invalid url, not related to a single YouTube video
        ({"entities":
              {"urls":
                   [{"url": "https://t.co/8GZOrZkZpO", "expanded_url": "https://m.youtube.com/watch?feature=yo",
                     "display_url": "m.youtube.com/watch?feature=\u2026", "indices": [102, 125]}]
               }}, None)

    ]

    def setUp(self):
        logging.config.fileConfig('../conf/logging.conf')
        self.testcases = self._TESTS
        # self.testcases = VidExtractor().get_testcases()

    def tearDown(self):
        self.testcases = None

    def test_extract(self):
        logger = logging.getLogger('TestVidExtractor.test_extract')
        for t in self.testcases:
            get = VidExtractor(t[0]).extract()
            expect = t[1]
            logger.debug('Get   : %r', get)
            logger.debug('Expect: %r', expect)
            self.assertEqual(get, expect, msg='Incorrect Video ID')


if __name__ == '__main__':
    unittest.main()
