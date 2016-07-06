# -*- coding: utf-8 -*-

"""
UnitTest, YTDict class in helper module

Author: Siqi Wu
Date last modified: 06/07/2016
"""

import unittest
import logging
import logging.config

from yt_longevity.helper import YTDict


class TestHelper(unittest.TestCase):
    def setUp(self):
        logging.config.fileConfig('../conf/logging.conf')
        self.ytdict = YTDict()

    def tearDown(self):
        self.ytdict = None

    def test_update_tweetcount_to_one(self):
        logger = logging.getLogger('TestHelper.test_update_tweetcount_to_one')
        self.ytdict.update_tc('a')
        get = self.ytdict['a'][2]
        expect = 1
        logger.debug('Get   : %r', get)
        logger.debug('Expect: %r', expect)
        self.assertEqual(get, expect, msg='update tweetcount to 1 fails')

    def test_update_tweetcount_to_two(self):
        logger = logging.getLogger('TestHelper.test_update_tweetcount_to_two')
        self.ytdict.update_tc('a')
        self.ytdict.update_tc('a')
        get = self.ytdict['a'][2]
        expect = 2
        logger.debug('Get   : %r', get)
        logger.debug('Expect: %r', expect)
        self.assertEqual(get, expect, msg='update tweetcount to 2 fails')

    def test_set_videocount(self):
        logger = logging.getLogger('TestHelper.test_set_videocount')
        self.ytdict.set_vc('a', 100)
        get = self.ytdict['a'][0]
        expect = 100
        logger.debug('Get   : %r', get)
        logger.debug('Expect: %r', expect)
        self.assertEqual(get, expect, msg='set viewcount fails')

    def test_set_sharecount(self):
        logger = logging.getLogger('TestHelper.test_set_sharecount')
        self.ytdict.set_sc('a', 10)
        get = self.ytdict['a'][1]
        expect = 10
        logger.debug('Get   : %r', get)
        logger.debug('Expect: %r', expect)
        self.assertEqual(get, expect, msg='set sharecount fails')


if __name__ == '__main__':
    unittest.main()
