# -*- coding: utf-8 -*-

"""The dailydata_crawler to download YouTube video viewcount history, based on Honglin Yu's work

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import urllib2
import os
import time
import sys
import random
from datetime import datetime

from yt_longevity.dailydata_crawler import Crawler
from logger import Logger


class SingleCrawler(Crawler):
    """Single dailydata Crawler Class.
    """

    def __init__(self):
        Crawler.__init__(self)

    def get_header(self, vid):
        """Get the request header for single crawler
        """
        headers = []
        headers.append(('Content-Type', 'application/x-www-form-urlencoded'))
        headers.append(('Cookie', self._cookie))
        headers.append(('Origin', 'https://www.youtube.com'))
        headers.append(('Referer', 'https://www.youtube.com/watch?v=' + vid))
        headers.append(('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'))
        return headers

    def request(self, opener, vid):
        """Make a request to YouTube server

        :param opener: proxy or non-proxy opener
        :param vid: target video id
        :return: flag 1 means done, continue for next; 0 for quota limit exceed and start over
        """
        url = self.get_url(vid)
        data = self.get_post_data()
        header = self.get_header(vid)
        opener.addheaders = header

        self.mutex_delay(random.uniform(0.1, 1))

        try:
            response = opener.open(url, data, timeout=5)
        except:
            self._logger.log_log("server is down, can't get response, retry...")
            return 0

        try:
            content = response.read()
        except:
            self._logger.log_log("response read time out, retry...")
            return 0

        if '<error_message><![CDATA[Sorry, quota limit exceeded, please retry later.]]></error_message>' in content:
            self.update_cookie_and_sessiontoken(vid)
            self._logger.log_log('Quota exceed at {0}'.format(str(datetime.now())))
            self._logger.log_log('******************* ban for 10 seconds')
            time.sleep(10)
            self._logger.log_log('******************* leave punishment')
            return 0

        if '<p>Public statistics have been disabled.</p>' in content:
            self._logger.log_fail(vid, 1)
        elif '<error_message><![CDATA[Video not found.]]></error_message>' in content:
            self._logger.log_fail(vid, 2)
        elif '<error_message><![CDATA[Video is private.]]></error_message>' in content:
            self._logger.log_fail(vid, 3)
        elif 'No statistics available yet' in content:
            self._logger.log_fail(vid, 4)
        elif '<error_message><![CDATA[Invalid request.]]></error_message>' in content:
            self._logger.log_fail(vid, 5)
        else:
            try:
                self.store(vid, content)
            except Exception as exc:
                if 'can not get viewcount in the xml response' in str(exc):
                    self._logger.log_fail(vid, 6)
                else:
                    self._logger.log_log('Exception happens when successfully, {0}, {1}'.format(vid, str(exc)))
        return 1

    def start(self, input_file, output_dir):
        """Single dailydata_crawler that runs on a single thread

        :param input_file: file that contains video ids
        :param output_dir: directory that contains possible information
        """

        # If output directory not exists, create a new one
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self._input = open(input_file, mode='r')
        self._output_dir = output_dir

        self._logger = Logger(self._output_dir)

        self.update_cookie_and_sessiontoken(self._seed_vid)

        opener = urllib2.build_opener()

        print '\nStart crawling video ids from tweet video id files...\n'
        self._logger.log_log('Start crawling video ids from tweet video id files...')

        offset_file = os.path.join(output_dir, 'tmp_offset')

        if not os.path.exists(offset_file):
            offset = 0
        else:
            with open(offset_file, 'r') as f:
                offset = int(f.readline().strip())

        cnt1 = 0
        for line in self._input:
            cnt1 += 1
            if cnt1 < offset:
                continue
            elif cnt1 > (offset+30000):
                with open(offset_file, 'w+') as f:
                    f.write(str(offset+30000))
                self._input.close()
                self._logger.log_log('Hit the {0} margin'.format(offset+30000))
                self._logger.log_log('Current program exits, restart...\n')
                print '\nHit the {0} margin\nCurrent program exits, restart...\n'.format(offset+30000)
                self._logger.close()
                sys.exit()

            vid = line.rstrip('\t\n')

            flag = self.request(opener, vid)
            # fail over 5 times, pass
            cnt2 = 0
            while not flag:
                if cnt2 > 5:
                    self._logger.log_fail(vid, 7)
                    self._logger.log_log('Warning: {0} failed over 5 times\n'.format(vid))
                    break
                flag = self.request(opener, vid)
                cnt2 += 1
            continue

        self._input.close()
        print '\nFinish crawling video ids from tweet video id files.\n'
        self._logger.log_log('Finish crawling video ids from tweet video id files.')
        self._logger.close()
