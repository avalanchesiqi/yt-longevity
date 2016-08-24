# -*- coding: utf-8 -*-

"""Base class for YouTube dailydata crawler

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

from urllib2 import Request, build_opener, HTTPCookieProcessor, HTTPHandler
import urllib
import threading
import os
import time
import re
import cookielib
import json

from xmlparser import parseString


class Crawler(object):
    """Base class for YouTube dailydata crawler. Both single crawler and batch crawler must inherit from this class.
    """

    def __init__(self):
        self._input = None
        self._output_dir = None

        self._logger = None

        self._mutex_crawl = threading.Lock()
        self._mutex_delay = threading.Lock()

        self._cookie = ''
        self._sessiontoken = ''

        self._cookie_update_delay_time = 0.1

        self._seed_vid = 'OQSNhk5ICTI'

    def set_seed_video_id(self, vid):
        """Set the seed videoID used to update cookies
        """
        self._seed_vid = vid

    def mutex_delay(self, t):
        """Delay some time
        """
        self._mutex_delay.acquire()
        time.sleep(t)
        self._mutex_delay.release()

    def get_url(self, vid):
        """Get the insight API URL
        """
        return 'https://www.youtube.com/insight_ajax?action_get_statistics_and_data=1&v=' + vid

    def get_post_data(self):
        """Get the session token
        """
        return urllib.urlencode({'session_token': self._sessiontoken})

    def get_header(self, vid):
        """Get the request header
        """
        pass

    def update_cookie_and_sessiontoken(self, key):
        """Update cookie and sessiontoken
        """
        # begin to update
        self._mutex_crawl.acquire()

        # get cookies
        cj = cookielib.CookieJar()
        opener = build_opener(HTTPCookieProcessor(cj), HTTPHandler())
        req = Request("https://www.youtube.com/watch?v=" + key)
        src = opener.open(req).read()

        time.sleep(self._cookie_update_delay_time)

        cookiename = ['YSC', 'PREF', 'VISITOR_INFO1_LIVE', 'ACTIVITY']
        self._cookie = ''
        for cookie in cj:
            if cookie.name in cookiename:
                self._cookie += (cookie.name + '=' + cookie.value + '; ')
        self._cookie = self._cookie[0:-2]

        re_st = re.compile('\'XSRF_TOKEN\'\: \"([^\"]+)\"\,')
        self._sessiontoken = re_st.findall(src)[0]

        self._mutex_crawl.release()

    def store(self, vid, response):
        """Store parsed response of vid with uploadDate, dailyViewcount, dailyWatchTime, dailySharecount,
        dailySubscriber in json format

        log result format, '\t' separated
        upload_date total_viewcount total_sharecount    daily_viewcount daily_sharecount
        """
        outdir = self._output_dir + '/data/' + vid[0] + '/' + vid[1] + '/' + vid[2] + '/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        stat = parseString(response)
        with open(outdir + vid, 'w') as f:
            json.dump(stat, f)

        upload_date = stat['uploadDate']
        res = []
        daily_viewcount = stat['dailyViewcount']
        daily_sharecount = stat['dailySharecount']
        res.append(upload_date)
        res.append(str(sum(daily_viewcount)))
        res.append(str(sum(daily_sharecount)))
        res.append(str(daily_viewcount)[1:-1])
        res.append(str(daily_sharecount)[1:-1])
        res2 = '\t'.join(res)
        self._logger.log_result(vid, res2)
