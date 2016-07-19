# -*- coding: utf-8 -*-

"""
The crawler to download YouTube video viewcount history, based on Honglin's work

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import urllib2
from urllib2 import Request, build_opener, HTTPCookieProcessor, HTTPHandler
import urllib
import threading
import os
import time
import re
import cookielib
import shutil
import random

import requests

from logger import Logger
from xmlparser import *


class Crawler(object):
    """
    YouTube video daily viewcount crawler Class.

       - for batch_crawl:
         - input is a file
         - output is a directory

       - for single_crawl
         - input is a video's ID
         - output is a dictionary containing possible information
    """

    def __init__(self):
        self._input_file = None
        self._output_dir = None

        self._num_thread = 20

        self._logger = None

        self._key_done = None

        self._mutex_crawl = threading.Lock()
        self._delay_mutex = None

        self._cookie = ''
        self._session_token = ''

        self._is_done = False

        self._update_cookie_period = 1800
        self._update_cookie_maximum_times = 20
        self._last_cookie_update_time = None
        self._current_update_cookie_timer = None

        self._crawl_delay_time = 0.1
        self._cookie_update_delay_time = 0.1
        self._cookie_update_on = False

        self._seed_vid = 'OQSNhk5ICTI'

        self._cnt = 0

        # self._email_from_addr = ''
        # self._email_password = ''
        # self._email_to_addr = ''

        self._proxy_index = 0
        # self._PROXY_LIST = ['203.210.8.41:80', '188.166.245.112:8080', '202.47.236.252:8080']
        self._PROXY_LIST = ['203.210.8.41:80', '183.91.33.41:89', '203.210.6.39:80', '188.166.245.112:8080',
                            '202.47.236.252:8080']

        self._USER_AGENT_CHOICES = [
            'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:23.0) Gecko/20100101 Firefox/23.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.62 Safari/537.36',
            'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; WOW64; Trident/6.0)',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.146 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.146 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64; rv:24.0) Gecko/20140205 Firefox/24.0 Iceweasel/24.3.0',
            'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:28.0) Gecko/20100101 Firefox/28.0',
            'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:28.0) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2'
        ]

    # def set_email_reminder(self, from_addr, password, to_addrs):
    #     """set the email reminder
    #
    #     Arguments:
    #     - `from_addr`: from which the email is sent
    #     - `password`: the password of mailbox "from_addr"
    #     - `to_addrs`: the mailbox that will recieve the reminder
    #     """
    #     self._email_from_addr = from_addr
    #     self._email_password = password
    #     self._email_to_addr = to_addrs

    def set_num_thread(self, n):
        """
        Set the number of threads used in crawling, default is 20
        """
        self._num_thread = n

    def set_cookie_update_period(self, t):
        """
        Control how long to update cookie once, default is 30 min
        """
        self._update_cookie_period = t

    def set_seed_video_id(self, vid):
        """
        Set the seed videoID used to update cookies
        """
        self._seed_vid = vid

    def set_crawl_delay_time(self, t):
        """
        Set crawl delay time (in seconds), default is 0.1 which will request 10 times per second
        """
        self._crawl_delay_time = t

    def set_cookie_update_delay_time(self, t):
        """
        Set cookie update delay time (in seconds), default is 0.1 which will hold 0.1 second per cookie update operation
        """
        self._cookie_update_delay_time = t

    def _update_proxy_index(self):
        n = len(self._PROXY_LIST)
        self._proxy_index = (self._proxy_index + 1) % n

    def mutex_delay(self, t):
        """
        Delay some time
        """
        self._delay_mutex.acquire()
        time.sleep(t)
        self._delay_mutex.release()

    #TODO
    def store(self, k, txt, yt_dict, i):
        """generate the filepath of the output file of key "k"

        Arguments:
        - `k`: the key
        - `txt`: the file content
        """
        outdir = self._output_dir + '/data/' + k[0] + '/' + k[1] + '/' + k[2] + '/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(outdir + k, 'w') as f:
            f.write(txt)
        f.close()

        # Siqi Wu
        # Update the statistic dict
        stat = parseString(txt)
        sc = sum(stat['numShare'])
        vc = sum(stat['dailyViewcount'])
        yt_dict.set_sc(k, sc)
        yt_dict.set_vc(k, vc)
        self._logger.log_result(k, yt_dict[k], i)

    def get_url(self, k):
        """
        Get the insight API URL
        """
        return 'https://www.youtube.com/insight_ajax?action_get_statistics_and_data=1&v=' + k

    def get_post_data(self):
        """
        Get the session token
        """
        return urllib.urlencode({'session_token': self._session_token})

    def get_header(self, k, cookie):
        """
        Get the request header
        """
        headers = {}
        # headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        # headers['Accept-Encoding'] = 'gzip, deflate'
        # headers['Accept-Language'] = 'en-US,en;q=0.5'
        # headers['Content-Length'] = '280'
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        headers['Cookie'] = cookie
        headers['Origin'] = 'https://www.youtube.com'
        headers['Referer'] = 'https://www.youtube.com/watch?v=' + k
        headers['User-Agent'] = self._USER_AGENT_CHOICES[random.randint(0, 7)]
        return headers

    def get_header2(self, k, i):
        """
        Get the request header
        """
        headers = []
        # headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        # headers['Accept-Encoding'] = 'gzip, deflate'
        # headers['Accept-Language'] = 'en-US,en;q=0.5'
        # headers['Content-Length'] = '280'
        headers.append(('Content-Type', 'application/x-www-form-urlencoded'))
        headers.append(('Cookie', self._cookie))
        headers.append(('Origin', 'https://www.youtube.com'))
        headers.append(('Referer', 'https://www.youtube.com/watch?v=' + k))
        headers.append(('User-Agent', self._USER_AGENT_CHOICES[i]))
        return headers

    def update_cookie_and_sessiontoken(self, key):
        # # if already begin to update
        if self._cookie_update_on:
            return None
        self._cookie_update_on = True
        # print "go into update cookie and sessiontoken"
        self.period_update(key)
        # print "break update cookie and sessiontoken"

    def period_update(self, key):
        # all the job is done
        # print "one"
        if self._is_done:
            return None
        # begin to update
        # print "two"
        # print self._mutex_crawl.locked()
        self._mutex_crawl.acquire()
        # print "three"

        # if self._last_cookie_update_time != None:
        #     time.sleep(10) # wait for the current job to finish

        i = 0
        state = 'fail'
        # print "I am here again"
        while i < self._update_cookie_maximum_times:

            # get cookies
            cj = cookielib.CookieJar()
            opener = build_opener(HTTPCookieProcessor(cj), HTTPHandler())
            # req = Request("https://www.youtube.com/watch?v=" + self._seed_vid)
            req = Request("https://www.youtube.com/watch?v=" + key)
            f = opener.open(req)
            src = f.read()

            time.sleep(self._cookie_update_delay_time)

            cookiename = ['YSC', 'PREF', 'VISITOR_INFO1_LIVE', 'ACTIVITY']
            self._cookie = ''
            for cookie in cj:
                if cookie.name in cookiename:
                    self._cookie += (cookie.name + '=' + cookie.value + '; ')
            self._cookie = self._cookie[0:-2]
            # print self._cookie

            re_st = re.compile('\'XSRF_TOKEN\'\: \"([^\"]+)\"\,')
            self._session_token = re_st.findall(src)[0]
            # print self._session_token

            # print 'cookie: ', self._cookie
            # print 'session_token: ', self._session_token

            # # test
            # try:
            #     print "single crawl seed video"
            #     print self.single_crawl(self._seed_vid)
            # except Exception as exc:
            #     if 'Invalid request' in str(exc):
            #         print str(exc)
            #         continue
            #     else:
            #         self._mutex_crawl.release()
            #         # self.email('meet error when update the cookies, please set a new seed video (%s)' % str(e))
            #         raise Exception('meet error when update the cookies, please set a new seed video (%s)' % str(exc))

            state = 'success'
            # print "break????????????????"
            break

        if state == 'fail':
            # self.email('times of updating cookies reaches maximum, please report this on github (%s)' % str(e))
            self._mutex_crawl.release()
            # raise Exception('times of updating cookies reaches maximum, please report this on github (%s)' % str(exc))
            raise Exception('times of updating cookies reaches maximum, please report this on github')

        self._mutex_crawl.release()

        # print self._cookie
        # print self._session_token

        self._last_cookie_update_time = datetime.datetime.now()

        self._current_update_cookie_timer = threading.Timer(self._update_cookie_period,
                                                            self.update_cookie_and_sessiontoken)
        self._current_update_cookie_timer.daemon = True
        self._current_update_cookie_timer.start()

        # return self._cookie, self._session_token

    # def email(self, s):
    #     send_email('[ acro: video history crawling: ]', s, self._email_from_addr, self._email_password,
    #                self._email_to_addr)

    @staticmethod
    def check_current_ip():
        url = "http://checkip.dyndns.org"
        request = urllib2.Request(url)
        txt = urllib2.urlopen(request).read()
        return txt

    @staticmethod
    def check_current_ip2(opener):
        url = "http://checkip.dyndns.org"
        txt = opener.open(url).read()
        return txt

    def crawl_thread(self, keyfile, yt_dict, i):
        """
        The function to iterate through the keyfile and try to download the data
        """
        # print self.check_current_ip()

        # proxy = urllib2.ProxyHandler({'http': self._PROXY_LIST[i]})
        # # auth = urllib2.HTTPBasicAuthHandler()
        # # # print proxy.proxies
        # cj = cookielib.CookieJar()
        # opener = urllib2.build_opener(proxy, urllib2.HTTPCookieProcessor(cj))
        # opener = urllib2.build_opener(proxy)
        opener = urllib2.build_opener()
        # urllib2.install_opener(opener)

        # print "I am in thread", i, ' My ip is', self.check_current_ip2(opener), '\n'

        # print self.check_current_ip()

        # cookie_cnt = 0

        while True:
            # read one line from the keyfile
            self._mutex_crawl.acquire()
            line = keyfile.readline()
            self._mutex_crawl.release()

            if not line:
                # the keyfile is finished
                self._is_done = True
                break

            key = line.rstrip(' \t\n')

            if key in self._key_done:
                # key has already been crawled
                continue

            # if cookie_cnt == 0 or cookie_cnt > 5:
            #     cookie, sessiontoken = self.period_update(key)
            #     print "thread", i, "change cookie"
            #     print self._cookie
            #     print self._session_token
            #     cookie_cnt = 0
            # """change cookie and sessiontoken"""
            # self.period_update(key)
            url = self.get_url(key)
            data = self.get_post_data()
            header = self.get_header2(key, i)

            # print "I am in thread", i
            # print 'finish keys:', self._cnt
            # print self._PROXY_LIST[i]
            # print header[1]
            # print self._session_token
            # print header[4]
            # print '--------------------\ n\n'

            try:
                # cookie_cnt += 1
                # self.mutex_delay(self._crawl_delay_time)
                self.mutex_delay(random.uniform(0.1, 1))
                # request = urllib2.Request(url, data, headers=header)
                # r = requests.get(url, data={'session_token': self._session_token}, headers=header)
                # txt = r.text
                # txt = txt.encode('ascii', 'ignore')
                # request.set_proxy(self._PROXY_LIST[i], 'http')

                # # print self._PROXY_LIST[i]
                # # proxy = urllib2.ProxyHandler({'http': self._PROXY_LIST[i]})
                # proxy = urllib2.ProxyHandler({'http': '203.210.8.41:80'})
                # # auth = urllib2.HTTPBasicAuthHandler()
                # # # print proxy.proxies
                # opener = urllib2.build_opener(proxy)
                # urllib2.install_opener(opener)

                # print self.check_current_ip()

                # print header['User-Agent']
                opener.addheaders = header
                txt = opener.open(url, data).read()
                # txt = urllib2.urlopen(request).read()

                if '<p>Public statistics have been disabled.</p>' in txt:
                    self._logger.log_warn(key, 'statistics disabled', 'disabled')
                    self._key_done.add(key)
                    self._cnt += 1
                    continue

                if '<error_message><![CDATA[Video not found.]]></error_message>' in txt:
                    self._logger.log_warn(key, 'Video not found', 'notfound')
                    self._key_done.add(key)
                    self._cnt += 1
                    continue

                if 'No statistics available yet' in txt:
                    self._logger.log_warn(key, 'No statistics available yet', 'nostatyet')
                    self._key_done.add(key)
                    self._cnt += 1
                    continue

                if '<error_message><![CDATA[Invalid request.]]></error_message>' in txt:
                    self._logger.log_warn(key, 'Invalid request', 'invalidrequest')
                    self._key_done.add(key)
                    self._cnt += 1
                    continue

                if '<error_message><![CDATA[Video is private.]]></error_message>' in txt:
                    self._logger.log_warn(key, 'Private video', 'private')
                    self._key_done.add(key)
                    self._cnt += 1
                    continue

                if '<error_message><![CDATA[Sorry, quota limit exceeded, please retry later.]]></error_message>' in txt:
                    self._logger.log_warn(key, 'Quota limit exceeded', 'quotalimit')
                    # print self.check_current_ip2(opener)
                    # self._cookie_update_on = False
                    # self._update_proxy_index()
                    # self.update_cookie_and_sessiontoken(key)
                    # print txt

                    # self.update_cookie_and_sessiontoken(key)
                    print "*******************I am in thread", i, ", I am banned for 10 second"
                    print 'finish keys:', self._cnt
                    print self.check_current_ip2(opener)
                    print header[1]
                    print self._session_token
                    print header[4]
                    print 'Quota exceed\n' + str(datetime.datetime.now()) + '\n'
                    print self.check_current_ip2(opener)
                    self.period_update(key)
                    time.sleep(10)
                    # print self._cookie
                    # print self._session_token
                    # print self._PROXY_LIST[self._proxy_index]
                    continue

                self._logger.log_done(key)
                self.store(key, txt, yt_dict, i)
                self._key_done.add(key)
                self._cnt += 1
                # print str(datetime.datetime.now())
            except Exception as exc:
                # self._update_proxy_index()
                self._logger.log_warn(key, str(exc))
                self._key_done.add(key)
                self._cnt += 1

    def batch_crawl(self, input_file, output_dir, yt_dict):
        """

        Arguments:
        - `input_file`: the file that includes the keys (e.g. video IDs)
        - `output_dir`: the dir to output crawled data

        # Siqi Wu
        - `yt_dict`: the dict that holds video_id, statistic_list key-pairs
        ###
        """

        # Delete target folder if exists, then create a new one
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        time.sleep(0.1)
        os.makedirs(output_dir)

        # SHARE_Q = Queue.Queue()

        self._input_file = open(input_file, 'r')
        self._output_dir = output_dir

        self._logger = Logger(self._output_dir)
        self._logger.add_log({'disabled': 'disabled', 'notfound': 'notfound', 'quotalimit': 'quotalimit',
                              'nostatyet': 'nostatyet', 'invalidrequest': 'invalidrequest',
                              'private': 'private'})

        self._key_done = set(self._logger.get_key_done(['disabled', 'notfound', 'nostatyet', 'disabled',
                                                        'invalidrequest', 'private']))

        self._delay_mutex = threading.Lock()

        self.update_cookie_and_sessiontoken(self._seed_vid)
        threads = []
        for i in range(0, self._num_thread):
            threads.append(
                threading.Thread(target=self.crawl_thread, args=(self._input_file, yt_dict, i)))

        print 'Start\n' + str(datetime.datetime.now()) + '\n'

        for t in threads:
            # t.daemon = True
            # time.sleep(5)
            t.start()
            # time.sleep(10)
        for t in threads:
            t.join()

        self._current_update_cookie_timer.cancel()

    def single_crawl(self, key):
        """crawl video

        Arguments:
        - `key`: videoID
        """

        if not self._last_cookie_update_time:
            self.update_cookie_and_sessiontoken()

        url = self.get_url(key)
        data = self.get_post_data()
        header = self.get_header(key)

        txt = ''

        request = urllib2.Request(url, data, headers=header)
        txt = urllib2.urlopen(request).read()

        if '<p>Public statistics have been disabled.</p>' in txt:
            raise Exception('Statistics disabled')

        if '<error_message><![CDATA[Video not found.]]></error_message>' in txt:
            raise Exception('Video not found')

        if '<error_message><![CDATA[Sorry, quota limit exceeded, please retry later.]]></error_message>' in txt:
            raise Exception('Quota limit exceeded')

        if 'No statistics available yet' in txt:
            raise Exception('No statistics available yet')

        if '<error_message><![CDATA[Invalid request.]]></error_message>' in txt:
            raise Exception('Invalid request')

        if '<error_message><![CDATA[Video is private.]]></error_message>' in txt:
            raise Exception('Private video')

        return parseString(txt)
