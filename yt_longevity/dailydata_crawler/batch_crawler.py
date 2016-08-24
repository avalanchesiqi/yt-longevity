# # -*- coding: utf-8 -*-
#
# """The dailydata_crawler to download YouTube video viewcount history, based on Honglin Yu's work
#
# Author: Siqi Wu
# Email: Siqi.Wu@anu.edu.au
# """
#
# import urllib2
# from urllib2 import Request, build_opener, HTTPCookieProcessor, HTTPHandler
# import urllib
# import threading
# import os
# import time
# import re
# import cookielib
# import sys
# import random
# import json
#
# from logger import Logger
# from xmlparser import *
#
#
# class Crawler(object):
#     """YouTube video viewcount/sharecount dailydata_crawler Class.
#
#         - for batch_crawl:
#             - input is a file
#             - output is a directory
#
#         - for single_crawl
#             - input is a video's ID
#             - output is a dictionary containing possible information
#     """
#
#     def __init__(self):
#         self._input_file = None
#         self._output_dir = None
#
#         self._num_thread = 20
#
#         self._logger = None
#
#         self._key_done = None
#
#         self._mutex_crawl = threading.Lock()
#         self._delay_mutex = None
#
#         self._cookie = ''
#         self._session_token = ''
#
#         self._is_done = False
#
#         self._update_cookie_period = 1800
#         self._update_cookie_maximum_times = 20
#         self._last_cookie_update_time = None
#         self._current_update_cookie_timer = None
#
#         self._cookie_update_delay_time = 0.1
#         self._cookie_update_on = False
#
#         self._seed_vid = 'OQSNhk5ICTI'
#
#         self._cnt = 0
#
#         self._PROXY_LIST = ['66.186.2.163:80', '54.67.117.173:8083', '216.220.165.62:8080', '45.55.173.10:3128', '52.36.84.86:8083']
#
#         self._USER_AGENT_CHOICES = [
#             'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0',
#             'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.62 Safari/537.36',
#             'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; WOW64; Trident/6.0)',
#             'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.146 Safari/537.36',
#             'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36',
#             'Mozilla/5.0 (X11; Linux x86_64; rv:24.0) Gecko/20140205 Firefox/24.0 Iceweasel/24.3.0',
#             'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:28.0) Gecko/20100101 Firefox/28.0',
#             'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:28.0) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2'
#         ]
#
#     def set_num_thread(self, n):
#         """Set the number of threads used in crawling, default is 20
#         """
#         self._num_thread = n
#
#     def set_cookie_update_period(self, t):
#         """Control how long to update cookie once, default is 30 min
#         """
#         self._update_cookie_period = t
#
#     def set_seed_video_id(self, vid):
#         """Set the seed videoID used to update cookies
#         """
#         self._seed_vid = vid
#
#     def mutex_delay(self, t):
#         """Delay some time
#         """
#         self._delay_mutex.acquire()
#         time.sleep(t)
#         self._delay_mutex.release()
#
#     def _store(self, vid, response):
#         """Store parsed response of vid with uploadDate, dailyViewcount, dailyWatchTime, dailySharecount,
#         dailySubscriber in json format
#
#         log result format, '\t' separated
#         upload_date total_viewcount total_sharecount    daily_viewcount daily_sharecount
#         """
#         outdir = self._output_dir + '/data/' + vid[0] + '/' + vid[1] + '/' + vid[2] + '/'
#         if not os.path.exists(outdir):
#             os.makedirs(outdir)
#
#         stat = parseString(response)
#         with open(outdir + vid, 'w') as f:
#             json.dump(stat, f)
#
#         upload_date = stat['uploadDate']
#         res = []
#         daily_viewcount = stat['dailyViewcount']
#         daily_sharecount = stat['dailySharecount']
#         res.append(upload_date)
#         res.append(str(sum(daily_viewcount)))
#         res.append(str(sum(daily_sharecount)))
#         res.append(str(daily_viewcount)[1:-1])
#         res.append(str(daily_sharecount)[1:-1])
#         res2 = '\t'.join(res)
#         self._logger.log_result(vid, res2)
#
#     def get_url(self, k):
#         """Get the insight API URL
#         """
#         return 'https://www.youtube.com/insight_ajax?action_get_statistics_and_data=1&v=' + k
#
#     def get_post_data(self):
#         """Get the session token
#         """
#         return urllib.urlencode({'session_token': self._session_token})
#
#     def get_header(self, k, i):
#         """Get the request header
#         """
#         headers = []
#         headers.append(('Content-Type', 'application/x-www-form-urlencoded'))
#         headers.append(('Cookie', self._cookie))
#         headers.append(('Origin', 'https://www.youtube.com'))
#         headers.append(('Referer', 'https://www.youtube.com/watch?v=' + k))
#         headers.append(('User-Agent', self._USER_AGENT_CHOICES[i]))
#         return headers
#
#     def update_cookie_and_sessiontoken(self, key):
#         # if already begin to update
#         if self._cookie_update_on:
#             return None
#         self._cookie_update_on = True
#         self.period_update(key)
#
#     def period_update(self, key):
#         # all the job is done
#         if self._is_done:
#             return None
#         # begin to update
#         self._mutex_crawl.acquire()
#
#         # if self._last_cookie_update_time != None:
#         #     time.sleep(10) # wait for the current job to finish
#
#         i = 0
#         state = 'fail'
#         while i < self._update_cookie_maximum_times:
#
#             # get cookies
#             cj = cookielib.CookieJar()
#             opener = build_opener(HTTPCookieProcessor(cj), HTTPHandler())
#             req = Request("https://www.youtube.com/watch?v=" + key)
#             f = opener.open(req)
#             src = f.read()
#
#             time.sleep(self._cookie_update_delay_time)
#
#             cookiename = ['YSC', 'PREF', 'VISITOR_INFO1_LIVE', 'ACTIVITY']
#             self._cookie = ''
#             for cookie in cj:
#                 if cookie.name in cookiename:
#                     self._cookie += (cookie.name + '=' + cookie.value + '; ')
#             self._cookie = self._cookie[0:-2]
#
#             re_st = re.compile('\'XSRF_TOKEN\'\: \"([^\"]+)\"\,')
#             self._session_token = re_st.findall(src)[0]
#
#             # # test
#             # try:
#             #     print "single crawl seed video"
#             #     print self.single_crawl(self._seed_vid)
#             # except Exception as exc:
#             #     if 'Invalid request' in str(exc):
#             #         print str(exc)
#             #         continue
#             #     else:
#             #         self._mutex_crawl.release()
#             #         # self.email('meet error when update the cookies, please set a new seed video (%s)' % str(e))
#             #         raise Exception('meet error when update the cookies, please set a new seed video (%s)' % str(exc))
#
#             state = 'success'
#             break
#
#         if state == 'fail':
#             # self.email('times of updating cookies reaches maximum, please report this on github (%s)' % str(e))
#             self._mutex_crawl.release()
#             # raise Exception('times of updating cookies reaches maximum, please report this on github (%s)' % str(exc))
#             raise Exception('times of updating cookies reaches maximum, please report this on github')
#
#         self._mutex_crawl.release()
#
#         self._last_cookie_update_time = datetime.datetime.now()
#
#         self._current_update_cookie_timer = threading.Timer(self._update_cookie_period,
#                                                             self.update_cookie_and_sessiontoken)
#         self._current_update_cookie_timer.daemon = True
#         self._current_update_cookie_timer.start()
#
#     # @staticmethod
#     # def check_https_ip(opener):
#     #     url = "https://www.privateinternetaccess.com/pages/whats-my-ip/"
#     #     try:
#     #         response = opener.open(url, timeout=5)
#     #     except:
#     #         return "Time out..."
#     #     content = response.read()
#     #     m = re.search(
#     #         '(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)',
#     #         content)
#     #     myip = m.group(0)
#     #     return myip if len(myip) > 0 else ''
#
#     # def crawl_thread(self, keyfile, yt_dict, i):
#     #     """
#     #     The function to iterate through the keyfile and try to download the data
#     #     """
#     #
#     #     proxy = urllib2.ProxyHandler({'https': self._PROXY_LIST[i]})
#     #     opener = urllib2.build_opener(proxy)
#     #
#     #     # print self.check_https_ip(opener)
#     #
#     #     while True:
#     #         # read one line from the keyfile
#     #         self._mutex_crawl.acquire()
#     #         line = keyfile.readline()
#     #         self._mutex_crawl.release()
#     #
#     #         if not line:
#     #             # the keyfile is finished
#     #             self._is_done = True
#     #             break
#     #
#     #         key = line.rstrip(' \t\n')
#     #
#     #         if key in self._key_done:
#     #             # key has already been crawled
#     #             continue
#     #
#     #         url = self.get_url(key)
#     #         data = self.get_post_data()
#     #         header = self.get_header(key, i)
#     #
#     #         try:
#     #             self.mutex_delay(random.uniform(0.1, 0.5))
#     #
#     #             opener.addheaders = header
#     #             txt = opener.open(url, data).read()
#     #
#     #             self._cnt += 1
#     #
#     #             if '<p>Public statistics have been disabled.</p>' in txt:
#     #                 self._logger.log_warn(key, 'statistics disabled', 'disabled')
#     #                 self._key_done.add(key)
#     #                 continue
#     #
#     #             if '<error_message><![CDATA[Video not found.]]></error_message>' in txt:
#     #                 self._logger.log_warn(key, 'Video not found', 'notfound')
#     #                 self._key_done.add(key)
#     #                 continue
#     #
#     #             if 'No statistics available yet' in txt:
#     #                 self._logger.log_warn(key, 'No statistics available yet', 'nostatyet')
#     #                 self._key_done.add(key)
#     #                 continue
#     #
#     #             if '<error_message><![CDATA[Invalid request.]]></error_message>' in txt:
#     #                 self._logger.log_warn(key, 'Invalid request', 'invalidrequest')
#     #                 self._key_done.add(key)
#     #                 continue
#     #
#     #             if '<error_message><![CDATA[Video is private.]]></error_message>' in txt:
#     #                 self._logger.log_warn(key, 'Private video', 'private')
#     #                 self._key_done.add(key)
#     #                 continue
#     #
#     #             if '<error_message><![CDATA[Sorry, quota limit exceeded, please retry later.]]></error_message>' in txt:
#     #                 self._logger.log_warn(key, 'Quota limit exceeded', 'quotalimit')
#     #
#     #                 # self.update_cookie_and_sessiontoken(key)
#     #                 print "*******************I am in thread", i, ", I am banned for 10 second"
#     #                 print 'finish keys:', self._cnt
#     #                 print 'Quota exceed\n' + str(datetime.datetime.now()) + '\n'
#     #                 print header[1]
#     #                 print self._session_token
#     #                 print header[4]
#     #                 print '\nQuota exceed\n' + str(datetime.datetime.now()) + '\n'
#     #                 # print self.check_current_ip2(opener)
#     #                 self.period_update(key)
#     #                 time.sleep(10)
#     #                 print "*******************I am gonna leave punishment...."
#     #                 continue
#     #
#     #             self._logger.log_done(key)
#     #             self.store(key, txt, yt_dict, i)
#     #             self._key_done.add(key)
#     #         except Exception as exc:
#     #             self._logger.log_warn(key, str(exc))
#     #             self._key_done.add(key)
#
#     # def batch_crawl(self, input_file, output_dir, yt_dict):
#     #     """
#     #
#     #     Arguments:
#     #     - `input_file`: the file that includes the keys (e.g. video IDs)
#     #     - `output_dir`: the dir to output crawled data
#     #
#     #     # Siqi Wu
#     #     - `yt_dict`: the dict that holds video_id, statistic_list key-pairs
#     #     ###
#     #     """
#     #
#     #     # Delete target folder if exists, then create a new one
#     #     if os.path.exists(output_dir):
#     #         shutil.rmtree(output_dir)
#     #     time.sleep(0.1)
#     #     os.makedirs(output_dir)
#     #
#     #     self._input_file = open(input_file, 'r')
#     #     self._output_dir = output_dir
#     #
#     #     self._logger = Logger(self._output_dir)
#     #     self._logger.add_log({'disabled': 'disabled', 'notfound': 'notfound', 'quotalimit': 'quotalimit',
#     #                           'nostatyet': 'nostatyet', 'invalidrequest': 'invalidrequest',
#     #                           'private': 'private'})
#     #
#     #     self._key_done = set(self._logger.get_key_done(['disabled', 'notfound', 'nostatyet',
#     #                                                     'invalidrequest', 'private']))
#     #
#     #     self._delay_mutex = threading.Lock()
#     #
#     #     self.update_cookie_and_sessiontoken(self._seed_vid)
#     #     threads = []
#     #     for i in range(0, self._num_thread):
#     #         threads.append(
#     #             threading.Thread(target=self.crawl_thread, args=(self._input_file, yt_dict, i)))
#     #
#     #     print '****** start time: ' + str(datetime.datetime.now()) + '\n'
#     #
#     #     for t in threads:
#     #         t.start()
#     #     for t in threads:
#     #         t.join()
#     #
#     #     self._current_update_cookie_timer.cancel()
#
#     # Single dailydata_crawler part
#     def _request(self, opener, vid):
#         """Make a request to YouTube server
#
#         :param opener: proxy or non-proxy opener
#         :param vid: targer video id
#         :return: flag 1 means done, continue for next; 0 for quota limit exceed and start over
#         """
#         url = self.get_url(vid)
#         data = self.get_post_data()
#         header = self.get_header(vid, 0)
#         opener.addheaders = header
#
#         self.mutex_delay(random.uniform(0.1, 1))
#
#         try:
#             response = opener.open(url, data, timeout=5)
#         except:
#             print "server is down, can't get response, retry..."
#             return 0
#
#         try:
#             content = response.read()
#         except:
#             print "response read time out, retry..."
#             return 0
#
#         self._cnt += 1
#
#         if '<error_message><![CDATA[Sorry, quota limit exceeded, please retry later.]]></error_message>' in content:
#             self._logger.log_warn(vid, 'Quota limit exceeded', 'quotalimit')
#             self.period_update(vid)
#             print "*******************I am banned for 10 second"
#             print 'finish keys:', self._cnt
#             print 'Quota exceed\n' + str(datetime.datetime.now()) + '\n'
#             time.sleep(10)
#             print "*******************I am gonna leave punishment...."
#             return 0
#
#         if '<p>Public statistics have been disabled.</p>' in content:
#             self._logger.log_warn(vid, 'statistics disabled', 'disabled')
#         elif '<error_message><![CDATA[Video not found.]]></error_message>' in content:
#             self._logger.log_warn(vid, 'Video not found', 'notfound')
#         elif 'No statistics available yet' in content:
#             self._logger.log_warn(vid, 'No statistics available yet', 'nostatyet')
#         elif '<error_message><![CDATA[Invalid request.]]></error_message>' in content:
#             self._logger.log_warn(vid, 'Invalid request', 'invalidrequest')
#             # self._logger.log_done(content)
#         elif '<error_message><![CDATA[Video is private.]]></error_message>' in content:
#             self._logger.log_warn(vid, 'Private video', 'private')
#         else:
#             try:
#                 self._store(vid, content)
#             except Exception as exc:
#                 self._logger.log_warn(vid, str(exc))
#         self._key_done.add(vid)
#         return 1
#
#     def single_crawl(self, input_file, output_dir):
#         """Single dailydata_crawler that runs on a single machine, assume 1 cpu 1 thread (NeCTAR m2.small instance)
#
#         :param input_file: directory that contains all video ids bz2 files
#         :param output_dir: directory that contains crawling results
#         :return:
#         """
#
#         print "\nStart crawling video ids from tweet video id files...\n"
#
#         # If not exist, create a new one
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#
#         offset_file = os.path.join(output_dir, 'tmp_offset')
#
#         if not os.path.exists(offset_file):
#             offset = 0
#         else:
#             with open(offset_file, 'r') as f:
#                 offset = int(f.readline())
#
#         self._input_file = open(input_file, mode='r')
#         self._output_dir = output_dir
#
#         self._logger = Logger(self._output_dir)
#         self._logger.add_log({'disabled': 'disabled', 'notfound': 'notfound', 'quotalimit': 'quotalimit',
#                               'nostatyet': 'nostatyet', 'invalidrequest': 'invalidrequest',
#                               'private': 'private'})
#
#         self._key_done = set(self._logger.get_key_done(['disabled', 'notfound', 'nostatyet',
#                                                         'invalidrequest', 'private']))
#
#         self._delay_mutex = threading.Lock()
#         self.update_cookie_and_sessiontoken(self._seed_vid)
#
#         opener = urllib2.build_opener()
#
#         cnt1 = 0
#         while True:
#             # read one line from the keyfile
#             line = self._input_file.readline()
#             cnt1 += 1
#             if cnt1 < offset:
#                 continue
#             if cnt1 > (offset+30000):
#                 with open(offset_file, 'w+') as f:
#                     f.write(str(offset+30000))
#                 self._input_file.close()
#                 print "hit the {0} margin".format(offset+30000)
#                 sys.exit()
#
#             if not line:
#                 # the keyfile is finished
#                 self._is_done = True
#                 break
#
#             key = line.rstrip(' \t\n')
#
#             # if key in self._key_done:
#             #     # key has already been crawled
#             #     continue
#
#             flag = self._request(opener, key)
#             # fail over 3 times, pass
#             cnt2 = 0
#             while not flag:
#                 if cnt2 > 3:
#                     break
#                 flag = self._request(opener, key)
#                 cnt2 += 1
#             continue
#
#         self._current_update_cookie_timer.cancel()
#         self._input_file.close()
#         print "\nFinish crawling video ids from tweet video id files.\n"
