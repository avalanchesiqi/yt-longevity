#!/usr/bin/python

# Usage example:
# python channel_videos_crawler.py --input='<input_file>' --output='<output_dir> [--v3api] [--selenium]'

import sys
import os
import re
import time
import random
import json
import argparse
import urllib2
import urllib
import cookielib
from apiclient import discovery
from selenium import webdriver
from bs4 import BeautifulSoup
import logging

from insights_crawler.xmlparser import parsexml


YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
BASE_DIR = '../'
logging.basicConfig(filename='{0}/log/channel_videos_crawler.log'.format(BASE_DIR), level=logging.WARNING)


def list_channel_statistics(youtube, channel_id):
    """Call the API's channels.list method to list the existing channel statistics."""
    for i in xrange(0, 5):
        try:
            list_response = youtube.channels().list(
                part="snippet, statistics",
                id=channel_id
            ).execute()
            json_doc = list_response["items"][0]

            return json_doc
        except:
            time.sleep((2 ** i) + random.random())


def list_channel_videos(youtube, channel_id):
    """Call the API's search.list method to list the existing channel videos."""
    for i in xrange(0, 5):
        try:
            videos = []

            search_response = youtube.search().list(
                part="snippet",
                channelId=channel_id,
                type='video',
                order='date',
                publishedBefore='2016-08-30T00:00:00Z',
                maxResults=50,
            ).execute()

            if 'nextPageToken' in search_response:
                next_pagetoken = search_response['nextPageToken']
            else:
                next_pagetoken = None
            for search_result in search_response.get("items", []):
                videos.append(search_result["id"]["videoId"])

            while next_pagetoken is not None:
                search_response = youtube.search().list(
                    part="snippet",
                    channelId=channel_id,
                    type='video',
                    order='date',
                    publishedBefore='2016-08-30T00:00:00Z',
                    maxResults=50,
                    pageToken=next_pagetoken
                ).execute()
                if 'nextPageToken' in search_response:
                    next_pagetoken = search_response['nextPageToken']
                else:
                    next_pagetoken = None
                for search_result in search_response.get("items", []):
                    videos.append(search_result["id"]["videoId"])

            return videos
        except:
            time.sleep((2 ** i) + random.random())


def get_webdriver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--mute-audio")

    if sys.platform == 'win32':
        driver = webdriver.Chrome('../conf/webdriver/chromedriver.exe', chrome_options=chrome_options)
    elif sys.platform == 'darwin':
        driver = webdriver.Chrome('../conf/webdriver/chromedriver_mac64', chrome_options=chrome_options)
    elif sys.maxsize > 2 ** 32:
        driver = webdriver.Chrome('../conf/webdriver/chromedriver_linux64', chrome_options=chrome_options)
    else:
        driver = webdriver.Chrome('../conf/webdriver/chromedriver_linux32', chrome_options=chrome_options)
    return driver


def list_channel_videos_selenium(channel_id):
    """Simulate a browser behavior to click button via selenium"""
    target_page = 'https://www.youtube.com/channel/{0}/videos'.format(channel_id)

    driver.get(target_page)

    while True:
        try:
            loadmore_btn = driver.find_element_by_xpath("//button[contains(@aria-label,'Load more')]")
            time.sleep(random.uniform(2, 3))
            loadmore_btn.click()
            time.sleep(random.uniform(2, 3))
        except:
            logging.debug('Selenium browser: Hit end of video page for channel {0}.'.format(channel_id))
            break
    time.sleep(1)

    vids = []
    soup = BeautifulSoup(driver.page_source, 'lxml')
    video_divs = soup.find_all("div", class_="yt-lockup-content")
    for video_div in video_divs:
        vids.append(video_div.find('a')['href'][-11:])
    driver.close()
    return vids


def list_video_metadata(youtube, video_id):
    """Call the API's videos.list method to list the existing video metadata."""
    for i in xrange(0, 5):
        try:
            list_response = youtube.videos().list(
                part="snippet,statistics,topicDetails,contentDetails",
                id=video_id,
                fields='items(id,snippet(publishedAt,channelId,title,description,channelTitle,categoryId,tags),'
                       'statistics,'
                       'topicDetails,'
                       'contentDetails(duration,dimension,definition,caption,regionRestriction))'
            ).execute()
            json_doc = list_response["items"][0]

            return json_doc
        except:
            time.sleep((2 ** i) + random.random())


def get_cookie_and_sessiontoken():
    """Get cookie and sessiontoken."""
    # get cookies
    cj = cookielib.CookieJar()
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj), urllib2.HTTPHandler())
    req = urllib2.Request('https://www.youtube.com/watch?v=' + 'OQSNhk5ICTI')
    src = opener.open(req).read()

    time.sleep(random.random())

    cookiename = ['YSC', 'PREF', 'VISITOR_INFO1_LIVE', 'ACTIVITY']
    cookie = ''
    for cookie_i in cj:
        if cookie_i.name in cookiename:
            cookie += (cookie_i.name + '=' + cookie_i.value + '; ')
    cookie = cookie[0:-2]

    re_st = re.compile('\'XSRF_TOKEN\'\: \"([^\"]+)\"\,')
    sessiontoken = re_st.findall(src)[0]
    return cookie, sessiontoken


def get_url(vid):
    """Get the insight request URL."""
    return 'https://www.youtube.com/insight_ajax?action_get_statistics_and_data=1&v=' + vid


def get_post_data(sessiontoken):
    """Get the session token."""
    return urllib.urlencode({'session_token': sessiontoken})


def get_header(cookie, vid):
    """Get the request header for historical data crawler."""
    headers = []
    headers.append(('Content-Type', 'application/x-www-form-urlencoded'))
    headers.append(('Cookie', cookie))
    headers.append(('Origin', 'https://www.youtube.com'))
    headers.append(('Referer', 'https://www.youtube.com/watch?v=' + vid))
    headers.append(('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'))
    return headers


def request(opener, vid, cookie, postdata):
    """Make a request to YouTube server to get historical data."""
    url = get_url(vid)
    header = get_header(cookie, vid)
    opener.addheaders = header

    time.sleep(random.uniform(0.1, 1))

    try:
        response = opener.open(url, postdata, timeout=5)
    except:
        logging.debug('Video historical crawler: {0} server is down, can not get response, retry...'.format(vid))
        return 0, None

    try:
        content = response.read()
    except:
        logging.debug('Video historical crawler: {0} response read time out, retry...'.format(vid))
        return 0, None

    try:
        csvstring = parsexml(content)
    except:
        logging.debug('Video historical crawler: {0} corrupted or empty xml, skip...'.format(vid))
        return 1, None

    return 1, csvstring


if __name__ == "__main__":
    developer_key = open(BASE_DIR+'conf/developer.key').readline().rstrip()
    youtube = discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=developer_key)
    opener = urllib2.build_opener()
    cookie, sessiontoken = get_cookie_and_sessiontoken()
    postdata = get_post_data(sessiontoken)

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file path of channel ids, relative to base dir', required=True)
    parser.add_argument('-o', '--output', help='output dir path of video data, relative to base dir', required=True)
    parser.add_argument('--selenium', dest='selenium', action='store_true', help='crawl video list via selenium')
    parser.add_argument('--v3api', dest='selenium', action='store_false', help='crawl video list via datav3 api, default')
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output
    use_selenium = args.selenium

    if not os.path.exists(BASE_DIR+output_dir):
        os.makedirs(BASE_DIR+output_dir)

    if use_selenium:
        driver = get_webdriver()

    with open(BASE_DIR+input_path, 'r') as channel_ids:
        for cid in channel_ids:
            cid = cid.rstrip()
            with open('{0}/{1}/{2}'.format(BASE_DIR, output_dir, cid), 'w') as output:
                try:
                    if use_selenium:
                        # get video ids from selenium
                        videos = list_channel_videos_selenium(cid)
                    else:
                        # get video ids from YouTube Data3 API
                        videos = list_channel_videos(youtube, cid)

                    for vid in videos:
                        # crawl video metadata
                        try:
                            video_data = list_video_metadata(youtube, vid)
                        except Exception, e:
                            logging.debug('Video metadata crawler: Error occurred when crawl {0}:\n{1}'.format(vid, str(e)))

                        # crawl historical data
                        flag, csvstring = request(opener, vid, cookie, postdata)
                        # fail over 5 times, pass
                        for i in xrange(5):
                            flag, csvstring = request(opener, vid, cookie, postdata)
                            if flag:
                                break
                            else:
                                # time.sleep(2 ** i + random.random())
                                time.sleep(random.random())

                        if csvstring is not None:
                            startdate, days, dailyviews, totalview, dailyshares, totalshare, dailywatches, avgwatch, dailysubscribers, totalsubscriber = csvstring.split()
                            video_data['insights'] = {}
                            video_data['insights']['startDate'] = startdate
                            video_data['insights']['days'] = days
                            video_data['insights']['dailyView'] = dailyviews
                            video_data['insights']['totalView'] = totalview
                            video_data['insights']['dailyShare'] = dailyshares
                            video_data['insights']['totalShare'] = totalshare
                            video_data['insights']['dailyWatch'] = dailywatches
                            video_data['insights']['avgWatch'] = avgwatch
                            video_data['insights']['dailySubscriber'] = dailysubscribers
                            video_data['insights']['totalSubscriber'] = totalsubscriber
                        else:
                            logging.debug('Video historical crawler: {0} failed to crawl historical data'.format(vid))

                        output.write('{0}\n'.format(json.dumps(video_data)))
                except Exception, e:
                    logging.debug('Channel videos crawler: Error occurred when crawl {0}:\n{1}'.format(cid, str(e)))

    if use_selenium:
        driver.quit()
