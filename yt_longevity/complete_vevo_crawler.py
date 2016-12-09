#!/usr/bin/python

# Usage example:
# python channel_localizations.py --action='<action>' --channel_id='<channel_id>'

import sys
import os
import re
import random
import json
import urllib2
import urllib
import cookielib
import time
from apiclient import discovery, errors
from selenium import webdriver
from bs4 import BeautifulSoup

from dailydata_crawler.xmlparser import parsexml


YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


# Call the API's channels.list method to list the existing channel statistics.
def list_channel_statistics(youtube, channel_id):
    list_response = youtube.channels().list(
        part="snippet, statistics",
        id=channel_id
    ).execute()

    json_doc = list_response["items"][0]
    return json_doc


# Call the API's search.list method to list the existing channel videos.
def list_channel_videos(youtube, channel_id):
    search_response = youtube.search().list(
        type='video',
        part="snippet",
        channelId=channel_id,
        maxResults=50,
    ).execute()

    videos = set()

    if 'nextPageToken' in search_response:
        next_pagetoken = search_response['nextPageToken']
    else:
        next_pagetoken = None
    for search_result in search_response.get("items", []):
        videos.add(search_result["id"]["videoId"])

    while next_pagetoken is not None:
        search_response = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=50,
            type='video',
            pageToken=next_pagetoken
        ).execute()
        if 'nextPageToken' in search_response:
            next_pagetoken = search_response['nextPageToken']
        else:
            next_pagetoken = None
        for search_result in search_response.get("items", []):
            videos.add(search_result["id"]["videoId"])

    return videos


# Simulate a browser behavior via selenium
def list_channel_videos_selenium(channel_id):
    target_page = 'https://www.youtube.com/channel/{0}/videos'.format(channel_id)

    if sys.platform == 'win32':
        driver = webdriver.Chrome('../conf/webdriver/chromedriver.exe')
    elif sys.platform == 'darwin':
        driver = webdriver.Chrome('../conf/webdriver/chromedriver_mac64')
    elif sys.maxsize > 2**32:
        driver = webdriver.Chrome('../conf/webdriver/chromedriver_linux64')
    else:
        driver = webdriver.Chrome('../conf/webdriver/chromedriver_linux32')
    driver.get(target_page)

    while True:
        try:
            loadmore_btn = driver.find_element_by_xpath("//button[contains(@aria-label,'Load more')]")
            time.sleep(random.uniform(2, 3))
            loadmore_btn.click()
            time.sleep(random.uniform(2, 3))
        except:
            print 'Hit end of video page.'
            break
    time.sleep(1)

    vids = []
    soup = BeautifulSoup(driver.page_source, 'lxml')
    video_divs = soup.find_all("div", class_="yt-lockup-content")
    for video_div in video_divs:
        vids.append(video_div.find('a')['href'][-11:])
    driver.quit()
    return vids


# Call the API's videos.list method to list the existing video metadata.
def list_video_metadata(youtube, video_id):
    list_response = youtube.videos().list(
        part="snippet,statistics,topicDetails,contentDetails",
        id=video_id,
        fields='items(id,snippet(publishedAt,channelId,title,description,channelTitle,categoryId,tags),statistics,topicDetails,contentDetails(duration,dimension,definition,caption,regionRestriction))'
    ).execute()

    json_doc = list_response["items"][0]
    return json_doc


# Get cookie and sessiontoken
def get_cookie_and_sessiontoken():
    # get cookies
    cj = cookielib.CookieJar()
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj), urllib2.HTTPHandler())
    req = urllib2.Request('https://www.youtube.com/watch?v=' + 'OQSNhk5ICTI')
    src = opener.open(req).read()

    time.sleep(0.1)

    cookiename = ['YSC', 'PREF', 'VISITOR_INFO1_LIVE', 'ACTIVITY']
    cookie = ''
    for cookie_i in cj:
        if cookie_i.name in cookiename:
            cookie += (cookie_i.name + '=' + cookie_i.value + '; ')
    cookie = cookie[0:-2]

    re_st = re.compile('\'XSRF_TOKEN\'\: \"([^\"]+)\"\,')
    sessiontoken = re_st.findall(src)[0]
    return cookie, sessiontoken


# Get the insight API URL
def get_url(vid):
    return 'https://www.youtube.com/insight_ajax?action_get_statistics_and_data=1&v=' + vid


# Get the session token
def get_post_data(sessiontoken):
    return urllib.urlencode({'session_token': sessiontoken})


# Get the request header for single crawler
def get_header(cookie, vid):
    headers = []
    headers.append(('Content-Type', 'application/x-www-form-urlencoded'))
    headers.append(('Cookie', cookie))
    headers.append(('Origin', 'https://www.youtube.com'))
    headers.append(('Referer', 'https://www.youtube.com/watch?v=' + vid))
    headers.append(('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'))
    return headers


# Make a request to YouTube server to get historical data
def request(opener, vid, cookie, postdata):
    url = get_url(vid)
    header = get_header(cookie, vid)
    opener.addheaders = header

    try:
        response = opener.open(url, postdata, timeout=5)
    except:
        print ('Video historical crawler: : {0} server is down, can not get response, retry...\n'.format(vid))
        return 0, None

    try:
        content = response.read()
    except:
        print ('Video historical crawler: : {0} response read time out, retry...\n'.format(vid))
        return 0, None

    time.sleep(random.uniform(0.1, 1))

    try:
        csvstring = parsexml(content)
    except Exception, e:
        print ('Video historical crawler: : {0} {1}\n'.format(vid, str(e)))
        return 1, None

    return 1, csvstring


if __name__ == "__main__":
    base_dir = '../'
    developer_key = open(base_dir+'conf/developer.key').readline().rstrip()
    youtube = discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=developer_key)
    opener = urllib2.build_opener()
    cookie, sessiontoken = get_cookie_and_sessiontoken()
    postdata = get_post_data(sessiontoken)

    if not os.path.exists(base_dir+'data/vevo/'):
        os.makedirs(base_dir+'data/vevo/')

    with open(base_dir+'data/channel_ids.txt', 'r') as channel_ids:
        for cid in channel_ids:
            cid = cid.rstrip()
            with open(base_dir+'data/vevo/{0}'.format(cid), 'w') as output:
                try:
                    # get video ids from YouTube Data3 API
                    videos = list_channel_videos(youtube, cid)
                    # get video ids from selenium
                    # videos = list_channel_videos_selenium(cid)

                    print '{0} videos are waiting to be crawled'.format(len(videos))

                    for vid in videos:
                        # crawl video metadata
                        try:
                            video_data = list_video_metadata(youtube, vid)
                        except errors.HttpError, e:
                            print 'Video metadata crawler: An HTTP error {0} occurred when crawl {1}:\n{2}'.format(e.resp.status, vid, e.content)

                        # crawl historical data
                        flag, csvstring = request(opener, vid, cookie, postdata)
                        # fail over 5 times, pass
                        cnt = 0
                        while not flag:
                            if cnt > 5:
                                print ('Video historical crawler: : {0} failed over 5 times\n'.format(vid))
                                break
                            flag, csvstring = request(opener, vid, cookie, postdata)
                            cnt += 1

                        if csvstring is not None:
                            startdate, dailyview, totalview, dailyshare, totalshare, dailywatch, avgwatch, dailysubscriber, totalsubscriber = csvstring.split()
                            video_data['statistics'] = {}
                            video_data['statistics']['startDate'] = startdate
                            video_data['statistics']['dailyView'] = dailyview
                            video_data['statistics']['totalView'] = totalview
                            video_data['statistics']['dailyShare'] = dailyshare
                            video_data['statistics']['totalShare'] = totalshare
                            video_data['statistics']['dailyWatch'] = dailywatch
                            video_data['statistics']['avgWatch'] = avgwatch
                            video_data['statistics']['dailySubscriber'] = dailysubscriber
                            video_data['statistics']['totalSubscriber'] = totalsubscriber
                        else:
                            print ('Video historical crawler: : {0} failed to crawl historical data\n'.format(vid))

                        output.write('{0}\n'.format(json.dumps(video_data)))
                except errors.HttpError, e:
                    print 'Channel videos crawler: An HTTP error {0} occurred when crawl {1}:\n{2}'.format(e.resp.status, cid, e.content)
