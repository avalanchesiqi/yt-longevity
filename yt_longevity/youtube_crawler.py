#!/usr/bin/python

# Usage example:
# Change to your developer key first
# It takes an input file from assigned path and generates an output file to assigned path plus a log in current dir
# python youtube_crawler.py --input='<input_file>' --output='<output_file>'

# The program returns a file contains following fields in hierarchical json format
# * note:
# 1. insights field will not exist if insight crawler fails
# 2. unavailable items in insights field will return string 'N'
# ......................................
# id
# snippet
#     description
#     tags
#     channelId
#     publishedAt
#     channelTitle
#     title
#     categoryId
# topicDetails
#     topicIds
#     relevantTopicIds
# contentDetails
#     duration
#     definition
#     dimension
#     caption
# statistics
#     commentCount
#     viewCount
#     favoriteCount
#     dislikeCount
#     likeCount
# insights
#     startDate
#     days
#     dailyView
#     totalView
#     dailyShare
#     totalShare
#     dailyWatch
#     avgWatch
#     dailySubscriber
#     totalSubscriber


import re
import time
import random
import json
import argparse
import datetime
from xml.etree import ElementTree
import urllib2
import urllib
import cookielib
from apiclient import discovery
import logging


# useful variables, change to your developer key
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
DEVELOPER_KEY = 'YOUR_DEVELOPER_KEY'
PARTS = "snippet,statistics,topicDetails,contentDetails"
FIELDS = 'items(id,snippet(publishedAt,channelId,title,description,channelTitle,categoryId,tags),statistics,' \
         'topicDetails,contentDetails(duration,dimension,definition,caption,regionRestriction))'


def list_video_metadata(youtube, video_id):
    """Call the API's videos.list method to list the existing video metadata."""
    for i in xrange(0, 5):
        try:
            list_response = youtube.videos().list(part=PARTS, id=video_id, fields=FIELDS).execute()
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


def _get_header(cookie, vid):
    """Get the request header for historical data crawler."""
    headers = []
    headers.append(('Content-Type', 'application/x-www-form-urlencoded'))
    headers.append(('Cookie', cookie))
    headers.append(('Origin', 'https://www.youtube.com'))
    headers.append(('Referer', 'https://www.youtube.com/watch?v=' + vid))
    headers.append(('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'))
    return headers


def _parse_xml(xml_string):
    tree = ElementTree.fromstring(xml_string)
    graphdata = tree.find('graph_data')

    if graphdata is None:
        raise Exception('can not find data in the xml response')

    jsondata = json.loads(graphdata.text)

    # try parse daily viewcount
    try:
        dailyviews = jsondata['views']['daily']['data']
    except KeyError:
        raise Exception('can not get viewcount in the xml response')

    # get start date
    startdate = datetime.date(1970, 1, 1) + datetime.timedelta(jsondata['day']['data'][0]/86400000)
    startdate = startdate.strftime("%Y-%m-%d")

    # get days with stats
    days = [(d - jsondata['day']['data'][0]) / 86400000 for d in jsondata['day']['data']]
    days = ','.join(map(str, days))

    # get total views
    try:
        totalview = jsondata['views']['cumulative']['data'][-1]
    except:
        totalview = sum(dailyviews)
    dailyviews = ','.join(map(str, dailyviews))

    # try parse daily sharecount and get total shares
    try:
        dailyshares = jsondata['shares']['daily']['data']
        try:
            totalshare = jsondata['shares']['cumulative']['data'][-1]
        except:
            totalshare = sum(dailyshares)
        dailyshares = ','.join(map(str, dailyshares))
    except:
        dailyshares = 'N'
        totalshare = 'N'

    # try parse daily watchtime and get average watchtime at the end
    try:
        dailywatches = jsondata['watch-time']['daily']['data']
        try:
            avgwatch = 1.0*jsondata['watch-time']['cumulative']['data'][-1]/totalview
        except:
            avgwatch = 1.0*sum(dailywatches)/totalview
        dailywatches = ','.join(map(str, dailywatches))
    except:
        dailywatches = 'N'
        avgwatch = 'N'

    # try parse daily subscribercount and get total subscribers
    try:
        dailysubscribers = jsondata['subscribers']['daily']['data']
        try:
            totalsubscriber = jsondata['subscribers']['cumulative']['data'][-1]
        except:
            totalsubscriber = sum(dailysubscribers)
        dailysubscribers = ','.join(map(str, dailysubscribers))
    except:
        dailysubscribers = 'N'
        totalsubscriber = 'N'

    return '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n'\
        .format(startdate, days, dailyviews, totalview, dailyshares, totalshare, dailywatches, avgwatch, dailysubscribers, totalsubscriber)


def request(opener, vid, cookie, postdata):
    """Make a request to YouTube server to get historical data."""
    url = get_url(vid)
    header = _get_header(cookie, vid)
    opener.addheaders = header

    time.sleep(random.uniform(0.1, 1))

    try:
        response = opener.open(url, postdata, timeout=5)
    except:
        logging.error('Video historical crawler: {0} server is down, can not get response, retry...'.format(vid))
        return 1, None

    try:
        content = response.read()
    except:
        logging.error('Video historical crawler: {0} response read time out, retry...'.format(vid))
        return 2, None

    try:
        csvstring = _parse_xml(content)
    except:
        logging.error('Video historical crawler: {0} corrupted or empty xml, skip...'.format(vid))
        return 3, None

    return 0, csvstring


if __name__ == "__main__":
    # I/O interface, read from an input file
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file path of video ids', required=True)
    parser.add_argument('-o', '--output', help='output file path of video data', required=True)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    logging.basicConfig(filename='./all_in_one_youtube_crawler.log', level=logging.WARNING)

    developer_key = DEVELOPER_KEY
    youtube = discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=developer_key)
    opener = urllib2.build_opener()
    cookie, sessiontoken = get_cookie_and_sessiontoken()
    postdata = get_post_data(sessiontoken)

    output_data = open(output_path, 'a+')

    # read the input file, start the crawler
    with open(input_path, 'r') as input_data:
        for vid in input_data:
            vid = vid.rstrip()
            # get video metadata from YouTube Data3 API
            try:
                video_data = list_video_metadata(youtube, vid)
            except Exception, e:
                logging.error('Video metadata crawler: Error occurred when crawl {0}:\n{1}'.format(vid, str(e)))

            # get video insights data from request
            csvstring = None
            # fail over 5 times, pass
            for i in xrange(5):
                exit_code, csvstring = request(opener, vid, cookie, postdata)
                if exit_code == 0 or exit_code == 3:
                    break
                else:
                    time.sleep(2 ** i + random.random())

            if csvstring is not None:
                insights_data = csvstring.split()
                video_data['insights'] = {}
                video_data['insights']['startDate'] = insights_data[0]
                video_data['insights']['days'] = insights_data[1]
                video_data['insights']['dailyView'] = insights_data[2]
                video_data['insights']['totalView'] = insights_data[3]
                video_data['insights']['dailyShare'] = insights_data[4]
                video_data['insights']['totalShare'] = insights_data[5]
                video_data['insights']['dailyWatch'] = insights_data[6]
                video_data['insights']['avgWatch'] = insights_data[7]
                video_data['insights']['dailySubscriber'] = insights_data[8]
                video_data['insights']['totalSubscriber'] = insights_data[9]
            else:
                logging.error('Video insight crawler: {0} failed to crawl insight data'.format(vid))

            output_data.write('{0}\n'.format(json.dumps(video_data)))

            # sleep random time to rest insight_data request
            time.sleep(1+random.random())

    output_data.close()
