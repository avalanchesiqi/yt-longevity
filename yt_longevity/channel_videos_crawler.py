#!/usr/bin/python

# Usage example:
# python channel_videos_crawler.py --input='<input_file>' --output='<output_dir> [--v3api] [--selenium]'

import sys, os, re, time, random, json, argparse, urllib2, urllib, cookielib
from apiclient import discovery
from selenium import webdriver
from bs4 import BeautifulSoup
import logging

from insights_crawler.xmlparser import parsexml


YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
logging.basicConfig(filename='../log/channel_videos_crawler.log', level=logging.WARNING)


def list_channel_statistics(yt_client, channel_id):
    """Call the API's channels.list method to list the existing channel statistics."""
    for i in xrange(0, 5):
        try:
            list_response = yt_client.channels().list(
                part='snippet, statistics',
                id=channel_id
            ).execute()
            json_doc = list_response['items'][0]
            return json_doc
        except:
            time.sleep((2 ** i) + random.random())
    return None


def list_channel_videos(yt_client, channel_id):
    """Call the API's search.list method to list the existing channel video ids."""
    for i in xrange(0, 5):
        try:
            vids = []
            search_response = yt_client.search().list(
                part='snippet',
                channelId=channel_id,
                type='video',
                order='date',
                maxResults=50,
            ).execute()
            if 'nextPageToken' in search_response:
                next_pagetoken = search_response['nextPageToken']
            else:
                next_pagetoken = None
            for search_result in search_response.get('items', []):
                vids.append(search_result['id']['videoId'])

            while next_pagetoken is not None:
                search_response = yt_client.search().list(
                    part='snippet',
                    channelId=channel_id,
                    type='video',
                    order='date',
                    maxResults=50,
                    pageToken=next_pagetoken
                ).execute()
                if 'nextPageToken' in search_response:
                    next_pagetoken = search_response['nextPageToken']
                else:
                    next_pagetoken = None
                for search_result in search_response.get('items', []):
                    vids.append(search_result['id']['videoId'])

            return vids
        except:
            time.sleep((2 ** i) + random.random())
    return []


def get_webdriver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--mute-audio')

    if sys.platform == 'win32':
        driver_path = '../conf/webdriver/chromedriver.exe'
    elif sys.platform == 'darwin':
        driver_path = '../conf/webdriver/chromedriver_mac64'
    elif sys.maxsize > 2 ** 32:
        driver_path = '../conf/webdriver/chromedriver_linux64'
    else:
        driver_path = '../conf/webdriver/chromedriver_linux32'
    driver = webdriver.Chrome(driver_path, chrome_options=chrome_options)
    return driver


def list_channel_videos_selenium(driver, channel_id):
    """Simulate a browser behavior to click button via selenium"""
    target_page = 'https://www.youtube.com/channel/{0}/videos'.format(channel_id)
    driver.get(target_page)

    # Scroll down to bottom to get all video ids
    last_height = driver.execute_script('return document.documentElement.scrollHeight')
    while True:
        # Scroll down to bottom
        driver.execute_script('window.scrollTo(0, document.documentElement.scrollHeight);')
        # Wait to load page
        time.sleep(2)
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script('return document.documentElement.scrollHeight')
        if new_height == last_height:
            break
        last_height = new_height
    time.sleep(2)

    vids = []
    soup = BeautifulSoup(driver.page_source, 'lxml')
    href_blocks = soup.find_all('a', href=True, id='video-title')
    for href_block in href_blocks:
        vids.append(href_block['href'][-11:])
    return vids


def list_video_metadata(youtube, video_id):
    """Call the API's videos.list method to list the existing video metadata."""
    for i in xrange(0, 5):
        try:
            list_response = youtube.videos().list(
                part='snippet,statistics,topicDetails,contentDetails',
                id=video_id
            ).execute()
            json_doc = list_response['items'][0]
            return json_doc
        except:
            time.sleep((2 ** i) + random.random())
    return None


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
    """Make a request to YouTube server to get insight data."""
    url = get_url(vid)
    header = get_header(cookie, vid)
    opener.addheaders = header

    time.sleep(random.uniform(0.1, 1))

    try:
        response = opener.open(url, postdata, timeout=5)
    except:
        logging.debug('Video insight crawler: {0} server is down, can not get response, retry...'.format(vid))
        return 0, None

    try:
        content = response.read()
    except:
        logging.debug('Video insight crawler: {0} response read time out, retry...'.format(vid))
        return 0, None

    try:
        csvstring = parsexml(content)
    except:
        logging.debug('Video insight crawler: {0} corrupted or empty xml, skip...'.format(vid))
        return 1, None

    return 1, csvstring


if __name__ == "__main__":
    developer_key = open('../conf/developer.key').readline().rstrip()
    youtube_client = discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=developer_key)
    opener = urllib2.build_opener()
    cookie, sessiontoken = get_cookie_and_sessiontoken()
    postdata = get_post_data(sessiontoken)

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='absolute input file path of channel ids', required=True)
    parser.add_argument('-o', '--output', help='absolute output dir path of video data', required=True)
    parser.add_argument('--selenium', dest='selenium', action='store_true', help='crawl video list via selenium')
    parser.add_argument('--v3api', dest='selenium', action='store_false', help='crawl video list via datav3 api, default')
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output
    use_selenium = args.selenium

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_path, 'r') as channel_ids:
        for cid in channel_ids:
            cid = cid.rstrip()
            try:
                if use_selenium:
                    # get video ids from selenium
                    driver = get_webdriver()
                    video_ids = list_channel_videos_selenium(driver, cid)
                    driver.quit()
                else:
                    # get video ids from YouTube Data3 API
                    video_ids = list_channel_videos(youtube_client, cid)

                if len(video_ids) == 0:
                    continue

                for vid in video_ids:
                    # crawl video metadata
                    video_data = list_video_metadata(youtube_client, vid)

                    if video_data is None:
                        continue

                    # fail over 5 times, pass
                    csvstring = None
                    for i in xrange(5):
                        flag, csvstring = request(opener, vid, cookie, postdata)
                        if flag:
                            break
                        else:
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

                        with open(os.path.join(output_dir, cid), 'a') as output:
                            output.write('{0}\n'.format(json.dumps(video_data)))
                    else:
                        logging.debug('Video insight crawler: {0} failed to crawl insight data'.format(vid))
            except Exception, e:
                logging.warning('Channel videos crawler: Error occurred when crawl {0}:\n{1}'.format(cid, str(e)))
