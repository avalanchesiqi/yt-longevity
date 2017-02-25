#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Usage example:
# python plot_relevant_videos.py -f selenium -v e-ORhEE9VVg

import sys
import os
import argparse
import time
from apiclient import discovery
from selenium import webdriver
from bs4 import BeautifulSoup

API_SERVICE = 'youtube'
API_VERSION = 'v3'
DEVELOPER_KEY = 'AIzaSyBxNpscfnZ5-we_4-PfGEB4LIadRYOjs-M'
BASE_DIR = '../'


def search_relevant_videos_api(vid, pageToken=None):
    """Finds the relevant videos about a videoId from youtube and returns the JSON object associated with it."""
    youtube = discovery.build(API_SERVICE, API_VERSION, developerKey=DEVELOPER_KEY)
    next_pagetoken = None

    output = open(BASE_DIR + 'log/{0}_relevant_list.txt'.format(vid), 'a+')
    # Call the videos().list method to retrieve results matching the specified video term.
    try:
        search_response = youtube.search().list(part="snippet", relatedToVideoId=vid, type='video', maxResults=50, pageToken=pageToken).execute()
        relevant_videos = search_response["items"]
        for video in relevant_videos:
            try:
                id = video['id']['videoId']
                print video['snippet']['title']
                output.write('{0}\n'.format(id))
            except Exception as e:
                print 'parse and write failed'
                print str(e)

        if 'nextPageToken' in search_response:
            next_pagetoken = search_response['nextPageToken']
    except Exception as e:
        print 'request failed'
        print str(e)
    output.close()

    # recursively request next page
    if next_pagetoken is not None:
        search_relevant_videos_api(vid, pageToken=next_pagetoken)


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


def search_relevant_videos_selenium(video_id):
    """Simulate a browser behavior to get relevant video list via selenium."""
    target_page = 'https://www.youtube.com/watch?v={0}'.format(video_id)

    driver.delete_all_cookies()
    driver.get(target_page)

    # click show more button
    # try:
    #     showmore_btn = driver.find_element_by_xpath("//button[contains(@id,'watch-more-related-button')]")
    #     time.sleep(3)
    #     showmore_btn.click()
    #     time.sleep(3)
    # except Exception as e:
    #     print 'click show more button fail'
    #     print str(e)

    # wait till browser load
    time.sleep(3)

    vids = []
    soup = BeautifulSoup(driver.page_source, 'lxml')
    relevant_videos_wrapper = soup.find_all('div', class_='content-wrapper')
    for video_div in relevant_videos_wrapper:
        video_a = video_div.find('a')
        vids.append(video_a['href'][-11:])
        stat_view = video_a.find('span', class_='stat view-count').text
        print video_a['title']
        print stat_view
        print

    driver.close()
    with open(BASE_DIR+'log/{0}_sidebar_list.txt'.format(video_id), 'a+') as f:
        f.write(' '.join(vids))
        f.write('\n')
    return vids


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='function of conduction', required=True)
    parser.add_argument('-v', help='request video id', required=True)
    args = parser.parse_args()

    if args.f == 'selenium':
        driver = get_webdriver()
        search_relevant_videos_selenium(args.v)
        driver.quit()
    elif args.f == 'api':
        search_relevant_videos_api(args.v)
