#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from datetime import datetime
from apiclient import discovery

API_SERVICE = 'youtube'
API_VERSION = 'v3'
DEVELOPER_KEY = 'AIzaSyBxNpscfnZ5-we_4-PfGEB4LIadRYOjs-M'


def relevant_videos(vid, pageToken=None):
    """Finds the relevant videos about a videoId from youtube and returns the JSON object associated with it."""

    youtube = discovery.build(API_SERVICE, API_VERSION, developerKey=DEVELOPER_KEY)

    # Call the videos().list method to retrieve results matching the specified video term.
    try:
        search_response = youtube.search().list(part="snippet", relatedToVideoId=vid, type='video', maxResults=50, pageToken=pageToken).execute()
    except Exception as e:
        print 'request failed'
        print str(e)

    output = open('relevant_videos2.txt', 'a+')

    relevant_videos_ = search_response["items"]
    for video in relevant_videos_:
        try:
            id = video['id']['videoId']
            title = video['snippet']['title'].encode('utf-8')
            output.write('{0} +++ {1}\n'.format(id, title))
        except Exception as e:
            print video
            print str(e)

    # output.write('end time {0}'.format(datetime.now()))
    # output.write('------------------\n')
    output.close()

    if 'nextPageToken' in search_response:
        relevant_videos(vid, pageToken=search_response['nextPageToken'])


if __name__ == '__main__':
    relevant_videos('wSBXfzgqHtE')

