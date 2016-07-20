# -*- coding: utf-8 -*-

"""
Extract YouTube video id from Tweet's expanded_url field

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import os
from multiprocessing import Process
import bz2
import json
import cPickle as pickle

from yt_longevity.extractor import Extractor
from yt_longevity.exceptions import InvalidVideoIdError
from yt_longevity.extractor.helper import YTDict


class VideoIdExtractor(Extractor):
    """
    YouTube video id Extractor Class.

    - input is a directory that contains all tweet bz2 files
    - output is a directory that contains daily viewcounts and video id list

    For a tweet, the dictionaries must include the following fields:

    id:             The integer representation of the unique identifier for this Tweet.
    ******
    entities:       Entities provide structured data from Tweets including resolved URLs, media, hashtags
                    and mentions without having to parse the text to extract that information.
                    ******
                    # We only care about urls information at this moment.
                    urls:       Optional. The URL of the video file
                                Potential fields:
                                * url           The t.co URL that was extracted from the Tweet text
                                * expanded_url	The resolved URL
                                * display_url	Not a valid URL but a string to display instead of the URL
                                * indices	    The character positions the URL was extracted from
                    ******
    ******
    """

    def __init__(self, input_dir, out_dir):
        super(Extractor, self).__init__()
        self.set_input_dir(input_dir)
        self.set_output_dir(out_dir)
        self.video_ids_path = "{0}/{1}".format(out_dir, "video_ids")
        self.video_stats_path = "{0}/{1}".format(out_dir, "video_stats")
        self._mkdirs(self.video_ids_path)
        self._mkdirs(self.video_stats_path)

    @staticmethod
    def _mkdirs(filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    def extract(self):
        self._extract()

    def _extract(self):
        print "\nStart extracting video ids from tweet bz2 files...\n"

        processes = list()

        cnt = 0
        for subdir, _, files in os.walk(self.input_dir):
            for f in files:
                filepath = os.path.join(subdir, f)
                p = Process(target=self._extract_vid, args=(filepath,))
                processes.append(p)
                p.start()
                cnt += 1
                print "Process {0} starts...".format(cnt)

        for proc in processes:
            proc.join()

    @staticmethod
    def _extract_single_vid(tweet):
        if 'entities' not in tweet.keys():
            raise InvalidVideoIdError('No entities in tweet')
        urls = tweet['entities']['urls']
        if len(urls) == 0:
            raise InvalidVideoIdError('No urls in tweet')
        expanded_url = urls[0]['expanded_url']
        if 'watch?' in expanded_url and 'v=' in expanded_url:
            vid = expanded_url.split('v=')[1][:11]
        elif 'youtu.be' in expanded_url:
            vid = expanded_url.rsplit('/', 1)[-1][:11]
        else:
            raise InvalidVideoIdError('Invalid video id')

        def check_vid(vid2):
            if len(vid2) == 11:
                try:
                    vid2.decode('utf-8').encode('ascii')
                    return vid2
                except:
                    raise InvalidVideoIdError('Video id coding issue')
            else:
                raise InvalidVideoIdError('Video id length not equals to 11')

        return check_vid(vid)

    def _extract_vid(self, filepath):
        datafile = bz2.BZ2File(filepath, mode='r')
        filename = os.path.basename(os.path.normpath(filepath)).split(".")[0]
        yt_dict = YTDict()
        for line in datafile:
            if line.rstrip():
                tweet = json.loads(line)
                try:
                    vid = self._extract_single_vid(tweet)
                except:
                    continue
                yt_dict.update_tc(vid)
        pickle.dump(yt_dict.getter(), open('{0}/{1}.p'.format(self.video_stats_path, filename), 'wb'))

        with open('{0}/{1}.txt'.format(self.video_ids_path, filename), 'wb') as vids:
            for vid in yt_dict.keys():
                vids.write('{0}\n'.format(vid))

        print "\nFinish extracting video ids from {0}".format(filename)
        print 'Number of distinct video ids: {0}'.format(len(yt_dict))
