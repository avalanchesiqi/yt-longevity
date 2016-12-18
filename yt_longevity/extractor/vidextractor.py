# -*- coding: utf-8 -*-

"""
Extract YouTube video ids from Tweet's urls

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import os
import bz2
import re
import json
import operator
import random
from collections import defaultdict
from multiprocessing import Process, Queue

from yt_longevity.extractor import Extractor


class VideoIdExtractor(Extractor):
    """YouTube video id Extractor Class.

    :param input_dir: directory that contains all tweet bz2 files
    :param output_dir: directory that contains daily viewcounts and video id list

    For a tweet, the dictionaries must include the following fields:

    id:               The integer representation of the unique identifier for this Tweet.
    ******
    entities:         Entities provide structured data from Tweets including resolved URLs, media, hashtags
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
    retweeted_status: entities: urls: expanded_url
                      extended_tweet: entities: urls: expanded_url
    ******
    quoted_status:    entities: urls: expanded_url
    """

    def __init__(self, input_dir, output_dir):
        Extractor.__init__(self)
        self.set_input_dir(input_dir)
        self.set_output_dir(output_dir)
        self.video_stats_path = "{0}/{1}".format(output_dir, 'video_stats')
        self._mkdirs(self.video_stats_path)
        self._setup_logger('vidextractor')

    @staticmethod
    def _mkdirs(filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    def extract(self, sampling_ratio=1):
        self.logger.debug('**> Start extracting video ids from tweet bz2 files...')

        processes = []
        filequeue = Queue()

        for subdir, _, files in os.walk(self.input_dir):
            for f in files:
                if f.endswith('bz2'):
                    filepath = os.path.join(subdir, f)
                    filequeue.put(filepath)

        for w in xrange(self.proc_num):
            p = Process(target=self._extract_tweet, args=(filequeue, sampling_ratio))
            p.daemon = True
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        self.logger.debug('**> Finish extracting video ids from tweet bz2 files.')

        self.logger.debug('**> Start aggregating video ids from stats folder...')
        self._aggregate_ids()
        self.logger.debug('**> aggregating video ids from stats folder.')

    @staticmethod
    def _extract_vid_from_expanded_url(expanded_url):
        if 'watch?' in expanded_url and 'v=' in expanded_url:
            vid = expanded_url.split('v=')[1][:11]
        elif 'youtu.be' in expanded_url:
            vid = expanded_url.rsplit('/', 1)[-1][:11]
        else:
            return None
        # valid condition: contains only alphanumeric, dash or underline
        valid = re.match('^[\w-]+$', vid) is not None
        if valid and len(vid) == 11:
            return vid
        return None

    def _extract_vids(self, tweet):
        urls = []
        if 'entities' in tweet.keys() and 'urls' in tweet['entities']:
            urls.extend(tweet['entities']['urls'])
        if 'retweeted_status' in tweet.keys():
            if 'entities' in tweet['retweeted_status'] and 'urls' in tweet['retweeted_status']['entities']:
                urls.extend(tweet['retweeted_status']['entities']['urls'])
            if 'extended_tweet' in tweet['retweeted_status']:
                if 'entities' in tweet['retweeted_status']['extended_tweet'] and 'urls' in tweet['retweeted_status']['extended_tweet']['entities']:
                    urls.extend(tweet['retweeted_status']['extended_tweet']['entities']['urls'])
        if 'quoted_status' in tweet.keys():
            if 'entities' in tweet['quoted_status'] and 'urls' in tweet['quoted_status']['entities']:
                urls.extend(tweet['quoted_status']['entities']['urls'])
        expanded_urls = []
        for url in urls:
            if url['expanded_url'] is not None:
                expanded_urls.append(url['expanded_url'])

        vids = set()
        for expanded_url in expanded_urls:
            vid = self._extract_vid_from_expanded_url(expanded_url)
            if vid is not None:
                vids.add(vid)
        return vids

    def _extract_tweet(self, filequeue, sampling_ratio):
        while not filequeue.empty():
            filepath = filequeue.get()
            try:
                filedata = bz2.BZ2File(filepath, mode='r')
            except:
                self.logger.warn('Exists non-bz2 file {0} in dataset folder'.format(filepath))
                continue
            filename, filetype = os.path.basename(os.path.normpath(filepath)).split(".")

            ytdict = defaultdict(int)
            for line in filedata:
                try:
                    # Sampling data
                    if line.rstrip() and (sampling_ratio == 1 or random.random() < sampling_ratio):
                        tweet = json.loads(line)
                        vids = self._extract_vids(tweet)
                        for vid in vids:
                            ytdict[vid] += 1
                except:
                    self.logger.error('EOFError: {0} ended before the logical end-of-stream was detected,'.format(filename))

            with open('{0}/{1}.txt'.format(self.video_stats_path, filename), 'w') as stats:
                for k, v in ytdict.items():
                    stats.write('{0}\t{1}\n'.format(k, v))

            filedata.close()
            self.logger.debug('{0} done!'.format(filename))

    def _aggregate_ids(self):
        tweetcount_dict = defaultdict(int)
        for subdir, _, files in os.walk(self.video_stats_path):
            for f in files:
                if f.endswith('txt'):
                    filepath = os.path.join(subdir, f)
                    with open(filepath, 'r') as filedata:
                        for line in filedata:
                            if line.rstrip():
                                vid, tweetcount = line.rstrip().split()
                                tweetcount_dict[vid] += int(tweetcount)

        with open('{0}/vid_tweetcount.txt'.format(self.output_dir), 'w') as aggregated_ids:
            sorted_tweetcount_tuple = sorted(tweetcount_dict.items(), key=operator.itemgetter(1), reverse=True)
            for item in sorted_tweetcount_tuple:
                aggregated_ids.write('{0}\t{1}\n'.format(item[0], item[1]))
