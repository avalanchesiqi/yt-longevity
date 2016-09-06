# -*- coding: utf-8 -*-

"""
Extract YouTube video id from Tweet's expanded_url field

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import os
import bz2
import json
import random
from collections import defaultdict
from multiprocessing import Process, Queue

from yt_longevity.extractor import Extractor


class VideoIdExtractor(Extractor):
    """YouTube video id Extractor Class.

    :param input_dir: directory that contains all tweet bz2 files
    :param output_dir: directory that contains daily viewcounts and video id list

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

    def __init__(self, input_dir, output_dir):
        Extractor.__init__(self)
        self.set_input_dir(input_dir)
        self.set_output_dir(output_dir)
        self.video_ids_path = "{0}/{1}".format(output_dir, "video_ids")
        self.video_stats_path = "{0}/{1}".format(output_dir, "video_stats")
        self._mkdirs(self.video_ids_path)
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
            p = Process(target=self._extract_vid, args=(filequeue, sampling_ratio))
            p.daemon = True
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        self.logger.debug('**> Finish extracting video ids from tweet bz2 files.')

    @staticmethod
    def _extract_single_vid(tweet):
        if 'entities' not in tweet.keys():
            raise Exception('No entities in tweet')
        urls = tweet['entities']['urls']
        num_urls = len(urls)
        if num_urls == 0:
            raise Exception('No urls in tweet')
        ret = []
        for i in xrange(num_urls):
            expanded_url = urls[i]['expanded_url']
            if 'watch?' in expanded_url and 'v=' in expanded_url:
                vid = expanded_url.split('v=')[1][:11]
            elif 'youtu.be' in expanded_url:
                vid = expanded_url.rsplit('/', 1)[-1][:11]
            else:
                continue
            if len(vid) == 11:
                try:
                    vid.decode('utf-8').encode('ascii')
                    ret.append(vid)
                except:
                    continue
        return ret

    def _extract_vid(self, filequeue, sampling_ratio):
        while not filequeue.empty():
            filepath = filequeue.get()
            try:
                datafile = bz2.BZ2File(filepath, mode='r')
            except:
                self.logger.warn('Exists non-bz2 file {0} in dataset folder'.format(filepath))
                continue
            filename, filetype = os.path.basename(os.path.normpath(filepath)).split(".")

            ytdict = defaultdict(int)
            for line in datafile:
                try:
                    # Sampling data
                    if line.rstrip() and (sampling_ratio == 1 or random.random() < sampling_ratio):
                        tweet = json.loads(line)
                        try:
                            vids = self._extract_single_vid(tweet)
                        except:
                            continue
                        for vid in vids:
                            ytdict[vid] += 1
                except:
                    self.logger.error('EOFError: {0} ended before the logical end-of-stream was detected,'.format(filename))

            with open('{0}/{1}.txt'.format(self.video_stats_path, filename), 'wb') as stats:
                for k, v in ytdict.items():
                    stats.write('{0}\t{1}\n'.format(k, v))

            with open('{0}/{1}.txt'.format(self.video_ids_path, filename), 'wb') as vids:
                for vid in ytdict.keys():
                    vids.write('{0}\n'.format(vid))

            datafile.close()
            self.logger.debug('{0} done!'.format(filename))
