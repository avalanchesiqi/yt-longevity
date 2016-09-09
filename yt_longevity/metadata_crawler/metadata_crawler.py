# -*- coding: utf-8 -*-

"""Metadata crawler bases on YouTube API V3 crawler.

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import sys
import os
import json
import requests
import time
from Queue import Queue
from threading import Thread
import thread
from apiclient import discovery

from yt_longevity.metadata_crawler import APIV3Crawler


class MetadataCrawler(APIV3Crawler):
    """Metadata crawler bases on YouTube API V3 crawler."""

    def __init__(self):
        APIV3Crawler.__init__(self)
        self._setup_logger('metadatacrawler')

        # Reading all values from options file to ensure we read only required fields
        # responseParts: the parts that you want to extract as per YouTube Data API
        # responseFields: the individual fields to be extracted from parts
        try:
            with open('conf/options.txt', 'r') as config:
                lines = config.readlines()
                self.responseParts = lines[1].strip()
                self.responseFields = lines[3].strip()
        except:
            self.logger.error('**> The options file options.txt does not exist! It should be in the conf folder.')
            sys.exit()

        # Reading developer_key from developer.key file from conf, roll over when exceed quota
        try:
            with open('conf/developer.key', 'r') as keys:
                developer_keys = keys.readlines()
                self.set_keys(developer_keys)
                self.set_key_index(0)
        except:
            self.logger.error('**> The keys file developer.key does not exist! It should be in the conf folder.')
            sys.exit()

        # Reading categories mapping from categorydict json, otherwise recrawl and log warning
        try:
            with open('conf/categorydict.json', 'r') as categorydict:
                self.category_dict = json.loads(categorydict.readline().rstrip())
        except:
            self.logger.warn('**> The mapping file categorydict.json does not exist! It should be in the conf folder. Recrawling...')
            self.category_dict = self._retrieve_categories()
            with open('conf/categorydict.json', 'w') as categorydict:
                json.dump(self.category_dict, categorydict)

    def _retrieve_categories(self, country_code="US"):
        """Populate the categories mapping between category Id and category title
        """
        r = requests.get("https://www.googleapis.com/youtube/v3/videoCategories?part=snippet&regionCode={0}&key={1}"
                         .format(country_code, self._keys[self._key_index]))

        data = json.loads(r.text)
        categories = {}
        for item in data['items']:
            categories[str(item[u'id'])] = str(item[u'snippet'][u'title'])
        return categories

    def _youtube_search(self, vid):
        """Finds the metadata about a specifies videoId from youtube and returns the JSON object associated with it.
        """
        start_time = time.time()
        if self._key_index >= len(self._keys):
            self.logger.error('Key index out of range, exit.')
            thread.interrupt_main()
        else:
            youtube = discovery.build(self._api_service, self._api_version, developerKey=self._keys[self._key_index])

            # Call the videos().list method to retrieve results matching the specified video term.
            try:
                video_data = youtube.videos().list(part=self.responseParts, id=vid.encode('utf-8'), fields=self.responseFields).execute()
            except Exception as e:
                if 'quota' in str(e):
                    self.logger.error('The request cannot be completed because quota exceeded.')
                    if self._key_index < len(self._keys):
                        self.set_key_index(self._key_index+1)
                        return self._youtube_search(vid)
                else:
                    self.logger.error('Request for {0} request number failed with error {1} at invocation.'.format(vid, str(e)))
                return

            # Check to get empty responses handled properly
            try:
                if len(video_data["items"]) == 0:
                    self.logger.warning('Request for {0} request number was empty.'.format(vid))
                    return
                else:
                    json_doc = video_data["items"][0]
            except Exception as e:
                self.logger.error('Request for {0} request number failed with error {1} while processing.'.format(vid, str(e)))
                return

            self.logger.debug(('Request for %s request number took %.05f sec.' % (vid, time.time() - start_time)))
            return json_doc

    def start(self, input_file, output_dir):
        self.logger.warning('**> Outputting result to files...')

        # If output directory not exists, create a new one
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # define the two queues: one for working jobs, one for results.
        threads = []
        to_process = Queue()
        to_write = Queue()

        # action executed by the worked threads
        # they input lines from the working queue, process them, obtain the JSON object and put it into writing queue
        def worker():
            while True:
                # make sure that our workers don't die on us :)
                try:
                    vid = to_process.get()
                    # process the file only if it was not already done
                    jobj = self._youtube_search(vid)
                    if jobj is not None:
                        to_write.put(jobj)
                    to_process.task_done()
                except Exception as e:
                    self.logger.error('[Input Queue] Error in writing: {0}.'.format(str(e)))
                    continue

        def writer():
            """Function to take values from the output queue and write it to a file
            We roll file after every 100k entries
            """
            i = 0
            output_path = "{0}/videoMetadata_{1}.json"
            video_metadata = open(output_path.format(output_dir, i), "w")
            j = 0
            while True:
                try:
                    jobj = to_write.get()
                    to_write.task_done()
                    # check for file termination object
                    if jobj == 0:
                        self.logger.warning('**> Termination object received and wait for termination...')
                        video_metadata.close()
                        break
                    elif jobj is not None:
                        video_metadata.write("{}\n".format(json.dumps(jobj)))  # , sort_keys=True

                    # handle the rolling of the output file, if needed
                    j += 1
                    if j == 100000:
                        video_metadata.close()
                        video_metadata = open(output_path.format(output_dir, i+1), "w")
                        j = 0
                        i += 1
                except Exception as e:
                    self.logger.error('[Output Queue] Error in writing: {0}.'.format(str(e)))
                    # in any case, mark the current item as done
                    to_write.task_done()
                    continue

        # start the working threads - 4 of them
        for i in range(self._num_threads):
            t = Thread(target=worker)
            t.daemon = True
            t.start()
            threads.append(t)

        # start the writer thread
        w = Thread(target=writer)
        w.daemon = True
        w.start()

        # all is good, start the work
        # opening vid file and reading the video id file to retrieve data
        with open(input_file, mode='r') as datafile:
            initial_time = time.time()
            for line in datafile:
                vid = line.strip()
                to_process.put(vid)

        # wait for jobs to be done
        for t in threads:
            t.join()
        # give the termination object and wait for termination
        to_write.put(0)
        to_write.join()

        self.logger.warning('**> Total time for requests {0:.4f} secs.'.format(time.time() - initial_time))
        sys.exit(1)
