#!/usr/bin/python

# a typical run is like
# ./yt_crawl_metadata.py <BZ2-FILE-WITH-YT-IDS> <ON-SCREEN?>
# <ON-SCREEN?> - t, true, 1, y -- will result in output of the JSON on the screen, as opposed to rotated files. Default True
# A more complex invocation:
#   ./yt_crawl_metadata.py videoIdFile.txt.bz2 | bzip2 > yt-metadata.json.bz2
# will create the file yt-metadata.json.bz2.
# Another:
#   ./yt_crawl_metadata.py videoIdFile.txt.bz2 | mongoimport --port=29997 -d twitter -c YT_metadata
# will directly import the result into MongoDB
#
# install Google API client with:
#   sudo apt-get install python-pip ; sudo pip install -I google-api-python-client requests

from Queue import Queue
from apiclient import discovery
from apiclient import errors
from oauth2client.tools import argparser
from threading import Thread
import time
import logging
import json
import bz2
import requests

# Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
# tab of https://cloud.google.com/console
# Please ensure that you have enabled the YouTube Data API for your project.
import sys

DEVELOPER_KEY = "AIzaSyBxNpscfnZ5-we_4-PfGEB4LIadRYOjs-M"
# DEVELOPER_KEY = "AIzaSyB9J-F6f5Ley261IxiVLVJsEGaQP94aa3Q"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Reading all values from options file to ensure we read only required fields
# responseParts: the parts that you want to extract as per YouTube Data API
# responseFields: the individual fields to be extracted from parts
config = open("options.txt", "r")
lines = config.readlines()
responseParts = lines[1].strip()
responseFields = lines[3].strip()
config.close()
num_workers = 10

logging.basicConfig(filename='../../datasets/metadata/request.log', level=logging.WARNING)


def retrieve_categories(country_code="US"):
    """
    Populate the categories mapping between ID and title
    """
    global categories
    # Get the feed
    r = requests.get("https://www.googleapis.com/youtube/v3/videoCategories?part=snippet&regionCode=" + country_code
                     + "&key=" + DEVELOPER_KEY)

    # Convert it to a Python dictionary
    data = json.loads(r.text)

    # Loop through the result.
    for item in data['items']:
        categories[item[u'id']] = item[u'snippet'][u'title']


def youtube_search(videoId):
    """Finds the metadata about a specifies videoId from youtube and returns the JSON object associated with it.
     :param videoId : videoId to look for
    """
    start_time = time.time()
    youtube = discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

    # Call the videos().list method to retrieve results matching the specified video term.
    try:
        video_data = youtube.videos().list(
            part=responseParts,
            id=videoId.encode('utf-8'),
            fields=responseFields
        ).execute()
    except Exception as e:
        logging.error('Request for %s request number failed with error %s at invocation.' % (videoId, e.message))
        video_data = {"items": []}

    # Creating the required json format as per mongodb structure
    jsondoc = {"YoutubeID": videoId}
    inner_doc = {}

    # Check to get empty responses handled properly
    try:
        if len(video_data["items"]) == 0:
            inner_doc["metadata"] = {}
            logging.warning('Request for %s request number was empty.' % (videoId))
        else:
            # transforming date into MongoDB format
            video_data["items"][0]["snippet"]["publishedAt"] = {
                "$date": video_data["items"][0]["snippet"]["publishedAt"]}
            # add the category title
            if video_data["items"][0]["snippet"]["categoryId"] in categories.keys():
                video_data["items"][0]["snippet"]["categoryTitle"] = \
                    categories[video_data["items"][0]["snippet"]["categoryId"]]
            else:
                logging.error(
                    'Unknown category id %s! Not in our list of categories. Try calling the function retrieve_categories with other country codes' %
                    (video_data["items"][0]["snippet"]["categoryId"]))

            inner_doc["metadata"] = video_data["items"][0]
    except Exception as e:
        logging.error('Request for %s request number failed with error %s while processing.' % \
                      (videoId, e.message))
        inner_doc["metadata"] = {}
    jsondoc["value"] = inner_doc

    logging.warning(('Request for %s request number took %.05f sec.' % (videoId, time.time() - start_time)))
    return jsondoc


class KeyDoneManager(object):
    """record the crawling status, the keys that were done
    """

    def __init__(self):
        """

        Arguments:
        """
        # open done key file
        self._done_file = open("../../datasets/metadata/key.metadata.done", "a+")
        # load the keys that were already done
        self._key_done = set([x.rstrip('\n') for x in self._done_file])

    def set_done(self, key):
        """ set a key to being done. This means writing in the internal set and the logger file

        Arguments:
        - `key`: the hey to mark as done
        """
        if key not in self._key_done:
            self._key_done.add(key)
            self._done_file.write("{}\n".format(key))
            self._done_file.flush()

    def is_done(self, key):
        """
            Verifies if a key was already done
        :param key: YoutubeID to be verified
        :return: true if it was already crawled
        """
        return key in self._key_done

    def shutdown(self):
        self._done_file.close()


if __name__ == '__main__':

    #TODO - get a parameter and choose the key based on that. Something like:
    #   ./yt_crawl_metadata.py FILE 2

    # where do we print? default to the screen
    if len(sys.argv) < 3:
        print_on_screen = True
    else:
        print_on_screen = (sys.argv[2].lower() in ("yes", "true", "t", "1", "y"))

    if not print_on_screen:
        logging.warning('**> Outputting result to files')
    else:
        logging.warning('**> Outputting result to screen.')

    # populate categories
    categories = {}
    # retrieve_categories(country_code="AU")
    retrieve_categories(country_code="US")

    # define the two queues: one for working jobs, one for results.
    to_process = Queue()
    to_write = Queue()

    # setup the done manager
    mng = KeyDoneManager()

    datapath = sys.argv[1]

    def writer():
        """Function to take values from the output queue and write it to a file
        We roll file after every 1k entries
        """
        i = 9
        output_file = "../../datasets/metadata/videoMetadata_{0}.json"
        if not print_on_screen:
            video_metadata = open(output_file.format(i), "w")
        j = 0
        while True:
            try:
                jobj = to_write.get()
                # write the current ID as done
                if jobj != 0:
                    mng.set_done(jobj["YoutubeID"])

                if print_on_screen:
                    # check for file termination object
                    if jobj != 0:
                        print json.dumps(jobj)
                else:  # dump output in files
                    # check for file termination object
                    if jobj == 0:
                        video_metadata.close()
                        to_write.task_done()
                        continue
                    elif jobj is not None:
                        video_metadata.write("{}\n".format(json.dumps(jobj)))  # , sort_keys=True

                    # handle the rolling of the output file, if needed
                    j += 1
                    if j == 100000:
                        video_metadata.close()
                        video_metadata = open(output_file.format(i + 1), "w")
                        j = 0
                        i += 1
            except Exception as e:
                logging.error('[Output Queue] Error in writing: %s.' % e.message)
                # in any case, mark the current item as done
                to_write.task_done()
                continue

            # in any case, mark the current item as done
            to_write.task_done()

    # action executed by the worked threads
    # they input lines from the working queue, process them, obtain the JSON object and put it into the writing queue
    def worker():
        while True:
            # make sure that our workers don't die on us :)
            try:
                vid = to_process.get()
                # process the file only if it was not already done
                if not mng.is_done(vid):
                    jobj = youtube_search(videoId=vid)
                    to_write.put(jobj)
                to_process.task_done()
            except Exception as e:
                logging.error('[Input Queue] Error in writing: %s.' % e.message)
                continue

    # start the working threads - 4 of them
    for i in range(num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

    # start the writer thread
    w = Thread(target=writer)
    w.daemon = True
    w.start()

    # all is good, start the work
    # opening bz2 file and reading the video id file to retrieve data
    # datafile = bz2.BZ2File(sys.argv[1], mode='r')
    datafile = open(sys.argv[1], mode='r')
    initial_time = time.time()
    for line in datafile:
        videoId = line.strip()
        try:
            # vid = videoId.strip('"').encode('utf-8')
            vid = videoId.split()[0]
        except Exception as e:
            logging.error("An invalid entry for video id:%s" % (videoId))
        to_process.put(vid)
    datafile.close()

    # wait for jobs to be done
    to_process.join()
    # give the termination object and wait for termination
    to_write.put(0)
    to_write.join()
    # close done file
    mng.shutdown()

    logging.warning('Total time for requests %.05f sec.' % (time.time() - initial_time))
    sys.exit(1)
