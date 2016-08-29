# -*- coding: utf-8 -*-

"""Combine YouTube video metadata with video dailydata

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import os
import json
import glob
from Queue import Queue
from threading import Thread


def combine(metadata_dir, dailydata_dir):
    to_process = Queue()
    num_threads = 10

    def worker():
        while True:
            try:
                datapath = to_process.get()
                process(datapath)
                to_process.task_done()
            except Exception as e:
                print "error happens in worker {0}".format(str(e))
                continue

    def process(datapath):
        with open(datapath, 'r') as datafile:
            for line in datafile:
                json_doc = json.loads(line.strip())
                vid = json_doc['YoutubeID']
                target_path = '{0}/{1}/{2}/{3}/{4}'.format(dailydata_dir, vid[0], vid[1], vid[2], vid)
                if os.path.isfile(target_path):
                    try:
                        category_id = json_doc['value']['metadata']['snippet']['categoryId']
                        title = json_doc['value']['metadata']['snippet']['title']
                        channel_id = json_doc['value']['metadata']['snippet']['channelId']
                        upload_date = json_doc['value']['metadata']['snippet']['publishedAt']['$date'][:10]
                        description = json_doc['value']['metadata']['snippet']['description']
                        duration = json_doc['value']['metadata']['contentDetails']['duration']
                        metadata = '\t'.join([category_id, title, description, channel_id, duration])
                        with open(target_path, 'r+') as dailydata:
                            record = dailydata.readline()
                            startdate = record.strip().split()[0]
                            if startdate == upload_date:
                                dailydata.seek(0)
                                dailydata.truncate()
                                dailydata.write(metadata+'\t'+record)
                                with open(dailydata_dir+'/../success2.txt', 'a+') as succ2:
                                    succ2.write(vid+'\n')
                    except Exception as e:
                        print "error happens in process {0}".format(str(e))

    for datapath in glob.glob(metadata_dir + "/*.json"):
        to_process.put(datapath)

    # start the working threads - 10 of them
    for i in range(num_threads):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

    to_process.join()
