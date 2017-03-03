# extract vid list from crawled metadata

import sys
import json

metadata_path = sys.argv[1]
to_crawl_vids_path = sys.argv[2]

to_crawl_vids_set = set()

with open(metadata_path, 'r') as f1:
    for line in f1:
        video = json.loads(line.rstrip())
        vid = video['id']
        to_crawl_vids_set.add(vid)

with open(to_crawl_vids_path, 'w') as f2:
    for vid in to_crawl_vids_set:
        f2.write('{0}\n'.format(vid))