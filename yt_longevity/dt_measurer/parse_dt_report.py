#!/usr/bin/python

# Usage example:
# python parse_dt_report.py --input='<input_file>' --output='<output_file>'

import os
import argparse
import re

BASE_DIR = '../../'


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file path of video ids, relative to base dir', required=True)
    parser.add_argument('-o', '--output', help='output file path of video data, relative to base dir', required=True)
    args = parser.parse_args()

    input_path = os.path.join(BASE_DIR, args.input)
    output_path = os.path.join(BASE_DIR, args.output)

    # remove output file if exists
    try:
        os.remove(output_path)
    except OSError:
        pass

    # Sample format, may contain multiple tweet_url
    #
    # Unit Metadata:
    #  username
    #  StreetPrezFanz
    #  id
    #  tag:search.twitter.com,2005:809271132603957256
    #  tweet_url
    #  https://www.youtube.com/watch?v=ASCsWL4FgAk&feature=youtu.be&a
    #  posted_time
    #  12/15/2016 05:37:28
    # Unit Codes:
    # Unit Classifications:
    # I added a video to a @YouTube playlist https://t.co/4f0CIXk5v2 11 Wake Up W Big Boy
    to_write = False
    category = None
    output_data = open(output_path, 'a+')
    with open(input_path, 'r') as input_data:
        for line in input_data:
            line = line.strip()
            if line:
                if line.startswith('username'):
                    username = line.split()[1]
                    output_data.write('{0}: {1}\n'.format('username', username))
                elif line.startswith('id'):
                    id = line.rsplit(':', 1)[1]
                    output_data.write('{0}: {1}\n'.format('id', id))
                elif line.startswith('tweet_url'):
                    if 'watch?' in line and 'v=' in line:
                        vid = line.split('v=')[1][:11]
                    elif 'youtu.be' in line:
                        vid = line.rsplit('/', 1)[-1][:11]
                    else:
                        vid = None
                    if vid:
                        valid = re.match('^[\w-]+$', vid) is not None
                        if valid and len(vid) == 11:
                            output_data.write('{0}: {1}\n'.format('video_id', vid))
                elif line.startswith('posted_time'):
                    posted_time = line.split(' ', 1)[1]
                    output_data.write('{0}: {1}\n'.format('posted_time', posted_time))
                    output_data.write('----------\n')
    output_data.close()
