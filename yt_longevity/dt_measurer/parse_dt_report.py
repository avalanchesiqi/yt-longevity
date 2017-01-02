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

    # Sample format, may contain multiple tweet_url
    #
    # Unit Metadata:
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
            line = line.rstrip()
            if line == ' id' or line == ' tweet_url' or line == ' posted_time':
                to_write = True
                category = line.rstrip()[1:]
                continue
            if to_write:
                if category == 'id':
                    write_content = line.split(':')[2]
                elif category == 'tweet_url':
                    if 'watch?' in line and 'v=' in line:
                        vid = line.split('v=')[1][:11]
                    elif 'youtu.be' in line:
                        vid = line.rsplit('/', 1)[-1][:11]
                    else:
                        vid = None

                    if vid is None:
                        to_write = False
                        continue
                    else:
                        valid = re.match('^[\w-]+$', vid) is not None
                        if valid and len(vid) == 11:
                            write_content = vid
                elif category == 'posted_time':
                    write_content = line
                if write_content is not None:
                    output_data.write('{0}: {1}\n'.format(category, write_content))
                if category == 'posted_time':
                    output_data.write('----------\n')
                to_write = False

    output_data.close()
