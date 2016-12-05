#!/usr/bin/python

# Usage example:
# python channel_localizations.py --action='<action>' --channel_id='<channel_id>'

import json

from apiclient import discovery, errors
from oauth2client.tools import argparser

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account.
YOUTUBE_READ_WRITE_SCOPE = "https://www.googleapis.com/auth/youtube"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


# Call the API's channels.list method to list the existing channel localizations.
def list_channel_statistics(youtube, channel_id):
    results = youtube.channels().list(
        part="snippet, statistics",
        id=channel_id
    ).execute()

    json_doc = results["items"][0]
    return json_doc


if __name__ == "__main__":
    debug = False
    youtube = discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                              developerKey=open('../conf/developer.key').readline().rstrip())

    if debug:
        # The "channel_id" option specifies the ID of the selected YouTube channel.
        argparser.add_argument("--channel_id", help="ID for channel for which the statistics will be returned.")
        args = argparser.parse_args()

        if not args.channel_id:
            exit("Please specify channel id using the --channel_id= parameter.")

        try:
            print list_channel_statistics(youtube, args.channel_id)
        except errors.HttpError, e:
            print "An HTTP error %d occurred:\n%s" % (e.resp.status, e.content)
    else:
        with open('../../data/channel_ids.txt', 'r') as channel_ids:
            for cid in channel_ids:
                cid = cid.rstrip()
                with open('../../data/vevo_channel_statistics/{0}'.format(cid), 'w') as output:
                    try:
                        output.write(json.dumps(list_channel_statistics(youtube, cid)))
                    except errors.HttpError, e:
                        print "An HTTP error %d occurred:\n%s" % (e.resp.status, e.content)
