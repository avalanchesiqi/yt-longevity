# -*- coding: utf-8 -*-

"""
Extract YouTube video ID from Tweet's expanded_url field

Author: Siqi Wu
Date last modified: 06/07/2016
"""


class VidExtractor(object):
    """
    YouTube video ID Extractor Class.

    For a tweet, the dictionaries must include the following fields:

    created_at:     UTC time when this Tweet was created.
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

    def __init__(self, tweet=None):
        """Constructor. Receives an optional tweet."""
        self.tweet = tweet

    def extract(self):
        if 'entities' not in self.tweet.keys():
            return
        vid = ''
        urls = self.tweet['entities']['urls']
        if len(urls) == 0:
            return
        expanded_url = urls[0]['expanded_url']
        if 'watch?' in expanded_url and 'v=' in expanded_url:
            vid = expanded_url.split('v=')[1][:11]
        elif 'youtu.be' in expanded_url:
            vid = expanded_url.rsplit('/', 1)[-1][:11]
        if len(vid) == 11:
            return vid
        else:
            return
