# -*- coding: utf-8 -*-

"""
Extract YouTube video Id from Tweet's expanded_url field

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

from exceptions import InvalidVideoIdError


class VideoIdExtractor(object):
    """
    YouTube video id Extractor Class.

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

    def __init__(self, tweet=None):
        """Constructor. Receives an optional tweet."""
        self._tweet = tweet

    def extract(self):
        if 'entities' not in self._tweet.keys():
            raise InvalidVideoIdError('No entities in tweet')
        urls = self._tweet['entities']['urls']
        if len(urls) == 0:
            raise InvalidVideoIdError('No urls in tweet')
        expanded_url = urls[0]['expanded_url']
        if 'watch?' in expanded_url and 'v=' in expanded_url:
            vid = expanded_url.split('v=')[1][:11]
        elif 'youtu.be' in expanded_url:
            vid = expanded_url.rsplit('/', 1)[-1][:11]
        else:
            raise InvalidVideoIdError('Invalid video id')

        def check_vid(_vid):
            if len(_vid) == 11:
                try:
                    _vid.decode('utf-8').encode('ascii')
                    return _vid
                except:
                    raise InvalidVideoIdError('Video id coding issue')
            else:
                raise InvalidVideoIdError('Video id length not equals to 11')

        return check_vid(vid)
