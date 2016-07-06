# -*- coding: utf-8 -*-

"""
A parser that parses tweet information.

Author: Siqi Wu
Date last modified: 06/07/2016
"""


class TweetParser(object):
    """
    Tweet Parser Class.

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
        pass

    def get_testcases(self):
        t = getattr(self, '_TEST', None)
        if t:
            assert not hasattr(self, '_TESTS'), \
                '%s has _TEST and _TESTS' % type(self).__name__
            tests = [t]
        else:
            tests = getattr(self, '_TESTS', [])
        for t in tests:
            yield t
