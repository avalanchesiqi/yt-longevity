# -*- coding: utf-8 -*-

"""The logger class supporting the crawler

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

from datetime import datetime


class Logger(object):
    """Class to record the crawling log, warnings and result
    """
    
    def __init__(self, output_dir=""):
        self._output_dir = output_dir
        self._success_file = open(self._output_dir + '/success.txt', 'a+')
        self._fail_file = open(self._output_dir + '/fail.txt', 'a+')
        self._log_file = open(self._output_dir + '/crawler.log', 'a+')

    def log_success(self, vid):
        """Log if this vid gets valid stat successfully
        """
        self._success_file.write('{0}\n'.format(vid))
        self._success_file.flush()

    def log_fail(self, vid, fail_id):
        """Log if this vid is fail, use fail id, mapping as following,

        1 disabled
        2 notfound
        3 private
        4 nostatyet
        5 invalidrequest
        6 noviewcount
        7 toomanyfails(quota limit exceed or server down or response timeout)
        """
        self._fail_file.write('{0}\t{1}\n'.format(vid, str(fail_id)))
        self._fail_file.flush()

    def log_log(self, msg):
        """Log message during the crawling, i.e, server down, quota limit exceed, etc,
        """
        self._log_file.write('{0}\t{1}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg))
        self._log_file.flush()

    def close(self):
        """Close all log files
        """
        self._success_file.close()
        self._fail_file.close()
        self._log_file.close()
