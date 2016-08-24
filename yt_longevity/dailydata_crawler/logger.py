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
        self._result_file = open(self._output_dir + '/result.txt', 'a+')
        self._log_file = open(self._output_dir + '/crawler.log', 'a+')
        self._warning_file = open(self._output_dir + '/warning.txt', 'a+')
        self._done_file = open(self._output_dir + '/vids_done.txt', 'a+')

    def log_result(self, vid, res):
        """Log result if daily status return successfully, '\t' separated

        format: vid upload_date total_viewcount total_sharecount    daily_viewcount daily_sharecount
        """
        self._result_file.write('{0}\t{1}\n'.format(vid, res))
        self._result_file.flush()

    def log_log(self, msg):
        """Log message during the crawling, i.e, server down, quota limit exceed, etc,
        """
        self._log_file.write('{0}\t{1}\n'.format(str(datetime.now()), msg))
        self._log_file.flush()

    def log_warn(self, vid, msg):
        """Log message as warning, i.e, invalid request, nostatyet, etc,
        """
        self._warning_file.write('{0}\t{1}\n'.format(vid, msg))
        self._warning_file.flush()

    def log_done(self, vid):
        """Log if this vid is done, i.e, success, disable, nostatyet, etc,
        """
        self._done_file.write('{0}\n'.format(vid))
        self._done_file.flush()

    def close(self):
        """Close all log files
        """
        self._result_file.close()
        self._log_file.close()
        self._warning_file.close()
        self._done_file.close()