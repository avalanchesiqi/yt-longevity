# -*- coding: utf-8 -*-

"""The xmlparser class parsing the xml response

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import datetime
from xml.etree import ElementTree
import json


def parsexml(s):
    tree = ElementTree.fromstring(s)
    graphdata = tree.find('graph_data')

    if graphdata is None:
        raise Exception("can not find data in the xml response")

    jsondata = json.loads(graphdata.text)

    # try parse daily viewcount
    try:
        dailyviews = jsondata['views']['daily']['data']
    except KeyError:
        raise Exception("can not get viewcount in the xml response")

    # get start date
    startdate = datetime.date(1970, 1, 1) + datetime.timedelta(jsondata['day']['data'][0]/86400000)
    startdate = startdate.strftime("%Y-%m-%d")

    # get total views
    totalview = jsondata['views']['cumulative']['data'][-1]

    # get daily sharecount
    dailyshares = jsondata['shares']['daily']['data']

    # get total shares
    totalshare = jsondata['shares']['cumulative']['data'][-1]

    # get daily watchtime
    dailywatches = jsondata['watch-time']['daily']['data']

    # get avg watchtime
    avgwatch = 1.0*jsondata['watch-time']['cumulative']['data'][-1]/totalview

    # get daily subscribercount
    dailysubscribers = jsondata['subscribers']['daily']['data']

    # get total subscribers
    totalsubscriber = jsondata['subscribers']['cumulative']['data'][-1]

    return {
        'startdate': startdate,
        'dailyviews': dailyviews,
        'totalview': totalview,
        'dailyshares': dailyshares,
        'totalshare': totalshare,
        'dailywatches': dailywatches,
        'avgwatch': avgwatch,
        'dailysubscribers': dailysubscribers,
        'totalsubscriber': totalsubscriber,
    }
