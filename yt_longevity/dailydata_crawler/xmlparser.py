# -*- coding: utf-8 -*-

"""The xmlparser class parsing the xml response into a space separated csv format

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
    dailyviews = ','.join(map(str, dailyviews))

    # try parse daily sharecount and get total shares
    try:
        dailyshares = jsondata['shares']['daily']['data']
        totalshare = jsondata['shares']['cumulative']['data'][-1]
        dailyshares = ','.join(map(str, dailyshares))
    except:
        dailyshares = 'N'
        totalshare = 'N'

    # try parse daily watchtime and get average watchtime at the end
    try:
        dailywatches = jsondata['watch-time']['daily']['data']
        avgwatch = 1.0*jsondata['watch-time']['cumulative']['data'][-1]/totalview
        dailywatches = ','.join(map(str, dailywatches))
    except:
        dailywatches = 'N'
        avgwatch = 'N'

    # try parse daily subscribercount and get total subscribers
    try:
        dailysubscribers = jsondata['subscribers']['daily']['data']
        totalsubscriber = jsondata['subscribers']['cumulative']['data'][-1]
        dailysubscribers = ','.join(map(str, dailysubscribers))
    except:
        dailysubscribers = 'N'
        totalsubscriber = 'N'

    return '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n'\
        .format(startdate, dailyviews, totalview, dailyshares, totalshare, dailywatches, avgwatch, dailysubscribers, totalsubscriber)
