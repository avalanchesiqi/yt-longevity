import os
import json
from collections import defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import operator


file_loc = '../../data/vevo'
file_loc2 = '../../data/vevo_channel_statistics'

def add_artist_dailyview():
    for subdir, _, files in os.walk(file_loc):
        for f in files:
            filepath = os.path.join(subdir, f)
            popularity_dict = defaultdict(int)
            with open(filepath, 'r') as filedata:
                for line in filedata:
                    if line.rstrip():
                        video = json.loads(line.rstrip())
                        channel_id = video['snippet']['channelId']
                        try:
                            start_date = datetime(*map(int, video['insights']['startDate'].split('-')))
                            dailyviews = map(int, video['insights']['dailyView'].split(','))
                            for i in xrange(len(dailyviews)):
                                popularity_dict[start_date + timedelta(days=i)] += dailyviews[i]
                        except:
                            continue

            record_date = sorted(popularity_dict.keys())
            artist_dailyviews = [popularity_dict[d] for d in record_date]

            with open(os.path.join(file_loc2, channel_id), 'r+') as output:
                artist = json.loads(output.readline().rstrip())
                artist['statistics']['startDate'] = record_date[0].strftime("%Y-%m-%d %H:%M:%S")
                artist['statistics']['dailyView'] = ','.join(map(str, artist_dailyviews))
                artist['statistics']['totalView'] = sum(artist_dailyviews)
                output.seek(0)
                output.write(json.dumps(artist))
                output.truncate()

if __name__ == '__main__':
    # add_artist_dailyview()

    popularity_dailyviews = defaultdict(int)
    popularity_artists = {}

    for subdir, _, files in os.walk(file_loc2):
        for f in files:
            filepath = os.path.join(subdir, f)
            with open(filepath, 'r') as filedata:
                artist = json.loads(filedata.readline().rstrip())
                start_date = datetime.strptime(artist['statistics']['startDate'], "%Y-%m-%d %H:%M:%S")
                dailyviews = map(int, artist['statistics']['dailyView'].split(','))
                channel_title = artist['snippet']['title']
                n = len(dailyviews)
                for i in xrange(n):
                    target_date = start_date+timedelta(days=i)
                    if dailyviews[i] > popularity_dailyviews[target_date]:
                        popularity_dailyviews[target_date] = dailyviews[i]
                        popularity_artists[target_date] = channel_title

    fig, ax1 = plt.subplots(1, 1)

    m = (sorted(popularity_dailyviews.keys())[-1]-datetime(2010, 1, 1)).days
    x_axis = [datetime(2010, 1, 1) + timedelta(days=i) for i in xrange(m)]
    y_axis = [popularity_dailyviews[k] for k in x_axis]

    # ax1.plot_date(x_axis, y_axis, '-', ms=0, c=[0.25, 0.25, 0.25], linewidth=1)

    cs_map = {'JustinBieberVEVO': 'b^', 'RihannaVEVO': 'g+', 'TaylorSwiftVEVO': 'rx', 'KatyPerryVEVO': 'cD'}
    labels = {'JustinBieberVEVO': 'JustinBieberVEVO, 604', 'RihannaVEVO': 'RihannaVEVO, 429', 'TaylorSwiftVEVO': 'TaylorSwiftVEVO, 359',
              'KatyPerryVEVO': 'KatyPerryVEVO, 212', 'OtherVEVOs': 'OtherVEVOs, 901'}
    for channel in cs_map.keys():
        cs = cs_map[channel]
        xx_axis = [x for x in x_axis if popularity_artists[x]==channel]
        yy_axis = [popularity_dailyviews[x] for x in xx_axis]
        ax1.scatter(xx_axis, yy_axis, c=cs[0], marker=cs[1], edgecolors='none', label=labels[channel])

    xx_axis = [x for x in x_axis if popularity_artists[x] not in cs_map.keys()]
    yy_axis = [popularity_dailyviews[x] for x in xx_axis]
    ax1.scatter(xx_axis, yy_axis, c='m', marker='o', edgecolors='none', label=labels['OtherVEVOs'])

    ax1.set_xlabel('Calender date')
    ax1.set_ylabel('Daily most views')
    ax1.set_title('Figure 1: Daily most-viewed artist')

    print 'num of days: {0}'.format(m)
    artist_popularity_count = defaultdict(int)
    for x in x_axis:
        artist_popularity_count[popularity_artists[x]] += 1

    sorted_x = sorted(artist_popularity_count.items(), key=operator.itemgetter(1), reverse=True)
    for x in sorted_x:
        print x

    ax1.legend(loc='upper left')
    plt.show()
