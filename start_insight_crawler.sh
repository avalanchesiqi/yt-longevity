#!/bin/bash
# usage: ./start_insight_crawler.sh $idx

# get hostname
host=$(echo $(hostname) | cut -d. -f1)

filename='/mnt/data/metadata_mar_2017/'$host'-meta'$1'.json'

# start new job
cd /home/ec2-user/yt-longevity/
python /home/ec2-user/yt-longevity/yt_longevity/extract_vids.py $filename /home/ec2-user/yt-longevity/vids-$1
mkdir /mnt/data/insightdata_mar_2017/$host-daily$1
nohup python -u /home/ec2-user/yt-longevity/run.py -f dailydata -i /home/ec2-user/yt-longevity/vids-$1 -o /mnt/data/insightdata_mar_2017/$host-daily$1/ > /home/ec2-user/yt-longevity/log/insightdata.log
