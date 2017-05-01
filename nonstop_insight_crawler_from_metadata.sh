#!/bin/bash
# usage: nohup ./nonstop_insight_crawler_from_metadata.sh

# get idx
# create idx.txt with value -1 to initialise
# iterate input file number 0 to 16
idxfile=/home/ec2-user/yt-longevity/conf/idx.txt
idx=$(cat $idxfile)

while [ $idx -lt 17 ]; do
    procnum=`ps -ef | grep "run.py" | grep "dailydata" | grep -v grep | wc -l`
    if [ $procnum -eq 0 ]; then
        let idx=$(($idx + 1))
        echo $idx > $idxfile
        /bin/bash /home/ec2-user/yt-longevity/start_insight_crawler.sh $idx
    fi
done
