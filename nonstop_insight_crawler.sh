#!/bin/bash
# usage: ./nonstop_insight_crawler.sh

# get idx
# create idx.txt with value 16 to initialise
# iterate input file number 17 to 33
idxfile=/home/ec2-user/yt-longevity/conf/idx.txt
idx=$(cat $idxfile)

while [ $idx -lt 34 ]; do
    procnum=`ps -ef | grep "run.py" | grep "dailydata" | grep -v grep | wc -l`
    if [ $procnum -eq 0 ]; then
        let idx=$(($idx + 1))
        echo $idx > $idxfile
        /bin/bash /home/ec2-user/yt-longevity/start_insight_crawler.sh $idx
    fi
done
