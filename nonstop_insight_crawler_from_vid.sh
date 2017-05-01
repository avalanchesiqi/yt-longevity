#!/bin/bash
# usage: nohup ./nonstop_insight_crawler_from_vid.sh

# get hostname
host=$(echo $(hostname) | cut -d. -f1)

# get idx
# create idx.txt with value 16 to initialise
# iterate input file number 17 to 34
idxfile=/home/ec2-user/yt-longevity/conf/idx.txt
idx=$(cat $idxfile)

while [ $idx -lt 35 ]; do
    procnum=`ps -ef | grep "run.py" | grep "dailydata" | grep -v grep | wc -l`
    if [ $procnum -eq 0 ]; then
        let idx=$(($idx + 1))
        echo $idx > $idxfile
        mkdir /mnt/data/insightdata_mar_2017/$host-daily$idx
        nohup python -u /home/ec2-user/yt-longevity/run.py -f dailydata -i /home/ec2-user/yt-longevity/input/$host-$idx.txt -o /mnt/data/insightdata_mar_2017/$host-daily$idx/ >> /home/ec2-user/yt-longevity/log/insightdata.log
    fi
done
