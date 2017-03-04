#!/bin/bash

idxfile=/home/ec2-user/yt-longevity/conf/idx.txt

# get hostname
host=$(echo $(hostname) | cut -d. -f1)

# kill old job if not finish
procnum=`ps -ef | grep "run.py" | grep -v grep | wc -l`
if [ $procnum -eq 1 ]; then
    kill $(ps -ef | grep "run.py" | grep -v grep | awk '{print $2}')
    let oldidx=$(cat $idxfile)
    oldfilename='/mnt/data/metadata_mar_2017/'$host'-meta'$oldidx'.json'
    tmpfilename='/mnt/data/metadata_mar_2017/'$host'-metatmp.json'
    head -n -1 $oldfilename > $tmpfilename
    mv $tmpfilename $oldfilename
    let idx=$(($oldidx + 1))
    echo $idx > $idxfile
fi


# get last output filename
transidx=$(($(cat $idxfile) - 1))
filename='/mnt/data/metadata_mar_2017/'$host'-meta'$transidx'.json'


# sleep random second to prevent peer connection reset
sleep $((RANDOM % 10))


# start transform
scp -i /home/ec2-user/yt-longevity/conf/siqwu.key $filename ec2-user@130.56.249.25:/home/ec2-user/data/metadata/crawled_mar_2017/


# start new job
cd /home/ec2-user/yt-longevity/
nohup python -u /home/ec2-user/yt-longevity/run.py -f metadata -i /home/ec2-user/yt-longevity/input/ -o /mnt/data/metadata_mar_2017/ > /home/ec2-user/yt-longevity/log/metadata.log &
python /home/ec2-user/yt-longevity/yt_longevity/extract_vids.py $filename /home/ec2-user/yt-longevity/vids
nohup python -u /home/ec2-user/yt-longevity/run.py -f dailydata -i /home/ec2-user/yt-longevity/vids -o /mnt/data/insightdata_mar_2017/ > /home/ec2-user/yt-longevity/log/insightdata.log &
