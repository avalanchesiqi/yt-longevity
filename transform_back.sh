#!/bin/bash

idxfile=/home/ec2-user/yt-longevity/conf/idx.txt

# get hostname
host=$(echo $(hostname) | cut -d. -f1)

# kill old job if not finish
procnum=`ps -ef | grep "run.py" | grep -v grep | wc -l`
if [ $procnum -eq 1 ]; then
    kill $(ps -ef | grep "run.py" | grep -v grep | awk '{print $2}')
    let oldidx=$(cat $idxfile)
    oldfilename='/mnt/data/'$host'-meta'$oldidx'.json'
    tmpfilename='/mnt/data/'$host'-metatmp.json'
    head -n -1 $oldfilename > $tmpfilename
    mv $tmpfilename $oldfilename
    let idx=$(($oldidx + 1))
    echo $idx > $idxfile
fi


# get last output filename
transidx=$(($(cat $idxfile) - 1))
filename='/mnt/data/'$host'-meta'$transidx'.json'


# sleep random second to prevent peer connection reset
sleep $((RANDOM % 10))


# start transform
scp -i /home/ec2-user/yt-longevity/conf/siqwu.key $filename ec2-user@130.56.249.25:/home/ec2-user/data/metadata/


# start new job
nohup python /home/ec2-user/yt-longevity/run.py &

