#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge YouTube video id from different file by Spark

Author: Siqi Wu
Email: Siqi.Wu@anu.edu.au
"""

import os
from pyspark import SparkContext, SparkConf

conf = SparkConf()
conf.setMaster('spark://130.56.253.56:7077')
conf.setAppName('spark-merge')
conf.set('driver-memory', '4G')
conf.set('executor-memory', '4G')
sc = SparkContext(conf=conf)

input_dir = "hdfs://130.56.253.56/video_ids"
file_content_pair = sc.wholeTextFiles(input_dir, 1000)
output_file = "hdfs://130.56.253.56:8020/aggregated_vids"
res = file_content_pair.values().flatMap(lambda s: s.split("\n")).distinct()
res.saveAsTextFile(output_file)

sc.stop()
