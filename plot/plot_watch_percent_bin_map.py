#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import isodate
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

datapath = '../../data/view_wt_wp.txt'

x_axis = []
y_axis = []
with open(datapath, 'r') as f:
    for line in f:
        view, wt, wp = line.rstrip().split()
        x_axis.append(int(view))
        y_axis.append(float(wt))

fig, ax1 = plt.subplots(1, 1)

ax1.scatter(x_axis, y_axis)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(xmin=1)
ax1.set_ylim(ymin=0)
# ax1.set_ylim(ymax=1)
ax1.set_xlabel('first 8 weeks views')
ax1.set_ylabel('first 8 weeks watch time')
# ax1.set_title('News: 5 mins (20th bin)')

plt.show()
