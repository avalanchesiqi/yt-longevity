#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""calculate mutual information between two different topics, i.e, I(Adele; Music),
by constructing 2x2 oc-occurrence matrix
X        0     1
Y  0    10    15
   1    20    25

I(X;Y) = sum(P(x, y) * log( P(x, y)/P(x)/P(y) ))
"""

