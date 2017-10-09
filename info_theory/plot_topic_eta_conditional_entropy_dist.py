from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def safe_log2(x):
    if x == 0:
        return 0
    else:
        return np.log2(x)


bin_gap = 0.05
bin_num = int(1 / bin_gap)


def get_conditional_entropy(topic_eta):
    # calculate condition entropy when topic appears
    binned_topic_eta = {i: 0 for i in range(bin_num)}
    for eta in topic_eta:
        binned_topic_eta[min(int(eta / bin_gap), bin_num - 1)] += 1
    # n = np.sum(engagement_col)

    # p_x1 = len(topic_eta) / n
    p_Y_given_x1 = [binned_topic_eta[i] / len(topic_eta) for i in range(bin_num)]
    return -np.sum([p * safe_log2(p) for p in p_Y_given_x1])


def get_conditional_entropy2(binned_topic_eta):
    # p_x1 = len(topic_eta) / n
    n = np.sum(binned_topic_eta)
    p_Y_given_x1 = [i / n for i in binned_topic_eta]
    return -np.sum([p * safe_log2(p) for p in p_Y_given_x1])


def rescale(x, pos):
    'The two args are the value and tick position'
    return '%1.1f' % (x/20)
formatter = FuncFormatter(rescale)

target1 = [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 3, 3, 3, 6, 5, 6, 17, 42, 121, 1274]
topic1 = 'Esther Hicks'
n1 = np.sum(target1)
prob_target1 = [x/n1 for x in target1]

target2 = [2819, 853, 394, 214, 118, 74, 53, 27, 18, 17, 10, 4, 10, 6, 1, 3, 1, 1, 1, 16]
topic2 = 'Android application package'
n2 = np.sum(target2)
prob_target2 = [x/n2 for x in target2]

target3 = [232, 251, 245, 252, 231, 227, 242, 246, 234, 268, 266, 244, 225, 287, 241, 247, 227, 257, 209, 230]
topic3 = 'Final Fantasy'
n3 = np.sum(target3)
prob_target3 = [x/n3 for x in target3]

fig = plt.figure(figsize=(8, 6))
width = 1/3
ind = np.arange(20)
ax1 = fig.add_subplot(111)
cornflower_blue = '#6495ed'
tomato = '#ff6347'
sea_green = '#2e8b57'

ax1.bar(ind+width*5/2, prob_target1, width, edgecolor='k', color=tomato, label='{0}'.format(topic1))
ax1.bar(ind+width/2, prob_target2, width, edgecolor='k', color=cornflower_blue, label='{0}'.format(topic2))
ax1.bar(ind+width*3/2, prob_target3, width, edgecolor='k', color=sea_green, label='{0}'.format(topic3))

ax1.set_xlim([0, 21])
ax1.set_ylim([0, 1])
ax1.set_xticks([0, 4, 8, 12, 16, 20])
ax1.xaxis.set_major_formatter(formatter)
ax1.set_xlabel('Relative engagement $\eta$', fontsize=16)
ax1.set_ylabel('Conditional probability $P(Y|X)$', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_title('$H(Y|X=1)$; Y=$\eta$, X=topic occurs', fontsize=18)
ax1.legend(loc='upper left', fontsize=14, frameon=False)

plt.tight_layout()
plt.show()
