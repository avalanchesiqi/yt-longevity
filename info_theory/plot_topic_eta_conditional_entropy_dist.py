from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def safe_log2(x):
    if x == 0:
        return 0
    else:
        return np.log2(x)


def get_conditional_entropy(engagement_col, topic_eta):
    bin_gap = 0.05
    bin_num = int(1 / bin_gap)

    # calculate condition entropy when topic appears
    binned_topic_eta = {i: 0 for i in range(bin_num)}
    for eta in topic_eta:
        binned_topic_eta[min(int(eta / bin_gap), bin_num - 1)] += 1
    n = np.sum(engagement_col)

    p_x1 = len(topic_eta) / n
    p_Y_given_x1 = [binned_topic_eta[i] / len(topic_eta) for i in range(bin_num)]
    return -p_x1 * np.sum([p * safe_log2(p) for p in p_Y_given_x1])


def rescale(x, pos):
    'The two args are the value and tick position'
    return '%1.2f' % (x/20)
formatter = FuncFormatter(rescale)

# overall = [225045, 242841, 252768, 245113, 251387, 256861, 252240, 247357, 252111, 251227, 249757, 255001, 244762, 254248, 243531, 247370, 245979, 244484, 245814, 305667]
# music = {0: 75892, 1: 79314, 2: 78985, 3: 74869, 4: 74375, 5: 73065, 6: 69923, 7: 65947, 8: 65019, 9: 62047, 10: 59424, 11: 57922, 12: 53036, 13: 53190, 14: 49020, 15: 48196, 16: 46128, 17: 44365, 18: 43538, 19: 84429}
# # Minecraft
# minecraft = {0: 13557, 1: 13473, 2: 11712, 3: 9513, 4: 8608, 5: 7750, 6: 7059, 7: 6545, 8: 6265, 9: 6129, 10: 6069, 11: 6130, 12: 5745, 13: 5909, 14: 5546, 15: 5532, 16: 5242, 17: 5080, 18: 4823, 19: 7502}
# music = [music[k] for k in range(20)]
# residual = [i-j for i, j in zip(overall, music)]

# target = {0: 1, 1: 4, 2: 7, 3: 7, 4: 12, 5: 13, 6: 16, 7: 25, 8: 23, 9: 42, 10: 39, 11: 57, 12: 62, 13: 72, 14: 92, 15: 104, 16: 104, 17: 141, 18: 174, 19: 105}
target = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 1, 7: 1, 8: 1, 9: 0, 10: 3, 11: 3, 12: 3, 13: 6, 14: 5, 15: 6, 16: 17, 17: 42, 18: 121, 19: 1274}
n = np.sum(target.values())
prob_target = [target[x] / n for x in range(20)]

# print(get_conditional_entropy(overall, music))

# overall = np.array(overall)
# music = np.array(music)
# residual = np.array(residual)
# total = np.sum(overall)

# prob_music = music/total
# prob_residual = residual/total

fig = plt.figure(figsize=(8, 6))
width = 1
ind = np.arange(20)
ax1 = fig.add_subplot(111)
cornflower_blue = '#6495ed'
tomato = '#ff6347'

# ax1.bar(ind+width/2, prob_residual, width, bottom=prob_music, edgecolor='k', color=cornflower_blue, label='Residual of Music videos')
ax1.bar(ind+width/2, prob_target, width, edgecolor='k', color=cornflower_blue, label='Probability of Traget labelled videos')

ax1.set_xlim([0, 21])
ax1.set_ylim(ymax=ax1.get_ylim()[1]+0.01)
ax1.set_xticks([0, 4, 8, 12, 16, 20])
ax1.xaxis.set_major_formatter(formatter)
ax1.set_xlabel('Relative engagement $\eta$', fontsize=16)
ax1.set_ylabel('Probability', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
# ax1.set_title('Conditional entropy between "" and $\eta$: {0:.4f}'.format(get_conditional_entropy(overall, music)), fontsize=18)
ax1.legend(loc='upper left', fontsize=14, frameon=False)

plt.tight_layout()
plt.show()
