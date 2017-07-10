import numpy as np
import matplotlib.pyplot as plt

global_errors = []
knn_errors = []
linreg_errors = []
logreg_errors = []
topic_errors = []

with open('prediction.log', 'r') as fin:
    for line in fin:
        true, mle, knn, linlin, linlog, topic_lin = map(float, line.rstrip().split())

        global_errors.append(abs(mle - true))
        knn_errors.append(abs(knn - true))
        linreg_errors.append(abs(linlin - true))
        logreg_errors.append(abs(linlog - true))
        topic_errors.append(abs(topic_lin - true))

fig = plt.figure()
ax1 = fig.add_subplot(111)
evaluation_matrix = [global_errors, knn_errors, linreg_errors, logreg_errors, topic_errors]
print([len(x) for x in evaluation_matrix])
ax1.boxplot(evaluation_matrix, labels=['MLE', 'KNN', 'LinReg', 'LinReg-Log', 'Topic-Lin'], showfliers=False, showmeans=True)
ax1.set_ylabel('mean absolute error')

means = [np.mean(x) for x in evaluation_matrix]
means_labels = ['{0:.2f}%'.format(s*100) for s in means]
pos = range(len(means))
for tick, label in zip(pos, ax1.get_xticklabels()):
    ax1.text(pos[tick] + 1, means[tick] + 0.01, means_labels[tick], horizontalalignment='center', size='medium', color='k')

plt.show()