import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib


def pdf(r, n):
    return 1-(1-r)**n


x_axis = np.arange(0.1, 1.1, 0.1)
y_axis = np.arange(1, 11, 1)

fig = plt.figure()
ax1 = fig.add_subplot(111)
patches = []

for x in x_axis:
    for y in y_axis:
        rec = Rectangle((x-0.1, y-1), 0.1, 1)
        patches.append(rec)

colors = []
for x in x_axis:
    for y in y_axis:
        color = pdf(x, y)
        colors.append(color)

p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.6)
p.set_array(np.array(colors))
ax1.add_collection(p)
plt.colorbar(p)

plt.xlabel('coverage ratio')
plt.ylabel('latent occurrence number')
plt.title('pdf wrt coverage ratio and latent occurrence number')
plt.xlim([0, 1])
plt.ylim([0, 10])
plt.show()
