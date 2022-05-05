import matplotlib.pyplot as plt
import numpy as np

view0 = np.asarray([0.8855,0.9342,0.864,0.6121,0.1093,0.1218])
view1 = np.asarray([0.103,0.1136,0.9067,0.5542,0.1094,0.0858])

x = [0,1,2,3,4,5]

plt.plot(x, view0, '-o', label = 'baseline MLP')
plt.plot(x, view1, '-o', label = 'baseline CNN')

plt.xlabel('Learning Rate')
plt.xticks(x, [0.0001, 0.001, 0.01, 0.1, 1, 10])

plt.ylabel('Accuracy')
plt.title("Accuracy vs Learning Rate (5 epochs)")

plt.legend(fontsize=8)
plt.savefig('base_acc.png')

plt.show()
