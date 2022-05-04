import matplotlib.pyplot as plt
import numpy as np

view0 = np.asarray([19.6208,11.3338,7.8413])
view1 = np.asarray([21.3965,13.9797,10.5469])

x = [0,1,2]

plt.plot(x, view0, '-o', label = 'baseline MLP')
plt.plot(x, view1, '-o', label = 'baseline CNN')

plt.xlabel('Batch Size')
plt.xticks(x, [32, 64, 128])

plt.ylabel('Training Time (s)')
plt.title("Training Time vs Batch Size (5 epochs)")

plt.legend(fontsize=8)
plt.savefig('base_time_bs.png')

plt.show()

view0 = np.asarray([0.8917,0.8528,0.7758])
view1 = np.asarray([0.9469,0.9336,0.8869])

x = [0,1,2]

plt.plot(x, view0, '-o', label = 'baseline MLP')
plt.plot(x, view1, '-o', label = 'baseline CNN')

plt.xlabel('Batch Size')
plt.xticks(x, [32, 64, 128])

plt.ylabel('Accuracy')
plt.title("Accuracy vs Batch Size (5 epochs)")

plt.legend(fontsize=8)
plt.savefig('base_acc_bs.png')

plt.show()
