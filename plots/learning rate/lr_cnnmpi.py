import matplotlib.pyplot as plt
import numpy as np

view0 = np.asarray([17.5816,18.2873,19.924,22.1247,27.5907])
view1 = np.asarray([18.0195,18.3637,20.3929,22.8966,27.6131])
view2 = np.asarray([17.9745,18.6945,20.9862,23.5722,29.1781])
view3 = np.asarray([18.3947,18.5498,20.606,23.7403,28.2402])
view4 = np.asarray([18.231,18.8345,20.4985,23.9818,29.7738])
view5 = np.asarray([18.4875,18.7099,20.9849,23.8215,28.5153])

x = [0,1,2,3,4]

plt.plot(x, view0, '-o', label = 'lr = 10')
plt.plot(x, view1, '-o', label = 'lr = 1')
plt.plot(x, view2, '-o', label = 'lr = 0.1')
plt.plot(x, view3, '-o', label = 'lr = 0.01')
plt.plot(x, view4, '-o', label = 'lr = 0.001')
plt.plot(x, view5, '-o', label = 'lr = 0.0001')

plt.xlabel('Number of Processors')
plt.xticks(x, [1, 4, 16, 64, 128])

plt.ylabel('Training Time (s)')
plt.title("Training Time vs # of Processors for Different Learning Rates (5 epochs)")

plt.legend(fontsize=8)
plt.savefig('cnn_mpi_lr_time.png')

plt.show()

view0 = np.asarray([0.9614103777,0.8824332463,0.7946593626,0.6372292113])
view1 = np.asarray([0.9812565006,0.8836163567,0.7869945756,0.6525707001])
view2 = np.asarray([0.9614859986,0.8564914086,0.7625295899,0.6160270888])
view3 = np.asarray([0.9916387239,0.8926865961,0.7748301412,0.6513657835])
view4 = np.asarray([0.9679577371,0.8893821499,0.7602014861,0.6123168692])
view5 = np.asarray([0.9881132449,0.8809906171,0.7760846294,0.6483361564])

x = [0,1,2,3]

plt.plot(x, view0, '-o', label = 'lr = 10')
plt.plot(x, view1, '-o', label = 'lr = 1')
plt.plot(x, view2, '-o', label = 'lr = 0.1')
plt.plot(x, view3, '-o', label = 'lr = 0.01')
plt.plot(x, view4, '-o', label = 'lr = 0.001')
plt.plot(x, view5, '-o', label = 'lr = 0.0001')

plt.xlabel('Number of Processors')
plt.xticks(x, [4, 16, 64, 128])

plt.ylabel('Speedup')
plt.title("Speedup vs # of Processors for Different Learning Rates (5 epochs)")

plt.legend(fontsize=8)
plt.savefig('cnn_mpi_lr_speed.png')

plt.show()

view0 = np.asarray([0.1136,0.1136,0.1136,0.1136,0.1136])
view1 = np.asarray([0.9676,0.9849,0.1136,0.1136,0.1136])
view2 = np.asarray([0.9829,0.9863,0.9861,0.9865,0.9864])
view3 = np.asarray([0.9365,0.9388,0.9385,0.9386,0.9385])
view4 = np.asarray([0.9365,0.1496,0.1473,0.1494,0.1494])
view5 = np.asarray([0.9365,0.101,0.101,0.101,0.101])

x = [0,1,2,3,4]

plt.plot(x, view0, '-o', label = 'lr = 10')
plt.plot(x, view1, '-o', label = 'lr = 1')
plt.plot(x, view2, '-o', label = 'lr = 0.1')
plt.plot(x, view3, '-o', label = 'lr = 0.01')
plt.plot(x, view4, '-o', label = 'lr = 0.001')
plt.plot(x, view5, '-o', label = 'lr = 0.0001')

plt.xlabel('Number of Processors')
plt.xticks(x, [1, 4, 16, 64, 128])

plt.ylabel('Test Accuracy')
plt.title("Test Accuracy vs # of Processors for Different Learning Rates (5 epochs)")

plt.legend(fontsize=8)
plt.savefig('cnn_mpi_lr_acc.png')

plt.show()
