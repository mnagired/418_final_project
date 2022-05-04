import matplotlib.pyplot as plt
import numpy as np

view0 = np.divide(np.asarray([5224,1388,3842,27293,53210]), 1000)
view1 = np.divide(np.asarray([25273,7029,10131,38446,82669]), 1000)
view2 = np.divide(np.asarray([25175,6976,10585,41382,77049]), 1000)
view3 = np.divide(np.asarray([25201,7050,10892,40665,71630]), 1000)
view4 = np.divide(np.asarray([25156,7132,11101,39985,75069]), 1000)
view5 = np.divide(np.asarray([25428,6788,10470,38194,85355]), 1000)

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
plt.savefig('mp_lr_time.png')

plt.show()

view0 = np.asarray([3.763688761,1.359708485,0.1914043894,0.09817703439])
view1 = np.asarray([3.595532793,2.494620472,0.6573635749,0.3057131452])
view2 = np.asarray([3.608801606,2.378365612,0.6083562902,0.3267401264])
view3 = np.asarray([3.574609929,2.313716489,0.6197221198,0.3518218623])
view4 = np.asarray([3.527201346,2.266102153,0.629135926,0.3351050367])
view5 = np.asarray([3.746022392,2.428653295,0.6657590197,0.2979087341])

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
plt.savefig('mp_lr_speed.png')

plt.show()

view0 = np.divide(np.asarray([8.92,12.55,11.9,9.8,9.8]), 100)
view1 = np.divide(np.asarray([89.59,63.39,39.36,9.8,9.8]), 100)
view2 = np.divide(np.asarray([94.74,78.89,52.78,43.18,23.1]), 100)
view3 = np.divide(np.asarray([90.13,75.51,33.41,11.63,9.81]), 100)
view4 = np.divide(np.asarray([80.75,48.53,14.59,13.47,10.84]), 100)
view5 = np.divide(np.asarray([44.99,23.41,9.09,10.23,7.04]), 100)

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
plt.savefig('mp_lr_acc.png')

plt.show()
