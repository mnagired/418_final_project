import matplotlib.pyplot as plt
import numpy as np

view0 = np.asarray([19.2842,18.7064,18.3791])
view1 = np.asarray([19.9767,19.0717,18.4798])
view2 = np.asarray([23.5462,21.1121,20.8717])
view3 = np.asarray([26.851,24.453,23.9298])
view4 = np.asarray([32.8994,29.6618,28.9335])

x = [0,1,2]

plt.plot(x, view0, '-o', label = '1 Proccessor')
plt.plot(x, view1, '-o', label = '4 Proccessors')
plt.plot(x, view2, '-o', label = '16 Proccessors')
plt.plot(x, view3, '-o', label = '64 Proccessors')
plt.plot(x, view4, '-o', label = '128 Proccessors')

plt.xlabel('Batch Size')
plt.xticks(x, [32, 64, 128])

plt.ylabel('Training Time (s)')
plt.title("Training Time vs Batch Sizes for Different # of Processors")

plt.legend(fontsize=8)
plt.savefig('cnn_mpi_bs_time.png')

plt.show()

view1 = np.asarray([0.9653346148,0.9808459655,0.9945508068])
view2 = np.asarray([0.8189941477,0.8860511271,0.8805751328])
view3 = np.asarray([0.718192991,0.7649940703,0.7680423572])
view4 = np.asarray([0.5861565864,0.630656265,0.6352186911])

x = [0,1,2]

plt.plot(x, view1, '-o', label = '4 Proccessors')
plt.plot(x, view2, '-o', label = '16 Proccessors')
plt.plot(x, view3, '-o', label = '64 Proccessors')
plt.plot(x, view4, '-o', label = '128 Proccessors')

plt.xlabel('Batch Size')
plt.xticks(x, [32, 64, 128])

plt.ylabel('Speedup')
plt.title("Speedup vs Batch Sizes for Different # of Processors")

plt.legend(fontsize=8)
plt.savefig('cnn_mpi_bs_speed.png')

plt.show()

view0 = np.asarray([0.9766,0.9647,0.9365])
view1 = np.asarray([0.9778,0.9653,0.9391])
view2 = np.asarray([0.9767,0.9671,0.9387])
view3 = np.asarray([0.9769,0.9654,0.9387])
view4 = np.asarray([0.977,0.9648,0.9385])

x = [0,1,2]

plt.plot(x, view0, '-o', label = '1 Proccessor')
plt.plot(x, view1, '-o', label = '4 Proccessors')
plt.plot(x, view2, '-o', label = '16 Proccessors')
plt.plot(x, view3, '-o', label = '64 Proccessors')
plt.plot(x, view4, '-o', label = '128 Proccessors')

plt.xlabel('Batch Size')
plt.xticks(x, [32, 64, 128])

plt.ylabel('Test Accuracy')
plt.title("Test Accuracy vs Batch Sizes for Different # of Processors (5 epochs)")

plt.legend(fontsize=8)
plt.savefig('cnn_mpi_bs_acc.png')

plt.show()
