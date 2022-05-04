import matplotlib.pyplot as plt
import numpy as np

view0 = np.asarray([330.0706,353.3178,780.5665,801.1539])
view1 = np.asarray([334.3142,369.7095,770.0582,801.4759])
view2 = np.asarray([337.3142,348.5446,743.2459,801.2346])

x = [0,1,2,3]

plt.plot(x, view0, '-o', label = 'lr = 10')
plt.plot(x, view1, '-o', label = 'lr = 1')
plt.plot(x, view2, '-o', label = 'lr = 0.1')

plt.xlabel('Number of Processors')
plt.xticks(x, [1, 4, 16, 32])

plt.ylabel('Training Time')
plt.title("Training Time vs # of Processors for Different Learning Rates (1 epoch)")

plt.legend(fontsize=8)
plt.savefig('res_lr_time.png')

plt.show()

view0 = np.asarray([0.9342031452,0.42286032,0.4119940002])
view1 = np.asarray([0.9042618596,0.4341414714,0.4171232098])
view2 = np.asarray([0.9677791594,0.4538393014,0.4209930525])

x = [0,1,2]

plt.plot(x, view0, '-o', label = 'lr = 10')
plt.plot(x, view1, '-o', label = 'lr = 1')
plt.plot(x, view2, '-o', label = 'lr = 0.1')

plt.xlabel('Number of Processors')
plt.xticks(x, [4, 16, 32])

plt.ylabel('Speedup')
plt.title("Speedup vs # of Processors for Different Learning Rates (1 epoch)")

plt.legend(fontsize=8)
plt.savefig('res_lr_speed.png')

plt.show()

view0 = np.asarray([0.098,0.098,0.1003,0.096])
view1 = np.asarray([0.1246,0.1136,0.101,0.0974])
view2 = np.asarray([0.9863,0.9759,0.9663,0.9676])

x = [0,1,2,3]

plt.plot(x, view0, '-o', label = 'lr = 10')
plt.plot(x, view1, '-o', label = 'lr = 1')
plt.plot(x, view2, '-o', label = 'lr = 0.1')

plt.xlabel('Number of Processors')
plt.xticks(x, [1, 4, 16, 32])

plt.ylabel('Test Accuracy')
plt.title("Test Accuracy vs # of Processors for Different Learning Rates (1 epoch)")

plt.legend(fontsize=8)
plt.savefig('res_lr_acc.png')

plt.show()
