import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(
    description="Train MNIST network across multiple distributed processes.")
parser.add_argument("--lr", dest="lr", default=10,
                    help="Learning rate for SGD optimizer. [0.9]")
parser.add_argument("--batch_size", dest="batch_size", default=512,
                    help="Batch size to use for each process.")
parser.add_argument("--nepochs", dest="nepochs", default=1,
                    help="Number of epochs (times to loop through the dataset).")
args = parser.parse_args()

print("model hyperparams:")
print("lr: ", args.lr)
print("epochs: ", args.nepochs)
print("batch size : ", args.batch_size, '\n')

tfs = torchvision.transforms.ToTensor();
# retrive and download dataset
train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=tfs, download = True)
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=tfs)

# Data loader
bs = 128 # batch size
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=int(args.batch_size), shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=int(args.batch_size), shuffle=False)

# simple NN with 1 hidden layer, fully-connected
class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # data are 28x28
        self.l1 = torch.nn.Linear(28**2, 50)
        # simple activation function
        self.act = torch.nn.Sigmoid()
        # 10 possible outputs (distinct digits)
        self.l2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        output = self.l1(x)
        output = self.act(output)
        output = self.l2(output)
        return output

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
model = NN()
# move model onto GPU (if possible)
model.to(device)

# simple criterion
criterion = torch.nn.CrossEntropyLoss()

# Stochastic Gradient Descent
optim = torch.optim.SGD(model.parameters(), lr=float(args.lr))

##################### MODEL TRAINING #####################

# epochs = 2 #2
for epoch in range(int(args.nepochs)):
    start = time.time()
    train_loss, train_acc = [], []
    for i, (data, labels) in enumerate(train_loader):

        # move data and labels to device
        data, labels = data.to(device), labels.to(device)
        data = data.reshape(data.shape[0], -1)

        outputs = model(data)
        loss = criterion(outputs, labels)

        train_loss += [loss.item()]
        train_acc += [(outputs.argmax(dim=1).cpu().numpy() == labels.cpu().numpy()).mean()]

        # back propagation
        loss.backward()
        # gradient descent
        optim.step()
        # optimization
        optim.zero_grad()

        if i == 0: print("Epoch", epoch+1)
        if (i+1) % 100 == 0:
            print(f'Train Accuracy: {np.mean(train_acc):.4f}, Train Loss: {np.mean(train_loss):.4f}')

    end = time.time()
    print(f'Total Time for Epoch {epoch + 1}: {end - start:.4f}s')


##################### MODEL TESTING #####################

test_loss, test_acc = [], []
for data,labels in test_loader:
    # optimization is to not use gradients (bc we don't need it for evaluation)
    with torch.no_grad():
        data, labels = data.to(device), labels.to(device)
        data = data.reshape(data.shape[0], -1)

        outputs = model(data)
        loss = criterion(outputs, labels)

        test_loss += [loss.item()]
        test_acc += [(outputs.argmax(dim=1).cpu().numpy() == labels.cpu().numpy()).mean()]

print(f'Test Accuracy: {np.mean(test_acc):.4f}, Test Loss: {np.mean(test_loss):.4f}')
