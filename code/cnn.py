import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

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

# simple CNN with 2 convolutional layers
class CNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(50, 10)
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out.view(-1, 320))
        return out

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
model = CNN()
# move model onto GPU (if possible)
model.to(device)

# simple criterion
criterion = torch.nn.CrossEntropyLoss()

# Stochastic Gradient Descent
optim = torch.optim.SGD(model.parameters(), lr=float(args.lr))

##################### MODEL TRAINING #####################

epochs = 1 #2
for epoch in range(epochs):
    start = time.time()
    train_loss, train_acc = [], []
    for i, (data, labels) in enumerate(train_loader):

        # move data and labels to device
        data, labels = data.to(device), labels.to(device)

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

        outputs = model(data)
        loss = criterion(outputs, labels)

        test_loss += [loss.item()]
        test_acc += [(outputs.argmax(dim=1).cpu().numpy() == labels.cpu().numpy()).mean()]

print(f'Test Accuracy: {np.mean(test_acc):.4f}, Test Loss: {np.mean(test_loss):.4f}')
