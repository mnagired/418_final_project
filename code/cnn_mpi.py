import numpy as np
import torch
import torchvision

from mpi4py import MPI

import argparse
import collections
import time

tfs = torchvision.transforms.ToTensor();

##################### DataLoader #####################

class DataMNIST(object):
    def __init__(self):
        super(DataMNIST, self).__init__()

    def dataset(self, train):
        return torchvision.datasets.MNIST(
            "./", train=train, download=True, transform=tfs)

    def build(self, train, batch_size):
        self.loader = torch.utils.data.DataLoader(
            self.dataset(train), shuffle=True, num_workers=0, batch_size=int(args.batch_size)

class ClassifyMNISTConv(torch.nn.Module):
    def __init__(self):
        super(ClassifyMNISTConv, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.ReLU()
        )

    def forward(self, input):
        return self.network(input).view(-1, 320)


class ClassifyMNISTFwd(torch.nn.Module):
    def __init__(self):
        super(ClassifyMNISTFwd, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(50, 10)
        )

    def forward(self, input):
        return self.network(input)


class ClassifyMNIST(torch.nn.Module):
    def __init__(self):
        super(ClassifyMNIST, self).__init__()
        self.conv = ClassifyMNISTConv()
        self.fwd = ClassifyMNISTFwd()

    def forward(self, input):
        return self.fwd(self.conv(input).view(-1, 320))


class Worker(object):
    def __init__(self, comm, rank, size):
        super(Worker, self).__init__()
        self.model = ClassifyMNIST()
        self.mnist_data = DataMNIST()
        self.loss = torch.nn.CrossEntropyLoss()

        self.comm = comm
        self.rank = rank
        self.size = size

    def iteration(self, x, y):
        self.optimizer.zero_grad()
        self.loss(self.model(x), y).backward()
        self.optimizer.step()

    def run(self, args):
        self.mnist_data.build(True, int(args.batch_size))
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=float(args.lr))

        self.model.to(device)

        for (x, y) in self.mnist_data.loader:
            x, y = x.to(device), y.to(device)
            self.iteration(x, y)

        self.comm.send(self.model.state_dict(), 0)


class Root(object):
    def __init__(self, comm, rank, size):
        super(Root, self).__init__()
        self.model = ClassifyMNIST()
        self.mnist_data = DataMNIST()
        self.loss = torch.nn.CrossEntropyLoss()

        self.comm = comm
        self.rank = rank
        self.size = size

    def mean_loss(self):
        val_losses = []
        self.model.to(device)
        for x, y in self.mnist_data.loader:
            x, y = x.to(device), y.to(device)
            val_losses.append(self.loss(self.model(x), y).item())
        return np.mean(val_losses)

    def run(self, args):
        self.mnist_data.build(False, int(args.batch_size))

        print(">>> Receiving state dicts from workers")
        state_dicts = [self.comm.recv() for _ in range(self.size - 1)]
        print(">>> State dicts received")

        averages = collections.OrderedDict()
        for key in state_dicts[0].keys():
            value = 0.0
            for state_dict in state_dicts:
                value += state_dict[key]
            value /= float(self.size - 1)
            averages[key] = value

        self.model.load_state_dict(averages)
        self.model.eval()

        with torch.no_grad():
            print("Average loss: {}".format(self.mean_loss()))


def is_root(rank):
    return rank == 0

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

def eval(args):
    """Schedule a distributed training job."""

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    torch.manual_seed(rank)

    epochs = int(args.nepochs)

    if is_root(rank):
        root = Root(MPI.COMM_WORLD, rank, size)
        for i in range(epochs):
            start = time.time()
            print(">>>>> Epoch {} <<<<<".format(i+1))
            for process in range(1, size):
                MPI.COMM_WORLD.send(root.model.state_dict(), process)
            root.run(args)
            end = time.time()
            print(f'Total Time for Epoch {i + 1}: {end - start:.4f}s')

            #### TESTING ####
            test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=tfs, download = True)
            test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=int(args.batch_size), shuffle=False)

            test_loss, test_acc = [], []
            for data,labels in test_loader:
                # optimization is to not use gradients (bc we don't need it for evaluation)
                with torch.no_grad():
                    data, labels = data.to(device), labels.to(device)

                    outputs = root.model(data)
                    loss = root.loss(outputs, labels)

                    test_loss += [loss.item()]
                    test_acc += [(outputs.argmax(dim=1).cpu().numpy() == labels.cpu().numpy()).mean()]

            print(f'Test Accuracy: {np.mean(test_acc):.4f}, Test Loss: {np.mean(test_loss):.4f}')

    else:
        for _ in range(epochs):
            worker = Worker(MPI.COMM_WORLD, rank, size)
            worker.model.load_state_dict(MPI.COMM_WORLD.recv())
            worker.run(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN on MNIST with MPI")
    parser.add_argument("--lr", dest="lr", default=10,
                        help="Learning Rate. [default 10]")
    parser.add_argument("--batch_size", dest="batch_size", default=128,
                        help="Batch Size. [default 128]")
    parser.add_argument("--nepochs", dest="nepochs", default=1,
                        help="Number of Epochs. [default 1]")
    args = parser.parse_args()

    eval(args)
