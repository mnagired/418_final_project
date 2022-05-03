import numpy as np
import torch
import torchvision

from mpi4py import MPI

import argparse
import collections
import time

# original resnet MNIST code: https://tinyurl.com/res18418


class MNISTResNet18(torchvision.models.resnet.ResNet):
    def __init__(self):
        super(MNISTResNet18, self).__init__(
            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3), bias=False)



class DataMNIST(object):
    def __init__(self):
        super(DataMNIST, self).__init__()

    def dataset(self, train):
        transform = torchvision.transforms.ToTensor()
        return torchvision.datasets.MNIST(
            "./", train=train, download=True, transform=transform)

    def build(self, train, batch_size):
        self.loader = torch.utils.data.DataLoader(
            self.dataset(train),
            shuffle=True,
            num_workers=0,
            batch_size=batch_size
        )


class Worker(object):
    def __init__(self, comm, rank, size):
        super(Worker, self).__init__()
        self.model = MNISTResNet18()
        self.mnist_data = DataMNIST()
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.comm = comm
        self.rank = rank
        self.size = size

    def iteration(self, x, y):
        logits = self.model(x)
        cost = self.loss(logits, y)
        self.optimizer.zero_grad()

        cost.backward()

        self.optimizer.step()

    def run(self, args):
        print(">>> Beginning training on worker {}".format(self.rank))
        self.mnist_data.build(True, args.batch_size)
        self.model.train()

        for (i, (x, y)) in enumerate(self.mnist_data.loader):
            if i % 10 == 0: print('Worker Batch:', i)
            # print(">>> Iterating on worker {}".format(self.rank))
            self.iteration(x, y)

        print(">>> Sending state dict from worker {}".format(self.rank))
        self.comm.send(self.model.state_dict(), 0)


class Root(object):
    def __init__(self, comm, rank, size):
        super(Root, self).__init__()
        self.model = MNISTResNet18()
        self.mnist_data = DataMNIST()
        self.loss = torch.nn.CrossEntropyLoss()

        self.comm = comm
        self.rank = rank
        self.size = size

    def mean_loss(self):
        val_losses = []
        for (i, (x, y)) in enumerate(self.mnist_data.loader):
            if i % 10 == 0: print('Master Batch:', i)
            val_losses.append(self.loss(self.model(x), y).item())
        return np.mean(val_losses)

    def run(self, args):
        self.mnist_data.build(False, args.batch_size)

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


def train(args):
    """Schedule a distributed training job."""

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    torch.manual_seed(rank)

    epochs = int(args.nepochs)

    if is_root(rank):
        root = Root(MPI.COMM_WORLD, rank, size)
        for i in range(epochs):
            start = time.time()
            print(">>>>> Epoch {} <<<<<".format(i))
            for process in range(1, size):
                MPI.COMM_WORLD.send(root.model.state_dict(), process)
            root.run(args)
            end = time.time()
            print(f'Total Time for Epoch {i + 1}: {end - start:.4f}s')

        #### TESTING ####
        tfs = torchvision.transforms.ToTensor()
        test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=tfs, download = True)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

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
    parser = argparse.ArgumentParser(
        description="Train MNIST network across multiple distributed processes.")
    parser.add_argument("--lr", dest="lr", default=10,
                        help="Learning rate for SGD optimizer. [0.9]")
    parser.add_argument("--momentum", dest="momentum", default=0.9,
                        help="Momentum for SGD optimizer [0.9].")
    parser.add_argument("--batch_size", dest="batch_size", default=512,
                        help="Batch size to use for each process.")
    parser.add_argument("--nepochs", dest="nepochs", default=1,
                        help="Number of epochs (times to loop through the dataset).")
    args = parser.parse_args()

    train(args)
