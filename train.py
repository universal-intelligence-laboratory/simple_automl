# -*- coding: utf-8 -*-

# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import argparse
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    metavar='N',
    help='input batch size for training (default: 64)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    metavar='N',
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    metavar='N',
    help='number of epochs to train (default: 10)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.5,
    metavar='M',
    help='SGD momentum (default: 0.5)')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--smoke-test', action="store_true", help="Finish quickly for testing")


def main(args):
    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data',
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data',
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
    
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    model = Net()
    
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum)
    
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(
                output, target,
                size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(
                1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    
        test_loss = test_loss.item() / len(test_loader.dataset)
        accuracy = correct.item() / len(test_loader.dataset)
        print(accuracy)
    
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    
    #time.sleep(1)
    #print("init finished.")
    #for i in range(100000000000000000000):
    #    if i%10000000 == 0:
    #        print("logs")
