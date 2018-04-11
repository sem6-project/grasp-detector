import utils
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable


class CornellDataLoader(Dataset):
    '''What a dataset'''

    def __init__(self, datapoints :list):
        '''datapoints : `utils.prepareDataPoints`
        '''
        self.datapoints = datapoints

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.datapoints[idx].X).view(1, 1, 480, 640)
        y = torch.FloatTensor(self.datapoints[idx].Y)
        # return self.datapoints[idx].X, self.datapoints[idx].Y
        return x, y


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description='Robot Grasping on Cornell Dataset')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-raw-dir', type=str, default='../../DataRaw',
                        help='Path to DataRaw where images are')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('Obtaining data from', os.path.abspath(args.data_raw_dir))

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    return args


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self._fc1_size = 37 * 27 * 30  # 29970

        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv4 = nn.Conv2d(20, 30, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(self._fc1_size, 500)
        self.fc2 = nn.Linear(500, 5)

    def forward(self, x):
        # convolution and dropouts
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        # x = F.dropout(x, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2))
        # x = F.dropout(x, training=self.training)
        x = F.relu(F.max_pool2d(self.conv4(x), kernel_size=2))

        # making fully connected
        x = x.view(-1, self._fc1_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        result = x.view(-1)
        return result
        # return F.log_softmax(x, dim=1)


def train(model, train_loader, optimizer, args, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.data[0]),
                end='')


def test(model, test_loader, optimizer, args):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        # test_loss += F.mse_loss(output, target)
        test_loss += F.l1_loss(output, target)
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}\n'.format(float(test_loss)))
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader),
    #     100. * correct / len(test_loader)))
    
    return 1-test_loss


def main():
    args = parse_arguments()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    datapoints = utils.prepare_datapoints(data_raw_path=args.data_raw_dir)
    train_datapoints, test_datapoints = \
        utils.train_test_split_datapoints(datapoints, test_size=0.2)

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                 transform=transforms.Compose([
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((0.1307,), (0.3081,))
    #                 ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((0.1307,), (0.3081,))
    #                 ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)

    train_loader = CornellDataLoader(train_datapoints)
    test_loader = CornellDataLoader(test_datapoints)
   
    model = Net()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, args, epoch)
        test(model, test_loader, optimizer, args)

    
   #  data, target = test_loader[0]
   #  if args.cuda:
   #      data, target = data.cuda(), target.cuda()
   #  data, target = Variable(data, volatile=True), Variable(target)
   #  image = data.view(480, 640).numpy()
   #  output = model(data)
   #  
   #  pred = utils.get_rectangle_vertices(list(output))
   #  actual = utils.get_rectangle_vertices(list(target))

   #  import cv2
   #  cv2.imwrite('original.png', image)
   #  cv2.polylines(image, list(actual), True, (0, 255, 0));
   #  cv2.polylines(image, list(pred), True, (255, 0, 0));
   #  cv2.imwrite('detected.png', image);


if __name__=='__main__':
    main()
