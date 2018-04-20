import utils
import os
from copy import deepcopy
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
        x1 = torch.FloatTensor(self.datapoints[idx].X).view(1, 1, 480, 640)
        x2 = torch.FloatTensor([self.datapoints[idx].intent])
        y = torch.FloatTensor(self.datapoints[idx].Y)
        # return self.datapoints[idx].X, self.datapoints[idx].Y
        x = (x1, x2)
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
    parser.add_argument('--data-raw-dir', type=str,
                        help='Path to DataRaw where images are')
    parser.add_argument('--intent-data', type=str, default='../intent-dataset/mappings.json',
                        help='File where intent dataset is present')
    parser.add_argument('--fix-rectangles', type=bool, default=False,
                        help='Fix rectangle coordinates from intent data to form proper rectangle')
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
        self._fc2_size = 30
        self._fc3_size = 6

        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv4 = nn.Conv2d(20, 30, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(self._fc1_size, 500)
        self.fc2 = nn.Linear(500, self._fc2_size)
        self.fc3 = nn.Linear(self._fc2_size + 1, self._fc3_size)

    def forward(self, x):
        x1, x2 = x
        x2 = x2.view(1, 1)
        # convolution and dropouts
        r = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2))
        r = F.relu(F.max_pool2d(self.conv2(r), kernel_size=2))
        # r = F.dropout(r, training=self.training)
        r = F.relu(F.max_pool2d(self.conv3(r), kernel_size=2))
        # r = F.dropout(r, training=self.training)
        r = F.relu(F.max_pool2d(self.conv4(r), kernel_size=2))

        # making fully connected
        r = r.view(-1, self._fc1_size)
        r = F.relu(self.fc1(r))
        r = F.dropout(r, training=self.training)
        r = F.relu(self.fc2(r))

        r = torch.cat((r, x2), dim=1)

        r = F.relu(self.fc3(r))
        result = r.view(-1)

        return result
        return F.log_softmax(r, dim=1)


def train(model, train_loader, optimizer, args, epoch) -> bool:
    model.train()
    for batch_idx, ((image, intent), target) in enumerate(train_loader):
        if args.cuda:
            image, intent, target = image.cuda(), intent.cuda(), target.cuda()
        image, intent, target = Variable(image), Variable(intent), Variable(target)
        optimizer.zero_grad()
        output = model((image, intent))
        # loss = F.nll_loss(output, target)
        loss = F.mse_loss(output, target)
        # loss = utils.IOU(output.data.cpu().numpy(), target.data.cpu().numpy())
        loss.backward()
        optimizer.step()

        if str(loss.data[0]).strip().lower() == 'nan':
            print('\nStuck somewhere in local minima :(')
            return False

        if batch_idx % args.log_interval == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader),
                100. * batch_idx / len(train_loader), loss.data[0]),
                end='')

    return True


def test(model, test_loader, optimizer, args):
    model.eval()
    test_loss = 0
    correct = 0
    for (image, intent), target in test_loader:
        if args.cuda:
            image, intent, target = image.cuda(), intent.cuda(), target.cuda()
        image, intent, target = Variable(image), Variable(intent), Variable(target)
        output = model((image, intent))
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


def visualize_result(datapoints :list, model :Net, cuda :bool, target_dir :str) -> None:
    viz_loader = CornellDataLoader(datapoints)
    get_fname = lambda finfo: 'result{}-{}'.format(finfo[0], os.path.basename(finfo[1]))
    get_target_file = lambda fpath: os.path.join(target_dir, get_fname(fpath))

    failed = 0

    for idx, ((image, intent), target) in enumerate(viz_loader):
        if cuda:
            image, intent, target = image.cuda(), intent.cuda(), target.cuda()
        image, intent, target = Variable(image), Variable(intent), Variable(target)
        dp = datapoints[idx]
        prediction = model((image, intent))
        prediction = tuple(prediction.cpu().data[0])
        if any([x < 0 for x in prediction]):
            failed += 1
        target_file = get_target_file((idx, dp.rgb_image_path))
        dp.visualize_result(prediction, target_file, gray=False)

        print('\r:: Visualize result : ({}/{} {:.2f} %) failed : {}'.format(
            idx+1, len(datapoints), ((idx+1)/len(datapoints))*100, failed
        ), end='')


def main():
    args = parse_arguments()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    datapoints = utils.prepare_datapoints(data_raw_path=args.data_raw_dir,
                                          intent_data_file=args.intent_data,
                                          fix_rectangles=args.fix_rectangles)
    train_datapoints, test_datapoints = \
        utils.train_test_split_datapoints(datapoints, test_size=0.2)

    train_loader = CornellDataLoader(train_datapoints)
    test_loader = CornellDataLoader(test_datapoints)

    model = Net()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        while not train(model, train_loader, optimizer, args, epoch):
            pass
        test(model, test_loader, optimizer, args)


    target_dir = os.path.abspath('./predictions')
    try:
        os.makedirs(target_dir)
    except FileExistsError:
        pass

    visualize_result(datapoints[:10], model, args.cuda, target_dir)

if __name__=='__main__':
    main()
