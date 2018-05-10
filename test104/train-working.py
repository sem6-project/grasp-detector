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

import models


class CornellDataLoader(Dataset):
    '''What a dataset'''

    def __init__(self, datapoints :list, window_size=(480, 640)):
        '''datapoints : `utils.prepareDataPoints`
        window_size : the size of window/cropped image to consider (crop around center)
        '''
        self.datapoints = datapoints
        self.window_size = window_size

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        x = self.datapoints[idx].X

        h, w = self.window_size
        H, W = x.shape[0], x.shape[1]
        t, b = int((H-h)/2), int((H+h)/2)
        l, r = int((W-w)/2), int((W+w)/2)
        x = x[t:b, l:r]

        x = torch.FloatTensor(x).contiguous().view(1, 1, h, w)
        y = torch.FloatTensor(self.datapoints[idx].Y)
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

        if str(loss.data[0]).strip().lower() == 'nan':
            pass
            # print('\nStuck somewhere in local minima :(')
            # # print('I QUIT')
            # return False

        if batch_idx % args.log_interval == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.data[0]),
                end='')

    return True


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


def visualize_result(datapoints :list, model, cuda :bool, target_dir :str, window_size=(640, 480)) -> None:
    viz_loader = CornellDataLoader(datapoints, window_size)
    get_fname = lambda finfo: 'result{}-{}'.format(finfo[0], os.path.basename(finfo[1]))
    get_target_file = lambda fpath: os.path.join(target_dir, get_fname(fpath))

    for idx, (data, _) in enumerate(viz_loader):
        data = Variable(data, volatile=True)  # dunno why
        if cuda:  data = data.cuda()
        dp = datapoints[idx]
        prediction = model(data).data
        # prediction = tuple(prediction.cpu().data)
        target_file = get_target_file((idx, dp.rgb_image_path))
        dp.visualize_result(prediction, target_file, gray=False)

        print('\r:: Visualize result : ({}/{} {:.2f} %)'.format(


            idx+1, len(datapoints), ((idx+1)/len(datapoints))*100
        ), end='')


def main():
    args = parse_arguments()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    datapoints = utils.prepare_datapoints(data_raw_path=args.data_raw_dir)
    train_datapoints, test_datapoints = \
        utils.train_test_split_datapoints(datapoints, test_size=0.2)

    # window_size = (240, 320)
    window_size = (480, 640)
    train_loader = CornellDataLoader(train_datapoints, window_size=window_size)
    test_loader = CornellDataLoader(test_datapoints, window_size=window_size)

    linear_model_weights = [
        # 640*480,
        320*240,
        # 240*180,
        # 160*120,
        40*30,
        20*15,
        6
    ]
    # linear_model_weights = [
    #    320*240,
    #    40*30,
    #    6
    # ] 
    # model = models.LinearNet(linear_model_weights)
    model = models.ConvNet()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print("Here's the model")
    print(model)
    for epoch in range(1, args.epochs + 1):
        while not train(model, train_loader, optimizer, args, epoch):
            model.reset_parameters()
            print('Restart train...')
        test(model, test_loader, optimizer, args)

    import pdb; pdb.set_trace()

    target_dir = os.path.abspath('./predictions')
    try:
        os.makedirs(target_dir)
    except FileExistsError:
        pass

    visualize_result(datapoints[:10], model, args.cuda, target_dir, window_size)

if __name__=='__main__':
    main()
