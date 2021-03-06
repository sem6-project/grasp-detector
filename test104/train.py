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

from sklearn.preprocessing import normalize

import models


class CornellDataLoader(Dataset):
    '''What a dataset'''

    def __init__(self, datapoints :list, backgrounds :dict, bg_mapping :dict, window_size=(480, 640), for_viz=False, normalize=False):
        '''datapoints : `utils.prepareDataPoints`
        window_size : the size of window/cropped image to consider (crop around center)
        '''
        self.datapoints = datapoints
        self.window_size = window_size
        self.bg_mapping = bg_mapping
        self.backgrounds = backgrounds
        self.for_viz = for_viz
        self.normalize = normalize

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        d = self.datapoints[idx]
        bg = None
        if not self.for_viz:
            bg = self.backgrounds[self.bg_mapping[d.image_name]]
            x = self.datapoints[idx].get_image(gray=True, background=bg)

        if idx < 5 and (not self.for_viz):
            utils.cv2.imwrite('image-{}.png'.format(idx), x)

        h, w = self.window_size
        H, W = x.shape[0], x.shape[1]
        t, b = int((H-h)/2), int((H+h)/2)
        l, r = int((W-w)/2), int((W+w)/2)
        x = x[t:b, l:r]
        x = x.astype('float')

        if idx < 5 and (not self.for_viz):
            utils.cv2.imwrite('image-{}-clipped.png'.format(idx), x)

        if self.normalize:
            x = normalize(x)

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
            print('\nStuck somewhere in local minima :(')
            # print('I QUIT')
            return False

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
            # test_loss += F.l1_loss(output, target)
        test_loss += 1 - utils.IOU(
            tuple(output.cpu().data),
            tuple(target.cpu().data)
        )
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}\n'.format(float(test_loss)))
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader),
    #     100. * correct / len(test_loader)))

    return 1-float(test_loss)


def visualize_result(datapoints :list, model, cuda :bool, target_dir :str, backgrounds :dict, bg_mapping :dict, window_size=(640, 480), normalize=False) -> None:
    viz_loader = CornellDataLoader(datapoints, backgrounds, bg_mapping, window_size, for_viz=True, normalize=normalize)
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

    linear_model_weights = [
        # 640*480,
        # 320*240,
        160*120,
        40*30,
        20*15,
        6
    ]
    model = models.LinearNet(linear_model_weights)
    # model = models.ConvNet()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print("Here's the model")
    print(model)

    datapoints = utils.prepare_datapoints(data_raw_path=args.data_raw_dir)
    train_datapoints, test_datapoints = \
                                        utils.train_test_split_datapoints(datapoints, test_size=0.2)

    bg_mapping = utils.get_background_mappings(args.data_raw_dir+'/backgroundMapping.txt')
    backgrounds = utils.read_backgrounds(args.data_raw_dir+'/backgrounds')

    window_size = (120, 160)
    # window_size = (240, 320)
    # window_size = (480, 640)
    # normalize = True
    normalize = False
    train_loader = CornellDataLoader(train_datapoints, backgrounds, bg_mapping, window_size=window_size, normalize=normalize)
    test_loader = CornellDataLoader(test_datapoints, backgrounds, bg_mapping, window_size=window_size, normalize=normalize)


    perf = 0
    for epoch in range(1, args.epochs + 1):
        while not train(model, train_loader, optimizer, args, epoch):
            model.reset_parameters()
            print('Restart train...')
            perf = test(model, test_loader, optimizer, args)

    target_dir = os.path.abspath('./predictions')
    try:
        os.makedirs(target_dir)
    except FileExistsError:
        pass

    visualize_result(datapoints[:10], model, args.cuda, target_dir, backgrounds, bg_mapping, window_size, normalize)

    from datetime import datetime
    model_fname = '{}-{}.pth'.format(datetime.now().isoformat(), perf)
    print('saving model as', model_fname)
    torch.save(model, model_fname)

if __name__=='__main__':
    main()
