# coding: utf-8
import torch
from torch.autograd import Variable

import utils
from train import CornellDataLoader

model = torch.load('saved-model.pth')
dps = utils.prepare_datapoints('/home/irm15006/DataRaw')
dl = CornellDataLoader(dps)

# visualizing fifth image
p = []
for v, actual in dl:
    v, actual = Variable(v), Variable(actual)
    # cuda and cpu methods are required.. this checkpoint is from GPU
    prediction = model(v.cuda()).cpu()
    p.append(prediction)
    print('\r {} / {}'.format(len(p) , len(dl)), end='')
    
