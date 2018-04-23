import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self._fc1_size = 37 * 27 * 30  # 29970
        self._fc2_size = 6

        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv4 = nn.Conv2d(20, 30, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(self._fc1_size, 500)
        self.fc2 = nn.Linear(500, self._fc2_size)

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

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class LinearNet(nn.Module):
    def __init__(self, layer_dims :list):
        '''Parameters
        -----------
        layer_dims : list telling the size of each linear fully connected layer
        '''
        super(LinearNet, self).__init__()
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1

        print('Setting up the neural network graph')
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims, layer_dims[1:])):
            setattr(self, 'fc{}'.format(i), nn.Linear(in_dim, out_dim))
            print('Created layer', i, 'with dimension', (in_dim, out_dim))

    def forward(self, x):
        x = x.view(-1, 1).view(-1)

        for i in range(self.n_layers - 1):
            fc = getattr(self, 'fc{}'.format(i))
            x = F.relu(fc(x))
            # x = fc(x)

        last_fc = getattr(self, 'fc{}'.format(self.n_layers - 1))
        x = last_fc(x)
        result = x.view(-1)
        return result

    def reset_parameters(self):
        for i in range(self.n_layers):
            fc = getattr(self, 'fc{}'.format(i))
            fc.reset_parameters()
