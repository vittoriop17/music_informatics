import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, maxpool_kernel=2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.avg_pool2 = nn.AvgPool1d(kernel_size=maxpool_kernel)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2,
                                 self.avg_pool2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = self.avg_pool2(x) if self.downsample is None else self.avg_pool2(self.downsample(x))
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs=1, num_channels=[2, 4, 8, 16, 32, 64], kernel_size=2, dropout=0.2, avgpool_kernel=2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.avgpool_kernel = avgpool_kernel
        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, maxpool_kernel=avgpool_kernel)]
        # self.output_size = (num_channels[-1], self.f_original_input)
        self.network = nn.Sequential(*layers)

    def f_original_input(self, input_size):
        # evaluate the output size after TemporalBLock dimensionality reduction,
        # given the length of the input (input_size)
        all_sizes = [np.floor(input_size/(self.avgpool_kernel**lev)).astype(np.int64) for lev in range(1, self.num_levels+1)]
        return all_sizes[-1]

    def forward(self, x):
        return self.network(x)


class ClassificationTCN(nn.Module):
    def __init__(self, args, num_inputs=2, num_channels=[1, 2, 4, 8, 16, 32, 64], kernel_size=2):
        super(ClassificationTCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, args.dropout, 2)
        self.output_size = self.tcn.f_original_input(args.input_size)
        self.lstm = nn.LSTM(input_size=self.output_size,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers,
                            batch_first=True,
                            device=args.device,
                            dropout=args.dropout)
        self.fc = nn.Sequential(nn.Linear(in_features=num_channels[-1]*args.hidden_size, out_features=150, device=args.device),
                                nn.Dropout(args.dropout),
                                nn.BatchNorm1d(num_features=150, device=args.device),
                                nn.ReLU(),
                                nn.Linear(in_features=150, out_features=args.n_classes, device=args.device),
                                nn.Softmax())

    def forward(self, x):
        x = self.tcn(x)
        x, (_, _) = self.lstm(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

