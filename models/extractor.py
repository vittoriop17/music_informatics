import torch
from torch.nn import Module
from torch.nn.modules import LSTM, Linear, Softmax, Conv1d, MaxPool1d, Sequential, ReLU, BatchNorm1d, Dropout, \
    AvgPool1d, LeakyReLU
import torch
import models.lstm_model as lstm_model


# TODO - this model works with input sequences of fixed length!
# modify the Dataset code in order to handle input of different length (normal case: 3 seconds audio)

def check_conv1d_out_dim(in_size, kernel, padding, stride, dilation):
    conv1d_out_size = (in_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    assert conv1d_out_size % 1 == 0, "Something went wront. The output of conv1d should have an integer dimension. Not float"
    return int(conv1d_out_size)


def get_number_params(module):
    pass


class PreProcessNet_stridepool(Module):
    def __init__(self, args):
        super(PreProcessNet_stridepool, self).__init__()
        self.input_size = 132299
        self.device = args.device
        self.args = args
        self.channels = [1, 32, 64, 64, 128, 128]
        self.strides = [1, 1, 1, 1, 1]
        self.kernel_sizes = [5, 5, 7, 13, 15]
        self.output_sizes = []
        self.pool_kernel = 5
        self.pool_stride = 5
        in_size = self.input_size
        for (s, k) in zip(self.strides, self.kernel_sizes):
            in_size = check_conv1d_out_dim(in_size=in_size, kernel=k, stride=s, padding=0, dilation=1)
            # pooling operation
            in_size = check_conv1d_out_dim(in_size=in_size, kernel=self.pool_kernel, stride=self.pool_stride, padding=0, dilation=1)
            self.output_sizes.append(in_size)

        self.conv1 = Conv1d(in_channels=self.channels[0],
                            out_channels=self.channels[1],
                            kernel_size=self.kernel_sizes[0],
                            stride=self.strides[0], device=args.device)
        self.conv2 = Conv1d(in_channels=self.channels[1],
                            out_channels=self.channels[2],
                            kernel_size=self.kernel_sizes[1],
                            stride=self.strides[1], device=args.device)
        self.res_block3 = lstm_model.DownSamplingBLock(args, channels=self.channels[3], kernel_size=self.kernel_sizes[2],
                                            stride=self.strides[2], dilation=1)
        self.conv4 = Conv1d(in_channels=self.channels[3], out_channels=self.channels[4],
                            kernel_size=self.kernel_sizes[3], stride=self.strides[3], device=args.device)
        self.res_block5 = lstm_model.DownSamplingBLock(args, channels=self.channels[4],  dilation=1,
                                            stride=self.strides[4], kernel_size=self.kernel_sizes[4])

        self.num_sequences = self.channels[-1]
        self.sequence_length = self.output_sizes[-1]
        self.down_sampling_net = Sequential(
            self.conv1,
            MaxPool1d(self.pool_kernel),
            LeakyReLU(0.1),
            self.conv2,
            MaxPool1d(self.pool_kernel),
            LeakyReLU(0.1),
            self.res_block3,
            MaxPool1d(self.pool_kernel),
            LeakyReLU(0.1),
            self.conv4,
            MaxPool1d(self.pool_kernel),
            LeakyReLU(0.1),
            self.res_block5,
            MaxPool1d(self.pool_kernel),
            LeakyReLU(0.1)
        )
        # self.la_net = LA_Net(args)
        self.lstm = LSTM(
            input_size=self.sequence_length,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            device=args.device,
            batch_first=True,
            bidirectional=False,
            num_layers=args.num_layers)

    def forward(self, x):
        x_down_sampled = self.down_sampling_net(x)
        # hidden_state = torch.unsqueeze(self.la_net(x), 0)
        # h_state = torch.stack((hidden_state, torch.rand((self.args.num_layers-1, x.shape[0], self.args.hidden_size),
        #                                               device=self.device))).squeeze(dim=1)
        # c_state = torch.rand((self.args.num_layers, x.shape[0], self.args.hidden_size), device=self.device)
        # x_down_sampled = torch.transpose(x_down_sampled, 1, 2)
        x_lstm, (_, _) = self.lstm(x_down_sampled)  # , (h_state, c_state))

        return x_lstm


class PreProcessNet_v2(Module):
    def __init__(self, args):
        super(PreProcessNet_v2, self).__init__()
        self.input_size = 132299
        self.device = args.device
        self.args = args
        self.channels = [1, 16, 64, 64, 128, 128]
        self.strides = [191, 2, 1, 2, 1]
        self.kernel_sizes = [509, 5, 3, 3, 3]
        self.output_sizes = []
        in_size = self.input_size
        for (s, k) in zip(self.strides, self.kernel_sizes):
            in_size = check_conv1d_out_dim(in_size=in_size, kernel=k, stride=s, padding=0, dilation=1)
            # pooling operation
            in_size = check_conv1d_out_dim(in_size=in_size, kernel=3, stride=1, padding=0, dilation=1)
            self.output_sizes.append(in_size)

        self.conv1 = Conv1d(in_channels=self.channels[0],
                            out_channels=self.channels[1],
                            kernel_size=self.kernel_sizes[0],
                            stride=self.strides[0], device=args.device)
        self.conv2 = Conv1d(in_channels=self.channels[1],
                            out_channels=self.channels[2],
                            kernel_size=self.kernel_sizes[1],
                            stride=self.strides[1], device=args.device)
        self.res_block3 = lstm_model.DownSamplingBLock(args, channels=self.channels[3], kernel_size=self.kernel_sizes[2],
                                            stride=self.strides[2], dilation=1)
        self.conv4 = Conv1d(in_channels=self.channels[3], out_channels=self.channels[4],
                            kernel_size=self.kernel_sizes[3], stride=self.strides[3], device=args.device)
        self.res_block5 = lstm_model.DownSamplingBLock(args, channels=self.channels[4],  dilation=1,
                                            stride=self.strides[4], kernel_size=self.kernel_sizes[4])

        self.num_sequences = self.channels[-1]
        self.sequence_length = self.output_sizes[-1]
        self.down_sampling_net = Sequential(
            self.conv1,
            MaxPool1d(3, stride=1),
            LeakyReLU(0.1),
            self.conv2,
            MaxPool1d(3, stride=1),
            LeakyReLU(0.1),
            self.res_block3,
            MaxPool1d(3, stride=1),
            LeakyReLU(0.1),
            self.conv4,
            MaxPool1d(3, stride=1),
            LeakyReLU(0.1),
            self.res_block5,
            MaxPool1d(3, stride=1),
            LeakyReLU(0.1)
        )
        # self.la_net = LA_Net(args)
        self.lstm = LSTM(
            input_size=self.sequence_length,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            device=args.device,
            batch_first=True,
            bidirectional=False,
            num_layers=args.num_layers)

    def forward(self, x):
        x_down_sampled = self.down_sampling_net(x)
        # hidden_state = torch.unsqueeze(self.la_net(x), 0)
        # h_state = torch.stack((hidden_state, torch.rand((self.args.num_layers-1, x.shape[0], self.args.hidden_size),
        #                                               device=self.device))).squeeze(dim=1)
        # c_state = torch.rand((self.args.num_layers, x.shape[0], self.args.hidden_size), device=self.device)
        # x_down_sampled = torch.transpose(x_down_sampled, 1, 2)
        x_lstm, (_, _) = self.lstm(x_down_sampled)  # , (h_state, c_state))

        return x_lstm


class LA_Net(Module):
    def __init__(self, args):
        super(LA_Net, self).__init__()
        self.input_size = 132299
        self.channels = [1, 2, 4, 4, 6, 6]
        self.strides = [2, 3, 3, 5, 3]
        self.dilation = [1024, 1024, 512, 248, 81]
        self.kernel_sizes = [5, 5, 6, 8, 10]
        self.output_sizes = []
        in_size = self.input_size
        for (s, k, d) in zip(self.strides, self.kernel_sizes, self.dilation):
            in_size = check_conv1d_out_dim(in_size=in_size, kernel=k, stride=s, padding=0, dilation=d)
            # pooling operation
            in_size = check_conv1d_out_dim(in_size=in_size, kernel=3, stride=1, padding=0, dilation=1)
            self.output_sizes.append(in_size)
        self.hidden_size = self.output_sizes[-1] * self.channels[-1]
        setattr(args, "hidden_size", self.hidden_size)
        self.conv1 = Conv1d(in_channels=self.channels[0],
                            out_channels=self.channels[1],
                            dilation=self.dilation[0],
                            kernel_size=self.kernel_sizes[0],
                            stride=self.strides[0], device=args.device)
        self.conv2 = Conv1d(in_channels=self.channels[1],
                            out_channels=self.channels[2],
                            dilation=self.dilation[1],
                            kernel_size=self.kernel_sizes[1],
                            stride=self.strides[1], device=args.device)
        self.res_block3 = lstm_model.DownSamplingBLock(args, channels=self.channels[3], kernel_size=self.kernel_sizes[2],
                                            stride=self.strides[2], dilation=self.dilation[2])
        self.conv4 = Conv1d(in_channels=self.channels[3], out_channels=self.channels[4], dilation=self.dilation[3],
                            kernel_size=self.kernel_sizes[3], stride=self.strides[3], device=args.device)
        self.res_block5 = lstm_model.DownSamplingBLock(args, channels=self.channels[4],  dilation=self.dilation[4],
                                            stride=self.strides[4], kernel_size=self.kernel_sizes[4])

        self.down_sampling_net = Sequential(
            self.conv1,
            MaxPool1d(3, stride=1),
            LeakyReLU(0.1),
            self.conv2,
            MaxPool1d(3, stride=1),
            LeakyReLU(0.1),
            self.res_block3,
            MaxPool1d(3, stride=1),
            LeakyReLU(0.1),
            self.conv4,
            MaxPool1d(3, stride=1),
            LeakyReLU(0.1),
            self.res_block5,
            MaxPool1d(3, stride=1),
            LeakyReLU(0.1)
        )


    def forward(self, x):
        x = self.down_sampling_net(x)
        x = torch.flatten(x, start_dim=1)
        return x