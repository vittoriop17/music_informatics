from torch.nn import Module
from torch.nn.modules import LSTM, Linear, Softmax, Conv1d, MaxPool1d, Sequential, ReLU, BatchNorm1d, Dropout, \
    AvgPool1d, LeakyReLU
import torch
from models.extractor import PreProcessNet_v2, PreProcessNet_stridepool

# TODO - this model works with input sequences of fixed length!
# modify the Dataset code in order to handle input of different length (normal case: 3 seconds audio)

def check_conv1d_out_dim(in_size, kernel, padding, stride, dilation):
    conv1d_out_size = (in_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    assert conv1d_out_size % 1 == 0, "Something went wront. The output of conv1d should have an integer dimension. Not float"
    return int(conv1d_out_size)


class PreProcessNet(Module):
    def __init__(self, args):
        super(PreProcessNet, self).__init__()
        self.input_size = 132299
        self.in_channels = 1
        # stride = 348: almost 70% overlap between contiguous windows.
        # the length of each window is represented by the kernel_size=1103
        # considering the sample rate, this window covers a number of samples in a temporal range of 25 ms
        self.conv1_1 = Conv1d(in_channels=self.in_channels, out_channels=4*self.in_channels, kernel_size=509, stride=138, dilation=1, device=args.device)
        self.conv1_1_out_size = check_conv1d_out_dim(self.input_size, 509, 0, 138, 1)
        self.avg_pool1 = AvgPool1d(kernel_size=4, stride=2)
        self.avg_pool1_out_size = check_conv1d_out_dim(self.conv1_1_out_size, 4, 0, 2, 1)
        self.conv1_2 = Conv1d(in_channels=4*self.in_channels, out_channels=16*self.in_channels, kernel_size=11, device=args.device)
        self.conv1_2_out_size = check_conv1d_out_dim(self.avg_pool1_out_size, 11, 0, 1, 1)
        self.conv1_3 = Conv1d(in_channels=16*self.in_channels, out_channels=64*self.in_channels, kernel_size=7, dilation=1, device=args.device)
        self.conv1_3_out_size = check_conv1d_out_dim(self.conv1_2_out_size, 7, 0, 1, 1)
        self.down_sampling_1 = DownSamplingBLock(args, channels=64*self.in_channels, dilation=1, stride=2)
        self.down_sampling_1_out_size = check_conv1d_out_dim(self.conv1_3_out_size, 3, 0, 2, 1)
        self.down_sampling_2 = DownSamplingBLock(args, channels=64*self.in_channels, dilation=2, stride=1)
        self.down_sampling_2_out_size = check_conv1d_out_dim(self.down_sampling_1_out_size, 3, 0, 1, 2)
        self.down_sampling_3 = DownSamplingBLock(args, channels=64*self.in_channels, dilation=3, stride=1)
        self.down_sampling_3_out_size = check_conv1d_out_dim(self.down_sampling_2_out_size, 3, 0, 1, 3)
        # self.conv1_3 = Conv1d(in_channels=16*self.in_channels, out_channels=32*self.in_channels, kernel_size=7, stride=1, dilation=4)
        # self.conv1_3_out_size = check_conv1d_out_dim(self.down_sampling_3_out_size, 7, 0, 1, 4)
        self.down_sampling_4 = DownSamplingBLock(args, channels=64 * self.in_channels, dilation=4, stride=1)
        self.down_sampling_4_out_size = check_conv1d_out_dim(self.down_sampling_3_out_size, 3, 0, 1, 4)
        # self.down_sampling_5 = DownSamplingBLock(args, channels=64 * self.in_channels, dilation=5, stride=1)
            # Conv1d(in_channels=32 * self.in_channels, out_channels=self.num_sequences, kernel_size=12, stride=4, dilation=1)

        # self.sequence_length = check_conv1d_out_dim(self.down_sampling_4_out_size, 3, 0, 1, 5)
        self.num_sequences = 64 * self.in_channels
        self.sequence_length = self.down_sampling_4_out_size
        self.down_sampling_net = Sequential(
            self.conv1_1,
            self.avg_pool1,
            self.conv1_2,
            self.conv1_3,
            self.down_sampling_1,
            ReLU(),
            self.down_sampling_2,
            ReLU(),
            self.down_sampling_3,
            ReLU(),
            self.down_sampling_4,
            ReLU(),
            # self.down_sampling_5,
            # ReLU()
        )
        self.lstm = LSTM(
            input_size=self.sequence_length,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            device=args.device,
            batch_first=True,
            bidirectional=False,
            num_layers=args.num_layers)
        # todo - add attention block!!!

    def forward(self, x):
        x_down_sampled = self.down_sampling_net(x)
        # x_down_sampled = torch.transpose(x_down_sampled, 1, 2)
        x_lstm, (h_lstm, c_lstm) = self.lstm(x_down_sampled)
        return x_lstm


class ClassificationNet(Module):
    def __init__(self, args, num_sequences):
        super(ClassificationNet, self).__init__()
        self.relu = ReLU()
        # dropout is set to 0 if args.dropout does not exist
        self.dropout = Dropout(p=getattr(args, "dropout", 0))
        self.linear_1 = Linear(in_features=num_sequences * args.hidden_size, out_features=200, device=args.device)
        self.batch_1 = BatchNorm1d(num_features=200, device=args.device)
        self.intro = Sequential(
            self.dropout,
            self.linear_1,
            self.batch_1,
            self.relu,
            self.dropout
        )
        self.linear_2 = Linear(in_features=200, out_features=100, device=args.device)
        self.batch_2 = BatchNorm1d(num_features=100, device=args.device)
        self.middle = Sequential(
            self.linear_2,
            self.batch_2,
            self.relu,
            self.dropout
        )
        # self.linear_2_right = Linear(in_features=250, out_features=100, device=args.device)
        # self.batch_2_right = BatchNorm1d(num_features=100, device=args.device)
        # self.right = Sequential(
        #     self.linear_2_right,
        #     self.batch_2_right,
        #     self.relu,
        #     self.dropout
        # )
        self.linear_3 = Linear(in_features=100, out_features=args.n_classes, device=args.device)
        self.end = Sequential(
            self.linear_3,
        )
        self.softmax = Softmax()

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x_intro = self.intro(x)
        x_middle = self.middle(x_intro)
        # x_right = self.right(x_intro)
        x_end = self.end(x_middle)
        return self.softmax(x_end)


class DownSamplingBLock(Module):
    def __init__(self, args, channels, dilation, stride, kernel_size=3):
        super(DownSamplingBLock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.device = args.device
        self.left = Sequential(MaxPool1d(kernel_size=self.kernel_size, dilation=self.dilation, stride=stride),
                               ReLU())
        self.right = Sequential(Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, device=self.device),
                                Conv1d(in_channels=channels, out_channels=channels, kernel_size=self.kernel_size, dilation=self.dilation, stride=stride, device=self.device))

    def forward(self, x):
        x_left = self.left(x)
        x_right = self.right(x)
        return x_left + x_right


class InstrumentClassificationNet(Module):
    def __init__(self, args):
        super(InstrumentClassificationNet, self).__init__()
        self.preprocessing_net = PreProcessNet_stridepool(args)
        self.classification_net = ClassificationNet(args, self.preprocessing_net.num_sequences)

    def forward(self, x):
        x_pre = self.preprocessing_net(x)
        y_pred = self.classification_net(x_pre)
        return y_pred