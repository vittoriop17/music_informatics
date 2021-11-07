from torch.nn import Module
from torch.nn.modules import LSTM, Linear, Softmax, Conv1d, MaxPool1d, Sequential, ReLU, BatchNorm1d, Dropout, \
    AvgPool1d, LeakyReLU
import torch


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
        self.device = args.device
        self.args = args
        self.channels = [1, 16, 16, 64, 64, 128]
        self.strides = [1, 2, 4, 7, 1]
        self.kernel_sizes = [5, 10, 13, 20, 21]
        self.output_sizes = []
        self.pool_kernel = 5
        self.pool_stride = 2
        self.modules = [Conv1d, ResBlock,
                        Conv1d, ResBlock,
                        Conv1d]
        self.down_sampling_modules = []
        in_size = self.input_size
        for (s, k) in zip(self.strides, self.kernel_sizes):
            in_size = check_conv1d_out_dim(in_size=in_size, kernel=k, stride=s, padding=0, dilation=1)
            # pooling operation
            in_size = check_conv1d_out_dim(in_size=in_size, kernel=self.pool_kernel, stride=self.pool_stride, padding=0, dilation=1)
            self.output_sizes.append(in_size)
        for (idx, (module, k, s)) in enumerate(zip(self.modules, self.kernel_sizes, self.strides)):
            self.down_sampling_modules.append(module(in_channels=self.channels[idx],
                                                 out_channels=self.channels[idx+1],
                                                 kernel_size=k,
                                                 dilation=1,
                                                 stride=s,
                                                 device=args.device))
            self.down_sampling_modules.append(MaxPool1d(self.pool_kernel, stride=self.pool_stride))
            self.down_sampling_modules.append(LeakyReLU(0.1))

        self.down_sampling_net = Sequential(*self.down_sampling_modules)

        self.num_sequences = self.channels[-1]
        self.sequence_length = self.output_sizes[-1]

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
        self.linear_3 = Linear(in_features=100, out_features=args.n_classes, device=args.device)
        self.end = Sequential(
            self.linear_3,
        )
        self.softmax = Softmax()

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x_intro = self.intro(x)
        x_middle = self.middle(x_intro)
        x_end = self.end(x_middle)
        return self.softmax(x_end)


class ResBlock(Module):
    def __init__(self, in_channels, out_channels, dilation, stride, kernel_size, device):
        super(ResBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.device = device
        self.in_c = in_channels
        self.out_c = out_channels
        self.left = Sequential(MaxPool1d(kernel_size=self.kernel_size, dilation=self.dilation, stride=stride),
                               ReLU())
        self.right = Sequential(Conv1d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=1, device=self.device),
                                Conv1d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=self.kernel_size, dilation=self.dilation, stride=stride, device=self.device))

    def forward(self, x):
        x_left = self.left(x)
        x_right = self.right(x)
        return x_left + x_right


class InstrumentClassificationNet(Module):
    def __init__(self, args):
        super(InstrumentClassificationNet, self).__init__()
        self.preprocessing_net = PreProcessNet(args)
        self.classification_net = ClassificationNet(args, self.preprocessing_net.num_sequences)

    def forward(self, x):
        x_pre = self.preprocessing_net(x)
        y_pred = self.classification_net(x_pre)
        return y_pred