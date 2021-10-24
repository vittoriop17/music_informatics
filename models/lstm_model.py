from torch.nn import Module
from torch.nn.modules import LSTM, Linear, Softmax, Conv1d, MaxPool1d, Sequential, ReLU, BatchNorm1d
import torch


# TODO - this model works with input sequences of fixed length!
# modify the Dataset code in order to handle input of different length (normal case: 3 seconds audio)

def check_conv1d_out_dim(in_size, kernel, padding, stride, dilation):
    conv1d_out_size = (in_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    assert conv1d_out_size % 1 == 0, "Something went wront. The output of conv1d should have an integer dimension. Not float"
    return int(conv1d_out_size)


def find_stride(in_size, max_stride):
    # N.B: in_size = original_in_size + 2*padding - dilation*(kernel_size-1) - 1
    # see conv1d for explanation
    for stride in range(2, max_stride+1):
        if in_size % stride == 0:
            return stride
    return 1


class LSTM_model(Module):
    def __init__(self, args):
        super(LSTM_model, self).__init__()
        self.check_args(args)
        self.device = args.device
        self.num_layers = args.num_layers
        self.input_size = args.input_size
        # self.sequence_length = args.sequence_length
        self.sequence_length = args.sequence_length
        self.batch_size = args.batch_size
        self.n_classes = args.n_classes
        self.hidden_size = args.hidden_size
        # in_channels = 2: we consider the audio with left and right channels (see utils.dataset.read_audio)
        # out_channels: represent the number of filters that we want to apply to our input
        self.kernel_1, self.stride_1, self.padding_1, self.dilation_1 = 8, 3, 6, 2
        out1 = check_conv1d_out_dim(132300, self.kernel_1, self.padding_1, self.stride_1, self.dilation_1)
        self.conv1d_1 = Conv1d(in_channels=2, out_channels=4, kernel_size=self.kernel_1, stride=self.stride_1,
                               padding=self.padding_1, dilation=self.dilation_1)

        self.kernel_2, self.stride_2, self.padding_2, self.dilation_2 = 12, 4, 4, 1
        out2 = check_conv1d_out_dim(out1, self.kernel_2, self.padding_2, self.stride_2, self.dilation_2)
        self.conv1d_2 = Conv1d(in_channels=4, out_channels=8, kernel_size=self.kernel_2, stride=self.stride_2,
                               padding=self.padding_2, dilation=self.dilation_2)

        self.kernel_3, self.stride_3, self.padding_3, self.dilation_3 = 17, 5, 6, 1
        self.refinement_output_size = check_conv1d_out_dim(out2, self.kernel_3, self.padding_3, self.stride_3,
                                                                self.dilation_3)
        self.conv1d_3 = Conv1d(in_channels=8, out_channels=self.sequence_length, kernel_size=self.kernel_3,
                               stride=self.stride_3, dilation=self.dilation_3, padding=self.padding_3)

        self.refinement_network = torch.nn.Sequential(self.conv1d_1, self.conv1d_2, self.conv1d_3)
        self.lstm = LSTM(
            input_size=self.refinement_output_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2,
            batch_first=True,
            device=self.device
        )
        self.fc = torch.nn.Sequential(Linear(self.sequence_length * self.hidden_size, 200, device=self.device),
                                      Linear(200, 100, device=self.device),
                                      Linear(100, self.n_classes, device=self.device),
                                      Softmax(dim=self.n_classes)
                                      )

    def check_args(self, args):
        assert hasattr(args, "num_layers"), "Argument 'num_layers' not found!"
        assert hasattr(args, "input_size"), "Argument 'input_size' not found!"
        assert hasattr(args, "hidden_size"), "Argument 'hidden_size' not found!"
        assert hasattr(args, "num_layers"), "Argument 'num_layers' not found!"
        assert hasattr(args, "sequence_length"), "Argument 'sequence_length' not found!"
        assert hasattr(args, "device"), "Argument 'device' not found!"
        assert hasattr(args, "n_classes"), "Argument 'n_classes' not found!"

    def init_state(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.input_size, dtype=torch.double, device=self.device),
                torch.zeros(self.num_layers, self.batch_size, self.input_size, dtype=torch.double, device=self.device))

    def forward(self, x, h, c):
        # print(f"x shape: {x.shape}")
        x_refined = self.refinement_network(x.float())
        # x_refined.shape[0]: keep the batch size unchanged. Can't use self.batch_size, since the last batch may have
        # size < self.batch_size
        # x_refined = x_refined.reshape(x_refined.shape[0], self.sequence_length, -1)
        x_lstm, (h_n, c_n) = self.lstm(x_refined.float())
        # x_extended = torch.cat((x_lstm, x_refined), dim=2)
        x_flatten = torch.flatten(x_lstm, start_dim=1).to(self.device)
        y_pred = self.fc(x_flatten)
        return y_pred


class PreProcessNet(Module):
    def __init__(self, args):
        super(PreProcessNet, self).__init__()
        self.input_size = 132299
        self.in_channels = 2
        self.conv1_1 = Conv1d(in_channels=self.in_channels, out_channels=2*self.in_channels, kernel_size=3, dilation=1, device=args.device)
        self.conv1_1_out_size = check_conv1d_out_dim(self.input_size, 3, 0, 1, 1)
        self.conv1_2 = Conv1d(in_channels=2*self.in_channels, out_channels=8*self.in_channels, kernel_size=7, dilation=3, device=args.device)
        self.conv1_2_out_size = check_conv1d_out_dim(self.conv1_1_out_size, 7, 0, 1, 3)
        self.down_sampling_1 = DownSamplingBLock(args, channels=8*self.in_channels, dilation=1, stride=2)
        self.down_sampling_1_out_size = check_conv1d_out_dim(self.conv1_2_out_size, 3, 0, 2, 1)
        self.down_sampling_2 = DownSamplingBLock(args, channels=8*self.in_channels, dilation=3, stride=2)
        self.down_sampling_2_out_size = check_conv1d_out_dim(self.down_sampling_1_out_size, 3, 0, 2, 3)
        self.down_sampling_3 = DownSamplingBLock(args, channels=8*self.in_channels, dilation=9, stride=2)
        self.down_sampling_3_out_size = check_conv1d_out_dim(self.down_sampling_2_out_size, 3, 0, 2, 9)
        self.conv1_3 = Conv1d(in_channels=8*self.in_channels, out_channels=args.sequence_length, kernel_size=7, stride=4, dilation=2)
        self.lstm_input_size = check_conv1d_out_dim(self.down_sampling_3_out_size, 7, 0, 4, 2)

        self.down_sampling_net = Sequential(
            self.conv1_1,
            self.conv1_2,
            self.down_sampling_1,
            self.down_sampling_2,
            self.down_sampling_3,
            self.conv1_3
        )
        self.lstm = LSTM(
            input_size=self.lstm_input_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            device=args.device,
            batch_first=True,
            bidirectional=True,
            num_layers=args.num_layers)
        # todo - add attention block!!!

    def forward(self, x):
        x_down_sampled = self.down_sampling_net(x)
        x_lstm, (h_lstm, c_lstm) = self.lstm(x_down_sampled)
        return x_lstm


class ClassificationNet(Module):
    def __init__(self, args):
        super(ClassificationNet, self).__init__()
        self.linear_1 = Linear(in_features=2*args.sequence_length*args.hidden_size, out_features=250, device=args.device)
        self.batch_1 = BatchNorm1d(num_features=250, device=args.device)
        self.intro = Sequential(
            self.linear_1,
            self.batch_1
        )
        self.linear_2_left = Linear(in_features=250, out_features=100, device=args.device)
        self.batch_2_left = BatchNorm1d(num_features=100, device=args.device)
        self.left = Sequential(
            self.linear_2_left,
            self.batch_2_left
        )
        self.linear_2_right = Linear(in_features=250, out_features=100, device=args.device)
        self.batch_2_right = BatchNorm1d(num_features=100, device=args.device)
        self.right = Sequential(
            self.linear_2_right,
            self.batch_2_right
        )
        self.linear_3 = Linear(in_features=100, out_features=args.n_classes, device=args.device)
        self.end = Sequential(
            self.linear_3,
        )
        self.softmax = Softmax()

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x_intro = self.intro(x)
        x_left = self.left(x_intro)
        x_right = self.right(x_intro)
        x_end = self.end(x_left + x_right)
        return self.softmax(x_end)


class DownSamplingBLock(Module):
    def __init__(self, args, channels, dilation, stride):
        super(DownSamplingBLock, self).__init__()
        self.kernel_size = 3
        self.dilation = dilation
        self.stride = stride
        self.device = args.device
        self.left = Sequential(MaxPool1d(kernel_size=self.kernel_size, dilation=self.dilation, stride=stride),
                               ReLU())
        self.right = Sequential(Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, device=self.device),
                                Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, dilation=self.dilation, stride=stride, device=self.device))

    def forward(self, x):
        x_left = self.left(x)
        x_right = self.right(x)
        return x_left + x_right


class InstrumentClassificationNet(Module):
    def __init__(self, args):
        super(InstrumentClassificationNet, self).__init__()
        self.preprocessing_net = PreProcessNet(args)
        self.classification_net = ClassificationNet(args)

    def forward(self, x):
        x_pre = self.preprocessing_net(x)
        y_pred = self.classification_net(x_pre)
        return y_pred