from torch.nn import Module
from torch.nn.modules import LSTM, Linear, Softmax, Conv1d
import torch

# TODO - this model works with input sequences of fixed length!
# modify the Dataset code in order to handle input of different length (normal case: 3 seconds audio)

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
        out1 = self.check_conv1d_out_dim(132300, self.kernel_1, self.padding_1, self.stride_1, self.dilation_1)
        self.conv1d_1 = Conv1d(in_channels=2, out_channels=4, kernel_size=self.kernel_1, stride=self.stride_1, padding=self.padding_1, dilation=self.dilation_1)

        self.kernel_2, self.stride_2, self.padding_2, self.dilation_2 = 12, 4, 4, 1
        out2 = self.check_conv1d_out_dim(out1, self.kernel_2, self.padding_2, self.stride_2, self.dilation_2)
        self.conv1d_2 = Conv1d(in_channels=4, out_channels=8, kernel_size=self.kernel_2, stride=self.stride_2, padding=self.padding_2, dilation=self.dilation_2)

        self.kernel_3, self.stride_3, self.padding_3, self.dilation_3 = 25, 25, 0, 1
        self.refinement_output_size = self.check_conv1d_out_dim(out2, self.kernel_3, self.padding_3, self.stride_3, self.dilation_3)
        self.conv1d_3 = Conv1d(in_channels=8, out_channels=self.sequence_length, kernel_size=self.kernel_3, stride=self.stride_3, dilation=self.dilation_3, padding=self.padding_3)

        self.refinement_network = torch.nn.Sequential(self.conv1d_1, self.conv1d_2)
        self.lstm = LSTM(
            input_size=self.refinement_output_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2,
            batch_first=True,
            device=self.device
        )
        self.fc = torch.nn.Sequential(Linear(self.sequence_length * (self.hidden_size + self.refinement_output_size), 200, device=self.device),
                                      Linear(200, 100, device=self.device),
                                      Linear(100, self.n_classes, device=self.device),
                                      Softmax()
                                      )


    def check_conv1d_out_dim(self, in_size, kernel, padding, stride, dilation):
        conv1d_out_size = (in_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
        assert conv1d_out_size % 1 == 0, "Something went wront. The output of conv1d should have an integer dimension. Not float"
        return int(conv1d_out_size)

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
        x_refined_2 = self.conv1d_3(x_refined.float())
        # print(f"x_refined shape: {x_refined.shape}")
        # x_refined.shape[0]: keep the batch size unchanged. Can't use self.batch_size, since the last batch may have
        # size < self.batch_size
        # x_refined = x_refined.reshape(x_refined.shape[0], self.sequence_length, -1)
        x_lstm, (h_n, c_n) = self.lstm(x_refined_2.float())
        x_extended = torch.cat((x_lstm, x_refined_2), dim=2)
        x_flatten = torch.flatten(x_extended, start_dim=1).to(self.device)
        y_pred = self.fc(x_flatten)
        return y_pred
