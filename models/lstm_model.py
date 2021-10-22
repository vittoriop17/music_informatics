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
        self.sequence_length = args.sequence_length
        self.batch_size = args.batch_size
        self.n_classes = args.n_classes
        # conv1d: such that it takes 22ms of audio per time
        self.conv1d = Conv1d(in_channels=1, out_channels=1, kernel_size=970, dilation=120)
        self.lstm = LSTM(
            input_size=self.input_size,
            hidden_size=self.input_size,
            num_layers=self.num_layers,
            dropout=0.2,
            batch_first=True,
            device=self.device
        )
        self.fc = torch.nn.Sequential(Linear(self.sequence_length * self.input_size, 200, device=self.device),
                                      Linear(200, 100, device=self.device),
                                      Linear(100, self.n_classes, device=self.device),
                                      Softmax()
                                      )

    def check_args(self, args):
        assert hasattr(args, "num_layers"), "Argument 'num_layers' not found!"
        assert hasattr(args, "input_size"), "Argument 'input_size' not found!"
        assert hasattr(args, "num_layers"), "Argument 'num_layers' not found!"
        assert hasattr(args, "sequence_length"), "Argument 'sequence_length' not found!"
        assert hasattr(args, "device"), "Argument 'device' not found!"
        assert hasattr(args, "n_classes"), "Argument 'n_classes' not found!"

    def init_state(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.input_size, dtype=torch.double, device=self.device),
                torch.zeros(self.num_layers, self.batch_size, self.input_size, dtype=torch.double, device=self.device))

    def forward(self, x, h, c):
        x = x.unsqueeze(1)
        x_conv = self.conv1d(x.float())
        x_conv = x_conv.reshape(self.batch_size, self.sequence_length, -1)
        print(f"x_conv after reshape: {x_conv.shape}. It should be (batch_size, sequence_length, input_size) ---> {self.batch_size, self.sequence_length, self.input_size}")
        x_lstm, (h_n, c_n) = self.lstm(x_conv.float())
        x_lstm = x_lstm.reshape(-1, self.sequence_length * self.input_size)
        y_pred = self.fc(x_lstm)
        return y_pred
