from torch.nn import Module
from torch.nn.modules import LSTM, Linear, Softmax
import torch

class LSTM_model(Module):
    def __init__(self, args):
        super(LSTM_model, self).__init__()
        self.check_args(args)
        self.device = args.device
        self.num_layers = args.num_layers
        self.input_size = args.input_size
        self.sequence_length = args.sequence_length
        self.lstm = LSTM(
            input_size=self.input_size,
            hidden_size=self.input_size,
            num_layers=self.num_layers,
            dropout=0.2,
            batch_first=True
        )
        self.fc = torch.nn.Sequential(Linear(self.sequence_length * self.input_size, 200),
                                      Linear(200, 100),
                                      Linear(100, 100),
                                      Softmax()
                                      )

    def check_args(self, args):
        assert hasattr(args, "num_layers"), "Argument 'num_layers' not found!"
        assert hasattr(args, "input_size"), "Argument 'input_size' not found!"
        assert hasattr(args, "num_layers"), "Argument 'num_layers' not found!"
        assert hasattr(args, "sequence_length"), "Argument 'sequence_length' not found!"
        assert hasattr(args, "device"), "Argument 'device' not found!"

    def init_state(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.input_size),
                torch.zeros(self.num_layers, self.batch_size, self.input_size))

    def forward(self, x, h, c):
        x, (h_n, c_n) = self.lstm(x, (h, c))
        y_pred = self.fc(x)
