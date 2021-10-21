from torch.utils.data import DataLoader
from utils.dataset import MusicDataset
from models import lstm_model
from torch import nn, optim
import torch

# TODO - think about a train_wrapper (automatically select the model to be trained)

def train_lstm(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setattr(args, "device", device)
    ds = MusicDataset(args=args)
    dataloader = DataLoader(ds, args.batch_size, shuffle=True)
    model = lstm_model.LSTM_model(args)
    model = model.float()
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        state_h, state_c = model.init_state()
        for batch, (x, y_true) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(x, state_h, state_c)
            loss = criterion(y_pred, y_true.reshape(-1, args.n_classes))
            loss.backward()
            optimizer.step()
            print({'epoch': epoch, 'batch': batch, 'loss': loss.item() })

