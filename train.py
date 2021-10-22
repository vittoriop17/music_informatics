from torch.utils.data import DataLoader, random_split
from utils.dataset import MusicDataset
from models import lstm_model
from torch import nn, optim
import torch
import numpy as np

# TODO - think about a train_wrapper (automatically select the model to be trained)

def train_lstm(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)
    setattr(args, "device", device)
    ds = MusicDataset(args=args)
    len_ds = len(ds)
    len_ds_train = int(0.7 * len_ds)
    ds_train, ds_test = random_split(ds, [len_ds_train, len_ds - len_ds_train], torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(ds_train, args.batch_size, shuffle=True)
    test_dataloader = DataLoader(ds_test, args.batch_size, shuffle=True)
    model = lstm_model.LSTM_model(args)
    model = model.float()
    model = model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = dict(train=[], val=[])

    for epoch in range(args.epochs):
        model = model.train()
        state_h, state_c = model.init_state()
        epoch_train_losses = list()
        for x, y_true in train_dataloader:
            x = x.to(device)
            y_true = y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(x, state_h, state_c)
            y_pred = y_pred.to(device)
            loss = criterion(y_pred, y_true.reshape(-1, args.n_classes))
            loss.backward()
            optimizer.step()
            # print({'epoch': epoch, 'batch_num': batch_num, 'loss': loss.item()})
            epoch_train_losses.append(loss.item())
        mean_train_loss = np.mean(epoch_train_losses)
        history['train'].append((mean_train_loss))
        epoch_test_losses = list()
        model = model.eval()
        with torch.no_grad():
            for x_test, y_true in test_dataloader:
                x_test = x_test.to(device)
                y_true = y_true.to(device)
                y_pred = model(x_test, state_h, state_c)
                y_pred = y_pred.to(device)
                loss = criterion(y_pred, y_true.reshape(-1, args.n_classes))
                # print({'epoch': epoch, 'batch_num': batch_num, 'loss': loss.item()})
                epoch_test_losses.append(loss.item())
            mean_test_loss = np.mean(epoch_test_losses)
            history['val'].append(mean_test_loss)
        print(f"Epoch: {epoch}, \t train loss: {mean_train_loss}, \t test loss: {mean_test_loss}")
