from torch.utils.data import DataLoader, random_split
from utils.dataset import MusicDataset
from models import lstm_model, ae
from torch import nn, optim
import torch
import numpy as np

# TODO - think about a train_wrapper (automatically select the model to be trained)
def train(args):
    options = ["lstm", "ae", "svm"]
    choice = getattr(args, "train", None)
    if choice is None:
        raise Exception("Argument not found! Please insert a valid argument for 'train' option (lstm, ae, svm)")
    if choice not in options:
        raise Exception("Invalid argument! 'train' option must be associated to one of the following arguments: lstm, ae, svm")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setattr(args, "device", device)
    torch.device(device)
    if choice == "lstm":
        train_lstm(args)
    if choice == "ae":
        train_ae(args)
    if choice == "svm":
        raise NotImplementedError("Implement train_svm")


def train_ae(args):
    ds = MusicDataset(args=args)
    len_ds = len(ds)
    len_ds_train = int(0.7 * len_ds)
    ds_train, ds_test = random_split(ds, [len_ds_train, len_ds - len_ds_train], torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(ds_train, args.batch_size, shuffle=True)
    test_dataloader = DataLoader(ds_test, args.batch_size, shuffle=True)
    model = ae.Music_AE(args)
    model = model.to(args.device)
    model = model.float()
    history = dict(train=[], eval=[])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_losses = list()
        for x, _ in train_dataloader:
            x = x.to(args.device)
            criterion.zero_grad()
            x_pred = model(x.float())
            loss = criterion(x_pred.float(), x.float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        mean_train_loss = np.mean(train_losses)
        history['train'].append(mean_train_loss)
        # start evaluation
        model.eval()
        test_losses = list()
        for x, _ in test_dataloader:
            x = x.to(args.device)
            x_pred = model(x.float())
            loss = criterion(x_pred.float(), x.float())
            test_losses.append(loss.item())
        mean_test_loss = np.mean(test_losses)
        history['eval'].append(mean_test_loss)
        print(f"Epoch: {epoch}, \t train loss: {mean_train_loss}, \t test loss: {mean_test_loss}")


def train_lstm(args):
    ds = MusicDataset(args=args)
    len_ds = len(ds)
    len_ds_train = int(0.7 * len_ds)
    ds_train, ds_test = random_split(ds, [len_ds_train, len_ds - len_ds_train], torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(ds_train, args.batch_size, shuffle=True)
    test_dataloader = DataLoader(ds_test, args.batch_size, shuffle=True)
    model = lstm_model.LSTM_model(args)
    model = model.float()
    model = model.to(args.device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = dict(train=[], val=[])

    for epoch in range(args.epochs):
        model = model.train()
        state_h, state_c = model.init_state()
        epoch_train_losses = list()
        for x, y_true in train_dataloader:
            x = x.to(args.device)
            y_true = y_true.to(args.device)
            optimizer.zero_grad()
            y_pred = model(x, state_h, state_c)
            y_pred = y_pred.to(args.device)
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
                x_test = x_test.to(args.device)
                y_true = y_true.to(args.device)
                y_pred = model(x_test, state_h, state_c)
                y_pred = y_pred.to(args.device)
                loss = criterion(y_pred, y_true.reshape(-1, args.n_classes))
                # print({'epoch': epoch, 'batch_num': batch_num, 'loss': loss.item()})
                epoch_test_losses.append(loss.item())
            mean_test_loss = np.mean(epoch_test_losses)
            history['val'].append(mean_test_loss)
        print(f"Epoch: {epoch}, \t train loss: {mean_train_loss}, \t test loss: {mean_test_loss}")
