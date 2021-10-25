from torch.utils.data import DataLoader, random_split
from utils.dataset import MusicDataset
from models import lstm_model, ae
from torch import nn, optim
import torch
import numpy as np
from sklearn.metrics import f1_score


# TODO - think about a train_wrapper (automatically select the model to be trained)
def train(args):
    options = ["lstm", "ae", "svm", "cnn"]
    choice = getattr(args, "train", None)
    if choice is None:
        raise Exception(f"Argument not found! Please insert a valid argument for 'train' option {options}")
    if choice not in options:
        raise Exception(f"Invalid argument! 'train' option must be associated to one of the following arguments: {options}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setattr(args, "device", device)
    torch.device(device)
    model, ds_train, ds_test, criterion = None, None, None, None
    if choice == "lstm":
        model = lstm_model.InstrumentClassificationNet(args)
        ds = MusicDataset(args=args)
        len_ds = len(ds)
        len_ds_train = int(0.7 * len_ds)
        ds_train, ds_test = random_split(ds, [len_ds_train, len_ds - len_ds_train], torch.Generator().manual_seed(42))
        criterion = nn.BCEWithLogitsLoss()
        print("\t TRAINING LSTM MODEL...")
        model, history = train_model(args, model, ds_train, ds_test, criterion)
    if choice == "ae":
        print("\t TRAINING AUTOENCODER...")
        train_ae(args)
    if choice == "svm":
        raise NotImplementedError("Implement train_svm")
    if choice == "cnn":
        model = None
        ds = MusicDataset(args=args)
        len_ds = len(ds)
        len_ds_train = int(0.7 * len_ds)
        ds_train, ds_test = random_split(ds, [len_ds_train, len_ds - len_ds_train], torch.Generator().manual_seed(42))
        criterion = nn.BCEWithLogitsLoss()
        print("\t TRAINING CNN MODEL")
        model, history = train_model(args, model, ds_train, ds_test, criterion)


def load_existing_model(model, optimizer, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except Exception as e:
        print(f"During loading the existing model, the following exception occured: {e}")
        print("The execution will continue anyway")


def train_model(args, model, ds_train, ds_test, criterion):
    checkpoint_path = args.checkpoint_path if getattr(args, "checkpoint_path") is not None else str("./checkpoint.pt")
    train_dataloader = DataLoader(ds_train, args.batch_size, shuffle=True)
    test_dataloader = DataLoader(ds_test, args.batch_size, shuffle=True)
    model = model.to(args.device)
    model = model.float()
    history = dict(train=[], train_f1=[], eval_f1=[], eval=[])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if getattr(args, "load_model", False):
        load_existing_model(model, optimizer, checkpoint_path)
    for epoch in range(args.epochs):
        model = model.train()
        epoch_train_losses = list()
        epoch_train_f1_scores = list()
        for x, y_true in train_dataloader:
            x = x.to(args.device)
            y_true = y_true.to(args.device)
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred = y_pred.to(args.device)
            loss = criterion(y_pred, y_true.reshape(-1, args.n_classes))
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
            y_true = torch.squeeze(y_true, dim=1)
            epoch_train_f1_scores.append((f1_score(y_true=np.argmax(y_true.detach().cpu().numpy(), axis=-1), y_pred=np.argmax(y_pred.detach().cpu().numpy(), axis=-1), average="macro")))
        mean_train_loss = np.mean(epoch_train_losses)
        mean_train_f1_score = np.mean(epoch_train_f1_scores)
        history['train'].append((mean_train_loss))
        history['train_f1'].append((mean_train_f1_score))
        epoch_test_losses = list()
        epoch_test_f1_scores = list()
        model = model.eval()
        with torch.no_grad():
            for x_test, y_true in test_dataloader:
                x_test = x_test.to(args.device)
                y_true = y_true.to(args.device)
                y_pred = model(x_test)
                y_pred = y_pred.to(args.device)
                test_loss = criterion(y_pred, y_true.reshape(-1, args.n_classes))
                # print({'epoch': epoch, 'batch_num': batch_num, 'loss': loss.item()})
                epoch_test_losses.append(test_loss.item())
                epoch_test_f1_scores.append(f1_score(y_true=np.argmax(y_true.detach().cpu().numpy(), axis=-1), y_pred=np.argmax(y_pred.detach().cpu().numpy(), axis=-1), average="macro"))
            mean_test_loss = np.mean(epoch_test_losses)
            mean_test_f1_score = np.mean(epoch_test_f1_scores)
            history['eval'].append(mean_test_loss)
            history['eval_f1'].append(mean_test_f1_score)

        print(f"Epoch: {epoch}, \n train loss & f1-score: {mean_train_loss}, {mean_train_f1_score}, "
              f"\t test loss & f1-score: {mean_test_loss}, {mean_test_f1_score}")
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, checkpoint_path)
    return model, history


def train_ae(args):
    checkpoint_path = args.checkpoint_path if hasattr(args, "checkpoint_path") else "./checkpoint.pt"
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
            test_loss = criterion(x_pred.float(), x.float())
            test_losses.append(test_loss.item())
        mean_test_loss = np.mean(test_losses)
        history['eval'].append(mean_test_loss)
        print(f"Epoch: {epoch}, \t train loss: {mean_train_loss}, \t test loss: {mean_test_loss}")
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, checkpoint_path)


def train_lstm(args):
    ds = MusicDataset(args=args)
    len_ds = len(ds)
    len_ds_train = int(0.7 * len_ds)
    ds_train, ds_test = random_split(ds, [len_ds_train, len_ds - len_ds_train], torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(ds_train, args.batch_size, shuffle=True)
    test_dataloader = DataLoader(ds_test, args.batch_size, shuffle=True)
    model = lstm_model.InstrumentClassificationNet(args)
    model = model.float()
    model = model.to(args.device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = dict(train=[], val=[])

    for epoch in range(args.epochs):
        model = model.train()
        epoch_train_losses = list()
        for x, y_true in train_dataloader:
            x = x.to(args.device)
            y_true = y_true.to(args.device)
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred = y_pred.to(args.device)
            loss = criterion(y_pred, y_true.reshape(-1, args.n_classes))
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        mean_train_loss = np.mean(epoch_train_losses)
        history['train'].append((mean_train_loss))
        epoch_test_losses = list()
        model = model.eval()
        with torch.no_grad():
            for x_test, y_true in test_dataloader:
                x_test = x_test.to(args.device)
                y_true = y_true.to(args.device)
                y_pred = model(x_test)
                y_pred = y_pred.to(args.device)
                loss = criterion(y_pred, y_true.reshape(-1, args.n_classes))
                # print({'epoch': epoch, 'batch_num': batch_num, 'loss': loss.item()})
                epoch_test_losses.append(loss.item())
            mean_test_loss = np.mean(epoch_test_losses)
            history['val'].append(mean_test_loss)
        print(f"Epoch: {epoch}, \t train loss: {mean_train_loss}, \t test loss: {mean_test_loss}")
