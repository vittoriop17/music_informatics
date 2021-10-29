from torch.utils.data import DataLoader, random_split
from utils.dataset import MusicDataset, stratified_split, TestDataset
from models import lstm_model, tcn
from torch import nn, optim
import torch
import os
import numpy as np
from sklearn.metrics import f1_score
from utils.plot import save_confusion_matrix
from utils.utils import upload_args
from utils.feature_extractor import dataset_preprocessor
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import precision_score, f1_score


def train(args):
    options = ["lstm", "svm", "cnn", "tcn"]
    choice = getattr(args, "train", None)
    if choice is None:
        raise Exception(f"Argument not found! Please insert a valid argument for 'train' option {options}")
    if choice not in options:
        raise Exception(
            f"Invalid argument! 'train' option must be associated to one of the following arguments: {options}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setattr(args, "device", device)
    torch.device(device)
    model, ds_train, ds_test, criterion = None, None, None, None
    if choice == "lstm":
        model = lstm_model.InstrumentClassificationNet(args)
        ds = MusicDataset(args=args)
        ds_train, ds_test = stratified_split(ds, args, 0.8) \
            if not getattr(args, "only_train", False) \
            else (ds, TestDataset(args, root_path=args.test_dataset_path))
        criterion = nn.BCEWithLogitsLoss()
        print("\t TRAINING LSTM MODEL...")
        model, history = train_model(args, model, ds_train, ds_test, criterion)
    if choice == "tcn":
        model = tcn.ClassificationTCN(args)
        ds = MusicDataset(args)
        ds_train, ds_test = stratified_split(ds, args, 0.8)
        criterion = nn.BCEWithLogitsLoss()
        print("\t TRAINING TCN MODEL...")
        model, history = train_model(args, model, ds_train, ds_test, criterion)
    if choice == "svm":
        print("\t TRAINING SVM...")
        dataset_path = args.dataset_path
        # generate the dataset, if it is not already there
        if len(os.listdir(args.features_dataset_path)) == 0:
            data, labels = dataset_preprocessor(dataset_path, normalize_amplitude=True, normalize_features=False,
                                                output_path=args.features_dataset_path)
        else:
            data = np.load(os.path.join(args.features_dataset_path, 'out_dataset.npy'))
            labels = np.load(os.path.join(args.features_dataset_path, 'out_labels.npy'))
        # train the svm model
        precision_per_classes, f1_score = train_svm(data, labels)
    if choice == "cnn":
        model = None
        ds = MusicDataset(args=args)
        ds.__getitem__(0)
        ds_train, ds_test = stratified_split(ds, args, 0.8)
        # len_ds = len(ds)
        # len_ds_train = int(0.7 * len_ds)
        # ds_train, ds_test = random_split(ds, [len_ds_train, len_ds - len_ds_train], torch.Generator().manual_seed(42))
        criterion = nn.BCEWithLogitsLoss()
        print("\t TRAINING CNN MODEL")
        raise NotImplementedError("Implement CNN network")
        model, history = train_model(args, model, ds_train, ds_test, criterion)


def load_existing_model(model, optimizer, checkpoint_path):
    try:
        print(f"Trying to load existing model from checkpoint @ {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("...existing model loaded")
        max_test_f1_score = getattr(checkpoint_path, "max_test_f1_score", 0)
    except Exception as e:
        print("...loading failed")
        print(f"During loading the existing model, the following exception occured: \n{e}")
        print("The execution will continue anyway")
        max_test_f1_score = 0
    return max_test_f1_score


def confusion_matrix_from_existing_model(args, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    args_checkpoint = checkpoint['args']
    setattr(args_checkpoint, "device", "cuda" if torch.cuda.is_available() else "cpu")
    setattr(args_checkpoint, "dataset_path", args.dataset_path)
    if args_checkpoint.train == 'lstm':
        model = lstm_model.InstrumentClassificationNet(args_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        ds = MusicDataset(args=args_checkpoint)
        _, ds_test = stratified_split(ds, args_checkpoint, 0.8)
        batch_size = 64
        data_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
        classes = ds.ohe.get_feature_names()
        classes = [classs.replace("x0_", "") for classs in classes]
    else:
        return
    y_true = np.zeros((len(ds_test), 1))
    y_pred = np.zeros((len(ds_test), 1))
    for idx, (batch, y_true_batch) in enumerate(data_loader):
        batch_size = batch.shape[0]
        start_idx = idx * batch.shape[0]
        y_true[start_idx:start_idx + batch_size] = np.argmax(y_true_batch.detach().numpy(), axis=-1).reshape(-1,
                                                                                                             1).astype(
            np.int64)
        y_pred[start_idx:start_idx + batch_size] = np.argmax(model(batch).detach().numpy(), axis=-1).reshape(-1,
                                                                                                             1).astype(
            np.int64)
    save_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=classes, name_method=args_checkpoint.train)


def train_model(args, model, ds_train, ds_test, criterion):
    checkpoint_path = args.checkpoint_path if getattr(args, "checkpoint_path", None) is not None else str(
        "./checkpoint.pt")
    train_dataloader = DataLoader(ds_train, args.batch_size, shuffle=True)
    test_dataloader = DataLoader(ds_test, args.batch_size, shuffle=True)
    model = model.to(args.device)
    model = model.float()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    max_test_f1_score = 0
    if getattr(args, "load_model", False):
        max_test_f1_score = load_existing_model(model, optimizer, checkpoint_path)
    history = dict(train=[], train_f1=[], eval_f1=[], eval=[], max_test_f1_score=max_test_f1_score)
    for epoch in range(args.epochs):
        model = model.train()
        epoch_train_losses = list()
        y_true_all = np.zeros((len(ds_train), 1), dtype=np.int64)
        y_pred_all = y_true_all.copy()
        for batch_idx, (x, y_true) in enumerate(train_dataloader):
            x = torch.squeeze(x.to(args.device).float(), dim=1)
            y_true = y_true.to(args.device).float()
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred = y_pred.to(args.device)
            loss = criterion(y_pred, y_true.reshape(-1, args.n_classes))
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
            # store ground truth and predicted targets. Necessary for f1 score of the whole TRAIN dataset
            start_idx = batch_idx * args.batch_size
            y_true_all[start_idx:start_idx + x.shape[0]] = np.argmax(
                torch.squeeze(y_true, dim=1).detach().cpu().numpy(), axis=-1).reshape(-1, 1)
            y_pred_all[start_idx:start_idx + x.shape[0]] = np.argmax(y_pred.detach().cpu().numpy(), axis=-1).reshape(-1,
                                                                                                                     1)

        mean_train_loss = np.mean(epoch_train_losses)
        train_f1_score = f1_score(y_true=y_true_all, y_pred=y_pred_all, average="micro")
        history['train'].append(mean_train_loss)
        history['train_f1'].append(train_f1_score)
        epoch_test_losses = list()
        y_true_all = np.zeros((len(ds_test), 1), dtype=np.int64)
        y_pred_all = y_true_all.copy()
        model = model.eval()
        with torch.no_grad():
            for batch_idx, (x_test, y_true) in enumerate(test_dataloader):
                x_test = torch.squeeze(x_test.to(args.device).float(), dim=1)
                y_true = y_true.to(args.device).float()
                y_pred = model(x_test)
                y_pred = y_pred.to(args.device)
                test_loss = criterion(y_pred, y_true.reshape(-1, args.n_classes))
                # print({'epoch': epoch, 'batch_num': batch_num, 'loss': loss.item()})
                # store ground truth and predicted targets. Necessary for f1 score of the whole TEST dataset
                start_idx = batch_idx * args.batch_size
                y_true_all[start_idx:start_idx + x_test.shape[0]] = np.argmax(
                    torch.squeeze(y_true, dim=1).detach().cpu().numpy(), axis=-1).reshape(-1, 1)
                y_pred_all[start_idx:start_idx + x_test.shape[0]] = np.argmax(y_pred.detach().cpu().numpy(),
                                                                              axis=-1).reshape(-1, 1)
                epoch_test_losses.append(test_loss.item())

            mean_test_loss = np.mean(epoch_test_losses)
            eval_f1_score = f1_score(y_true=y_true_all, y_pred=y_pred_all, average='micro')
            history['eval'].append(mean_test_loss)
            history['eval_f1'].append(eval_f1_score)

        print(f"Epoch: {epoch}, \n train loss & f1-score: {mean_train_loss}, {train_f1_score}, "
              f"\t test loss & f1-score: {mean_test_loss}, {eval_f1_score}")
        if eval_f1_score > history['max_test_f1_score']:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'max_test_f1_score': eval_f1_score,
                'args': args,
                'y_true': y_true_all,
                'y_pred': y_pred_all
            }, checkpoint_path)
            history['max_test_f1_score'] = eval_f1_score
    return model, history


def train_svm(data, labels, stratified=True):
    flattened_data = np.array([data_matrix.flatten() for data_matrix in data])

    if stratified:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
        for train_idx, test_idx in sss.split(flattened_data, labels):
            X_train, X_test = flattened_data[train_idx], flattened_data[test_idx]
            Y_train, Y_test = labels[train_idx], labels[test_idx]
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(flattened_data, labels, test_size=0.2, random_state=7)

    model = LinearSVC(C=0.0000005, dual=False, verbose=1, class_weight='balanced')
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    print('precision score (for each of the 11 classes): ')
    print(precision_score(Y_test, y_pred, average=None))
    print('f1 score: ')
    print(f1_score(Y_test, y_pred, average='weighted'))

    print('finished')

    return precision_score(Y_test, y_pred, average=None), f1_score(Y_test, y_pred, average='weighted')


if __name__ == '__main__':
    args = upload_args("configuration.json")
    setattr(args, "device", "cpu")
    # checkpoint_path = "C:\\Users\\vitto\\Downloads\\checkpoint (5).pt"
    checkpoint_path = "D:\\UNIVERSITA\\KTH\Semestre 1\\Music Informatics\\Labs\\Final_project\\checkpoints\\checkpoint.pt"
    # confusion_matrix_from_existing_model(args, checkpoint_path=checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    y_true = checkpoint["y_true"]
    y_pred = checkpoint["y_pred"]
    classes = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    save_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=classes, name_method="lstm")
