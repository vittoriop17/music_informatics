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
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
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
        y_true_all = torch.zeros((len(ds_train), 1), requires_grad=False)
        # y_pred_all = y_true_all.copy()
        y_pred_all = torch.zeros((len(ds_train), args.n_classes), requires_grad=False)
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
            y_true_all[start_idx:start_idx + x.shape[0]] = torch.argmax(torch.squeeze(y_true, dim=1), dim=-1).reshape(-1, 1)
            # y_pred_all[start_idx:start_idx + x.shape[0]] = np.argmax(y_pred.detach().cpu().numpy(), axis=-1).reshape(-1, 1)
            y_pred_all[start_idx:start_idx + x.shape[0], :] = y_pred

        topk_train_accuracies = accuracy(y_pred_all, y_true_all, topk=(1, 2, 3))
        mean_train_loss = np.mean(epoch_train_losses)
        train_f1_score = f1_score(y_true=y_true_all,
                                  y_pred=np.argmax(y_pred_all.detach().cpu().numpy(), axis=-1),
                                  average="micro")
        history['train'].append(mean_train_loss)
        history['train_f1'].append(train_f1_score)
        epoch_test_losses = list()
        y_true_all = torch.zeros((len(ds_test), 1), requires_grad=False)
        # y_pred_all = y_true_all.copy()
        y_pred_all = torch.zeros((len(ds_test), args.n_classes), requires_grad=False)
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
                y_true_all[start_idx:start_idx + x_test.shape[0]] = torch.argmax(
                    torch.squeeze(y_true, dim=1), dim=-1).reshape(-1, 1)
                y_pred_all[start_idx:start_idx + x_test.shape[0], :] = y_pred
                epoch_test_losses.append(test_loss.item())

            mean_test_loss = np.mean(epoch_test_losses)
            topk_test_accuracies = accuracy(y_pred_all, y_true_all, topk=(1,2,3))
            y_pred_all = np.argmax(y_pred_all.detach().cpu().numpy(), axis=-1).reshape(-1, 1)
            eval_f1_score = f1_score(y_true=y_true_all.detach().cpu().numpy(),
                                     y_pred=y_pred_all,
                                     average='micro')
            history['eval'].append(mean_test_loss)
            history['eval_f1'].append(eval_f1_score)

        print(f"Epoch: {epoch}, \n train: loss {mean_train_loss}, f1-score: {train_f1_score}, "
              f"top1,top2,top3 acc: {list([float(topk) for topk in topk_train_accuracies])}"
              f"\n test: loss {mean_test_loss}, f1-score: {eval_f1_score},"
              f" top1,top2,top3 acc: {list([float(topk) for topk in topk_test_accuracies])}")
        if eval_f1_score > history['max_test_f1_score']:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'max_test_f1_score': eval_f1_score,
                'args': args,
                'y_true': y_true_all,
                'y_pred': y_pred_all,
                'topk_test_acc': topk_test_accuracies,
                'topk_train_acc': topk_train_accuracies
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


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


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
