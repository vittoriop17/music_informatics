import matplotlib.pyplot as plt
# import torchaudio
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
from sklearn.metrics import confusion_matrix, f1_score
import itertools

from models import lstm_model
from utils.utils import accuracy
from utils.dataset import MusicDataset, stratified_split, TestDataset


# def save_audio_specs(dataset_path, specs_x_inst=5):
#     save_dir = "..\\data_exploration"
#     try:
#         os.makedirs(save_dir)
#     except Exception as e:
#         pass
#     for (root, dirs, files) in os.walk(dataset_path, topdown=True):
#         base_class = os.path.basename(root)
#         current_tot_files = 0
#         try:
#             os.makedirs(os.path.join(save_dir, base_class))
#         except:
#             pass
#         for file in files:
#             if not file.endswith(".wav"):
#                 continue
#             if current_tot_files >= specs_x_inst:
#                 break
#             current_tot_files += 1
#             plot_spec(os.path.join(root, file), base_class)
#             new_name = str(file).replace(".wav", ".png")
#             plt.savefig(os.path.join(save_dir, base_class, new_name))
#             plt.close()

#
# def plot_spec(audio_path, audio_class):
#     waveform, sample_rate = torchaudio.load(audio_path)
#     specgram = torchaudio.transforms.MelSpectrogram()(waveform)
#     specgram = np.mean(specgram.numpy(), axis=0, keepdims=True)
#     plt.figure()
#     plt.imshow(np.log2(specgram[0, :, :]), cmap='hot')
#     plt.title(audio_class)


def save_confusion_matrix(y_true: np.array, y_pred: np.array, classes: list, name_method: str):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    plot_confusion_matrix(cm, classes, title=str.upper(name_method)+f", micro F1-score: {micro_f1:.3f}")
    plt.savefig(name_method+"confusion_mat.png")


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__=='__main__':
    args = utils.upload_args("../configuration.json")
    # save_audio_specs(args.dataset_path)
    y_true = np.load('..\\data\\y_test_svm_strat.npy')
    y_pred= np.load('..\\data\\y_pred_svm_strat.npy')
    classes = ['cel','cla','flu','gac','gel','org','pia','sax','tru','vio','voi']
    save_confusion_matrix(y_true, y_pred, classes, "svm")


def confusion_matrix_from_existing_model(args, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    args_checkpoint = checkpoint['args']
    setattr(args_checkpoint, "device", "cuda" if torch.cuda.is_available() else "cpu")
    setattr(args_checkpoint, "dataset_path", args.dataset_path)
    if args_checkpoint.train == 'lstm':
        model = lstm_model.InstrumentClassificationNet(args_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        ds = MusicDataset(args=args_checkpoint)
        ds_train = MusicDataset(args)
        ds_test = TestDataset(args, args.test_dataset_path)
        batch_size = 64
        test_data_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
        train_data_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        classes = ds.ohe.get_feature_names()
        classes = [classs.replace("x0_", "") for classs in classes]
    else:
        return
    with torch.no_grad():
        for i in range(2):
            data_loader = train_data_loader if i == 0 else test_data_loader
            # data_loader = test_data_loader
            name = "train set" if i == 0 else "test set"
            y_true = np.zeros((len(ds_train), 1)) if i==0 else np.zeros((len(ds_test), 1))
            y_pred = np.zeros((len(ds_train), 1)) if i==0 else np.zeros((len(ds_test), 1))
            y_pred_all = np.zeros((len(ds_train), args.n_classes)) if i==0 else np.zeros((len(ds_test), args.n_classes))

            for idx, (batch, y_true_batch) in enumerate(data_loader):
                batch_size = batch.shape[0]
                batch = torch.squeeze(batch, dim=1).float()
                y_true_batch = torch.squeeze(y_true_batch, dim=1).float()
                start_idx = idx * batch.shape[0]
                y_true[start_idx:start_idx + batch_size] = np.argmax(y_true_batch.detach().numpy(), axis=-1).reshape(-1,
                                                                                                                     1).astype(
                    np.int64)
                pred_prob = model(batch)
                y_pred[start_idx:start_idx + batch_size] = np.argmax(pred_prob.detach().numpy(), axis=-1).reshape(-1,
                                                                                                                     1).astype(
                    np.int64)
                y_pred_all[start_idx:start_idx + batch_size, :] = pred_prob

                # x = torch.squeeze(x.to(args.device).float(), dim=1)
                # y_true = y_true.to(args.device).float()

            topk_train_accuracies = accuracy(torch.tensor(y_pred_all), torch.tensor(y_true), topk=(1, 2, 3))
            print(f"name: {name}, accuracies: {topk_train_accuracies}")
            save_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=classes, name_method=name)
            plt.close()

