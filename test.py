import torch.nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import TestDataset
from utils.utils import upload_args
from models.lstm_model import  InstrumentClassificationNet
from sklearn.metrics import f1_score, confusion_matrix
from utils.plot import plot_confusion_matrix, save_confusion_matrix


def test():
    args = upload_args("configuration.json")
    test_set = TestDataset(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    checkpoint_args = checkpoint['args']
    setattr(checkpoint_args, "device", device)
    if checkpoint_args.train == 'lstm':
        model = InstrumentClassificationNet(checkpoint_args)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise NotImplementedError()
    prova = test_set[0]
    dataloader_test = DataLoader(test_set, batch_size=checkpoint_args.batch_size)
    y_true_all = np.zeros((len(test_set), 1))
    y_pred_all = np.zeros((len(test_set), 1))
    with torch.no_grad():
        for batch_idx, (x_test, y_true) in enumerate(dataloader_test):
            x_test = x_test.to(checkpoint_args.device).float()
            y_true = y_true.to(checkpoint_args.device).float()
            y_pred = model(x_test)
            y_pred = y_pred.to(checkpoint_args.device)
            start_idx = batch_idx * checkpoint_args.batch_size
            y_true_all[start_idx:start_idx + x_test.shape[0]] = np.argmax(
                torch.squeeze(y_true, dim=1).detach().cpu().numpy(), axis=-1).reshape(-1, 1)
            y_pred_all[start_idx:start_idx + x_test.shape[0]] = np.argmax(y_pred.detach().cpu().numpy(),
                                                                          axis=-1).reshape(-1, 1)

        test_f1_score = f1_score(y_true=y_true_all, y_pred=y_pred_all, average='micro')
    print(f"Test Micro f1 score: {test_f1_score}")
    # cm = confusion_matrix(y_pred=y_pred_all, y_true=y_true_all)
    save_confusion_matrix(y_true=y_true_all, y_pred=y_pred_all, classes=list(test_set.ohe.categories_[0]),
                          name_method="LSTMxTest_Dataset_1_inst")
    # plot_confusion_matrix(cm, classes=['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi'], title="Test Dataset Metrics (1 Instrument)")


if __name__=='__main__':
    test()