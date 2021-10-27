import matplotlib.pyplot as plt
import torchaudio
import os
import numpy as np
import utils
from sklearn.metrics import confusion_matrix, f1_score
import itertools


def save_audio_specs(dataset_path, specs_x_inst=5):
    save_dir = "..\\data_exploration"
    try:
        os.makedirs(save_dir)
    except Exception as e:
        pass
    for (root, dirs, files) in os.walk(dataset_path, topdown=True):
        base_class = os.path.basename(root)
        current_tot_files = 0
        try:
            os.makedirs(os.path.join(save_dir, base_class))
        except:
            pass
        for file in files:
            if not file.endswith(".wav"):
                continue
            if current_tot_files >= specs_x_inst:
                break
            current_tot_files += 1
            plot_spec(os.path.join(root, file), base_class)
            new_name = str(file).replace(".wav", ".png")
            plt.savefig(os.path.join(save_dir, base_class, new_name))
            plt.close()


def plot_spec(audio_path, audio_class):
    waveform, sample_rate = torchaudio.load(audio_path)
    specgram = torchaudio.transforms.MelSpectrogram()(waveform)
    specgram = np.mean(specgram.numpy(), axis=0, keepdims=True)
    plt.figure()
    plt.imshow(np.log2(specgram[0, :, :]), cmap='hot')
    plt.title(audio_class)


def save_confusion_matrix(y_true, y_pred, classes, name_method):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    plot_confusion_matrix(cm, classes, title=name_method+f", micro F1-score: {micro_f1}")
    plt.savefig(name_method+"confusion_mat.png")


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
    save_audio_specs(args.dataset_path)

