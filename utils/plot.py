import matplotlib.pyplot as plt
import torchaudio
import os
import numpy as np
import utils

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

if __name__=='__main__':
    args = utils.upload_args("../configuration.json")
    save_audio_specs(args.dataset_path)

