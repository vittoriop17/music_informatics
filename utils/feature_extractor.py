import os
import librosa
import pydub
import numpy as np
import string, re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pydub import AudioSegment
import h5py


def read_audio(audio_path):
    wav_audio = AudioSegment.from_file(audio_path, format="wav")
    wav_audio = wav_audio.set_channels(1)           # set a single channel
    sample_rate = wav_audio.frame_rate              # extract the fs
    samples_audio = np.array(wav_audio.get_array_of_samples(), dtype=np.float64)  # transform the audio in a np.array
    return samples_audio, sample_rate


def read_label(audio_name, one_hot= False):
    label = audio_name.split('[')[1].split(']')[0]
    label_dict = {
        'cel': 0,
        'cla': 1,
        'flu': 2,
        'gac': 3,
        'gel': 4,
        'org': 5,
        'pia': 6,
        'sax': 7,
        'tru': 8,
        'vio': 9,
        'voi': 10
    }
    label = label_dict[label]
    if one_hot:
        label_1h = np.zeros(11)
        label_1h[label] = 1
        label = label_1h
    return label


def dataset_preprocessor(input_path, normalize_amplitude, spectrogram, normalize_features = False, class_set = None, output_path = 'C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\music_informatics\\data'):
    '''

    :param input_path: path of the audio files
    :param normalize: choose to normalize the audio segment

    :return:
    '''

    class_list = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

    # count the total number of file we have.
    # Useful to pre-allocate the dataset array.
    num_files = 0
    if class_set is None:
        num_files = sum([len(files) for r, d, files in os.walk(input_path)])
        class_set = class_list
    else:
        for class_instr in class_set:
            num_files += sum([len(files) for r, d, files in os.walk(os.path.join(input_path, class_instr))])



    instr_folder = os.listdir(input_path)
    # number of features: 25
    data = np.empty((num_files, 25, 65), dtype=np.float32)

    #data_labels = np.empty((num_files, 11), dtype= bool)        # num. classes = 11
    data_labels = np.empty(num_files, dtype= np.int32)

    index = 0
    for instr in instr_folder:
        instr_path = os.path.join(input_path, instr)

        if instr in class_set:
            for audio_name in os.listdir(instr_path):
                audio_path = os.path.join(instr_path, audio_name)   # retrieve the audio file path
                samples_audio, sample_rate = read_audio(audio_path)

                if normalize_amplitude:
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    samples_audio = scaler.fit_transform(samples_audio.reshape(-1, 1)).reshape(-1, )
                else:
                    pass
                # set here params for the feature extraction
                features_args = {"y":samples_audio, "sr":sample_rate, "hop_length": 8192 // 4, "n_fft":8192}

                if spectrogram:
                    print('extracting mel spectrogram for audio: ' + audio_name)
                    data[index] = librosa.amplitude_to_db(librosa.feature.melspectrogram(samples_audio, sample_rate))
                else:
                    if normalize_features:
                        print('extracting normalized features for audio: '+audio_name)
                        scaler = StandardScaler() # set here the scaler

                        data[index, 0] = \
                            scaler.fit_transform(
                                librosa.feature.spectral_centroid(**features_args)[0].reshape(-1,1)).reshape(-1,)
                        data[index, 1] = \
                            scaler.fit_transform(
                                librosa.feature.spectral_bandwidth(**features_args)[0].reshape(-1,1)).reshape(-1,)
                        data[index, 2] = \
                            scaler.fit_transform(
                                librosa.feature.spectral_rolloff(**features_args)[0].reshape(-1,1)).reshape(-1,)
                        data[index, 3] = \
                            scaler.fit_transform(
                                librosa.feature.zero_crossing_rate(samples_audio, hop_length=5012 // 2)[0].reshape(-1,1)).reshape(-1,)
                        data[index, 4] = \
                            scaler.fit_transform(
                                librosa.feature.rms(samples_audio, hop_length=5012 // 2)[0].reshape(-1,1)).reshape(-1,)
                        data[index, 5:25] = \
                            scaler.fit_transform(
                                librosa.feature.mfcc(**features_args, n_mfcc=20))

                    else:
                        print('extracting features for audio: '+audio_name)
                        data[index, 0] = librosa.feature.spectral_centroid(**features_args)[0]
                        data[index, 1] = librosa.feature.spectral_bandwidth(**features_args)[0]
                        data[index, 2] = librosa.feature.spectral_rolloff(**features_args)[0]
                        data[index, 3] = librosa.feature.zero_crossing_rate(samples_audio, hop_length= 8192//4)[0]
                        data[index, 4] = librosa.feature.rms(samples_audio, hop_length= 8192//4)[0]
                        data[index, 5:25] = librosa.feature.mfcc(**features_args, n_mfcc=20)

                data_labels[index] = read_label(audio_name)         # save the label of the sample

                index +=1



    print('saving data to .npy file...')
    np.save(os.path.join(output_path,'out_dataset_def.npy'), data)

    print('Done.')

    print('saving labels to .npy file...')
    np.save(os.path.join(output_path, 'out_labels_def.npy'), data_labels)

    print('Done.')

    return data, data_labels










dataset_preprocessor('C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\IRMAS-TrainingData', True, False)