import os
import librosa
import pydub
import numpy as np
import string, re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pydub import AudioSegment
import h5py


def read_audio(audio_path):
    wav_audio = AudioSegment.from_file(audio_path, format="wav")
    wav_audio = wav_audio.set_channels(1)           # set a single channel
    sample_rate = wav_audio.frame_rate              # extract the fs
    samples_audio = np.array(wav_audio.get_array_of_samples(), dtype=np.float64)  # transform the audio in a np.array
    return samples_audio, sample_rate


def read_label(audio_name, one_hot= True):
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

def extract_features_1v(audio_array, sample_rate):
    f1 = librosa.feature.zero_crossing_rate(audio_array, frame_length=132299)

    return [f1]

def extract_features(audio_array, sample_rate):
    f1 = librosa.feature.spectral_centroid(audio_array, sample_rate)[0]
    f2 = librosa.feature.spectral_bandwidth(audio_array, sample_rate)[0]
    f3 = librosa.feature.spectral_rolloff(audio_array, sample_rate)[0]
    f4 = librosa.feature.zero_crossing_rate(audio_array, sample_rate)[0]
    #f5 = librosa.feature.rmse(audio_array, sample_rate)[0]
    f6 = librosa.feature.mfcc(audio_array, sample_rate, n_mfcc=20)

    return [f1, f2, f3, f4, f6]


def dataset_preprocessor(input_path, normalize, output_path = 'C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\music_informatics\\data'):
    '''

    :param input_path: path of the audio files
    :param normalize: choose to normalize the audio segment
    :return:
    '''

    # count the total number of file we have. Useful to pre-allocate
    # the dataframe
    num_files = sum([len(files) for r, d, files in os.walk(input_path)])

    instr_folder = os.listdir(input_path)
    # number of features: 25
    # number of sample/each feature: 259
    data = np.empty((num_files, 25, 130), dtype=np.float32)
    data_labels = np.empty((num_files, 11), dtype= bool)        # num. classes = 11

    index = 0
    for instr in instr_folder:
        instr_path = os.path.join(input_path, instr)

        if instr == 'cel':
            for audio_name in os.listdir(instr_path):
                audio_path = os.path.join(instr_path, audio_name)   # retrieve the audio file path
                samples_audio, sample_rate = read_audio(audio_path)

                if normalize:
                    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
                    samples_audio = min_max_scaler.fit_transform(samples_audio.reshape(-1, 1)).reshape(-1, )
                else:
                    pass
                print('extracting features for audio: '+audio_name)
                #audio_features = extract_features(samples_audio, sample_rate)
                data[index, 0] = librosa.feature.spectral_centroid(samples_audio, sample_rate, hop_length= 5012//2)[0]
                data[index, 1] = librosa.feature.spectral_bandwidth(samples_audio, sample_rate, hop_length= 5012//2)[0]
                data[index, 2] = librosa.feature.spectral_rolloff(samples_audio, sample_rate, hop_length= 5012//2)[0]
                data[index, 3] = librosa.feature.zero_crossing_rate(samples_audio, sample_rate, hop_length= 5012//2)[0]
                # data[index, 0] = librosa.feature.rmse(audio_array, sample_rate)[0]
                data[index, 4:24] = librosa.feature.mfcc(samples_audio, sample_rate,hop_length= 5012//2, n_mfcc=20)

                data_labels[index] = read_label(audio_name)         # save the label of the sample
                index +=1

    print('saving to .h5t file...')
        # we use .h5 file since they can store 3-dim arrays (x,y,z)
    out_file = h5py.File(os.path.join(output_path,'out_file.h5'), 'w')
    out_file.create_dataset('trial_dataset', data= data)
    out_file.close()
    print('Done.')


    print('saving to .npy file...')
    np.save(os.path.join(output_path,'out_file_np.npy'), data)

    print('Done.')




    print('Done.')
        #df = pd.DataFrame(data)

    return data










#df = dataset_preprocessor('C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\IRMAS-TrainingData', False)