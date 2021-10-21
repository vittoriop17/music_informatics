import os
#import librosa
#import pydub
import numpy as np
import string, re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from pydub import AudioSegment



def dataset_preprocessor(input_path, normalize):
    '''

    :param input_path: path of the audio files
    :param normalize: choose to normalize the audio segment
    :return:
    '''

    # count the total number of file we have. Useful to pre-allocate
    # the dataframe
    num_files = sum([len(files) for r, d, files in os.walk(input_path)])

    instr_folder = os.listdir(input_path)
    data= np.empty((num_files, 6), dtype=np.float32)

    for instr in instr_folder:
        instr_path = os.path.join(input_path, instr)
        for audio_name in os.listdir(instr_path):
            label = read_label(audio_name)                      # read label from the name
            audio_path = os.path.join(instr_path, audio_name)   # retrieve the audio file path
            samples_audio, sample_rate = read_audio(audio_path)

            if normalize:
                min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
                samples_audio = min_max_scaler.fit_transform(samples_audio.reshape(-1, 1)).reshape(-1, )
            else:
                pass

            audio_features = extract_features(samples_audio, sample_rate)
            data.append(audio_features)

    df = pd.DataFrame(data)

    return df





def read_label(audio_name):
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
    return label_dict[label]

def read_audio(audio_path):
    wav_audio = AudioSegment.from_file(audio_path, format="wav")
    wav_audio = wav_audio.set_channels(1)           # set a single channel
    sample_rate = wav_audio.frame_rate              # extract the fs
    samples_audio = np.array(wav_audio.get_array_of_samples(), dtype=np.float64)  # transform the audio in a np.array
    return samples_audio, sample_rate

def extract_features(audio_array, sample_rate):

    f1 = librosa.feature.spectral_centroid(audio_array, sample_rate)[0]
    f2 = librosa.feature.spectral_bandwidth(audio_array, sample_rate)[0]
    f3 = librosa.feature.spectral_rolloff(audio_array, sample_rate)[0]
    f4 = librosa.feature.zero_crossing_rate(audio_array, sample_rate)[0]
    f5 = librosa.feature.rmse(audio_array, sample_rate)[0]
    f6 = librosa.feature.mfcc(audio_array, sample_rate, n_mfcc=20)

    return [f1, f2, f3, f4, f5, f6]



res=read_label('[sax][pop_roc]0014__1.wav')
#dataset_preprocessor('C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\IRMAS-TrainingData')