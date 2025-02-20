import argparse
import os

from npy_append_array import NpyAppendArray
import numpy as np
import mne

parser = argparse.ArgumentParser(description="Template")

parser.add_argument('-ed', '--eeg-dataset', default=r"./datasets/tuab", help="EEG dataset path")
parser.add_argument('-tc', '--top-channels', default=r"./datasets/tuab/top_chans.txt", help="EEG top channels file")
parser.add_argument('-sr', '--sampling-rate', default=500, type=int, help="Sampling rate")
parser.add_argument('-d', '--duration', default=0.5, type=float, help="Duration")

opt, unknown = parser.parse_known_args()


# with open(opt.top_channels, 'r') as f:
#     top_chans = [line[:-1] for line in f.readlines()]
top_chans = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
             'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
             'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 
             'EEG T1-REF', 'EEG T2-REF']

subsets = ['train', 'eval']
classes = ['normal', 'abnormal']

low_freq, high_freq = 0.1, 75  # Границы полосового фильтра
notch_freq = 50                # Частота сетевого фильтра
time = opt.duration            # Время сигнала
sr = opt.sampling_rate         # Целевая частота дискретизации сигнала
 
 # Границы обрезки сигнала
start_index = 10000
end_index = -1000 

for subset_i in subsets:
    for class_i in classes:
        path = opt.eeg_dataset + rf'/tuh_eeg_abnormal/v3.0.1/edf/{subset_i}/{class_i}/01_tcp_ar'
        path_resampled = path.replace('tuab', 'tuab_resampled')

        for file_name in os.listdir(path):
            if file_name.split('.')[-1] != 'edf':
                continue

            file_path = path + '/' + file_name
            file_path_resampled = path_resampled + '/' + file_name

            raw = mne.io.read_raw_edf(file_path, verbose=False, preload=True)
            raw.filter(l_freq=low_freq, h_freq=high_freq, fir_design='firwin')
            raw.notch_filter(freqs=notch_freq, fir_design='firwin')
            
            ch_names = raw.info.ch_names
            sfreq = raw.info["sfreq"]
            step = int(sfreq * time)

            # Находим каналы
            indices = [index for index, element in enumerate(ch_names) if element in top_chans]
            data = raw.get_data()[indices, start_index:end_index]

            # Заполняем недостающие каналы
            indices = [index for index, element in enumerate(top_chans) if element not in ch_names]
            for index in indices:
                data = np.insert(data, index, np.zeros(data.shape[1]), axis=0)

            # Семплирование и разбиение
            os.makedirs(path_resampled, exist_ok=True)    
            with NpyAppendArray(file_path_resampled.replace('.edf', '.npy'), delete_if_exists=True) as npaa:
                  for i in range(int(data.shape[1] / step)):
                    data_i = data[:, i*step:(i + 1)*step]

                    if data_i.shape[1] != step:
                        continue

                    if sfreq < sr:
                        data_i = mne.filter.resample(data_i, up=sr/sfreq, down=1)
                    else:
                        data_i = mne.filter.resample(data_i, up=1, down=sfreq/sr)

                    # Сохранение в файл
                    # n_channels = len(top_chans)  # Количество каналов
                    # n_times = data_i.shape[1]  # Количество временных точек
                    # ch_names = top_chans
                    # ch_types = ['eeg'] * n_channels   # Типы каналов

                    # info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                    # raw_to_save = mne.io.RawArray(data_i, info)

                    npaa.append(np.expand_dims(data_i, axis=0))
                    # mne.export.export_raw(file_path_resampled.replace('.edf', f'_{i + 1}.edf'),
                    #                       raw_to_save, 
                    #                       fmt='edf', 
                    #                       overwrite=True)