import argparse
import os

import numpy as np
import mne

parser = argparse.ArgumentParser(description="Template")

parser.add_argument('-ed', '--eeg-dataset', default=r".\datasets\tuab", help="EEG dataset path")
parser.add_argument('-tc', '--top-channels', default=r".\datasets\tuab\top_chans.txt", help="EEG top channels file")
parser.add_argument('-sr', '--sampling-rate', default=500, type=int, help="Sampling rate")
parser.add_argument('-d', '--duration', default=0.5, type=float, help="Duration")

opt, unknown = parser.parse_known_args()


with open(opt.top_channels, 'r') as f:
    top_chans = [line[:-1] for line in f.readlines()]

subsets = ['train', 'eval']
classes = ['normal', 'abnormal']

time = opt.duration
sr = opt.sampling_rate

for subset_i in subsets:
    for class_i in classes:
        path = opt.eeg_dataset + rf'\tuh_eeg_abnormal\v3.0.1\edf\{subset_i}\{class_i}\01_tcp_ar'
        path_resampled = path.replace('tuab', 'tuab_resampled')

        for file_name in os.listdir(path):
            if file_name.split('.')[-1] != 'edf':
                continue

            file_path = path + '\\' + file_name
            file_path_resampled = path_resampled + '\\' + file_name

            data = mne.io.read_raw_edf(file_path, verbose=False)
            ch_names = data.info.ch_names
            sfreq = data.info["sfreq"]
            step = int(sfreq * time)

            # Находим каналы
            indices = [index for index, element in enumerate(ch_names) if element in top_chans]
            raw = data.get_data()[indices, :]

            # Заполняем недостающие каналы
            indices = [index for index, element in enumerate(top_chans) if element not in ch_names]
            for index in indices:
                raw = np.insert(raw, index, np.zeros(raw.shape[1]), axis=0)

            # Семплирование и разбиение
            for i in range(int(raw.shape[1] / step)):
                raw_i = raw[:, i*step:(i + 1)*step]

                if raw_i.shape[1] != step:
                    continue

                if sfreq < sr:
                    raw_i = mne.filter.resample(raw_i, up=sr/sfreq, down=1)
                else:
                    raw_i = mne.filter.resample(raw_i, up=1, down=sfreq/sr)

                # Сохранение в файл
                n_channels = len(top_chans)  # Количество каналов
                n_times = raw_i.shape[1]  # Количество временных точек
                ch_names = top_chans
                ch_types = ['eeg'] * n_channels   # Типы каналов

                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                raw_to_save = mne.io.RawArray(raw_i, info)

                os.makedirs(path_resampled, exist_ok=True)    
                mne.export.export_raw('.'.join(file_path_resampled.split('.')[:-1]) + f'_{i + 1}' + '.edf', raw_to_save, fmt='edf', overwrite=True)