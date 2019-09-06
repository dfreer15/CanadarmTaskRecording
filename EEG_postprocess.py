import numpy as np
import pandas as pd
import os

from mne.channels import read_montage
from mne import create_info, find_events, Epochs
from mne.io import RawArray
from mne.epochs import concatenate_epochs
from mne.decoding import CSP
from mne.viz.topomap import _prepare_topo_plot, plot_topomap

from scipy.signal import welch

from braindecode.visualization import plot
from braindecode.datasets.sensor_positions import get_channelpos, CHANNEL_10_20_APPROX

from sklearn.svm import SVC


EEG_data_folder = "/data2/SpaceTrial092019/EEG_Data/"
ch_names = list(['FP1', 'FP2', 'AF3', 'AF4', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5',
                    'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'CZ', 'T8', 'CP5',
                    'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO7', 'PO3',
                    'PO4', 'PO8', 'OZ'])
positions = [get_channelpos(name, CHANNEL_10_20_APPROX) for name in ch_names]
positions = np.array(positions)

def create_mne_raw_object(fname):
    """ Create a mne raw instance from csv file """
    data_in = pd.read_csv(fname)
    timestamps = data_in.values[:,0]
    data = np.transpose(np.asarray(data_in.values[:, 1:]))

    montage = read_montage('standard_1020', ch_names)
    ch_type = ['eeg']*len(ch_names)
    info = create_info(ch_names, sfreq=250.0, ch_types=ch_type, montage=montage)

    print(data.shape)
    print(len(info))

    raw = RawArray(data, info, verbose=False)

    return raw


def segment_signal_without_transition(data_in, label, window_size, overlap=1):

    data = np.transpose(data_in)
    # print(data.shape)

    for (start, end) in windows(data, window_size, overlap=overlap):
        if len(data[start:end]) == window_size:
            x1_F = data[start:end]
            if start == 0:
                unique, counts = np.unique(label[start:end], return_counts=True)
                labels = unique[np.argmax(counts)]
                segments = x1_F
            else:
                try:
                    unique, counts = np.unique(label[start:end], return_counts=True)
                    labels = np.append(labels, unique[np.argmax(counts)])
                    segments = np.vstack([segments, x1_F])
                except ValueError:
                    continue

    segments = segments.reshape((-1, window_size, 32))
    # print(segments.shape)
    # print(labels.shape)

    return segments, labels


def windows(data, size, overlap=1):
    start = 0
    while (start + size) < data.shape[0]:
        yield int(start), int(start + size)
        start += (size / overlap)


if __name__ == "__main__":


    win_size = 2000
    i = 0
    y_label_A_tp = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
    y_label_A_lat = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 3, 2, 1, 3, 2]
    y_label_B_tp = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1]
    y_label_B_lat = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 2, 3, 1, 2, 3, 1]

    X_i_tot = np.zeros((1, win_size, 32))
    y_i_tot_lat = np.zeros(1)
    y_i_tot_tp = np.zeros(1)

    folder_name = EEG_data_folder + "C/"

    for filename in os.listdir(folder_name):
        if filename[-8] == 'a':
            print(filename)
            raw = create_mne_raw_object(folder_name + filename)

            raw.filter(0.5, 60, method='iir', verbose=False)

            y_lat = [int(filename[-5])]
            y_tp = [int(filename[-6])]
            y_i_lat = y_lat * (raw.get_data().shape[1])
            y_i_tp = y_tp * (raw.get_data().shape[1])

            segments, labels_lat = segment_signal_without_transition(raw.get_data(), np.asarray(y_i_lat), win_size)
            segments, labels_tp = segment_signal_without_transition(raw.get_data(), np.asarray(y_i_tp), win_size)

            X_i_tot = np.append(X_i_tot, segments, axis=0)
            y_i_tot_lat = np.append(y_i_tot_lat, labels_lat, axis=0)
            y_i_tot_tp = np.append(y_i_tot_tp, labels_tp, axis=0)

            i += 1

    csp = CSP()
    svc = SVC()

    X = np.transpose(X_i_tot, axes=[0, 2, 1])
    X = X[1:]
    y_fin_lat = np.array(y_i_tot_lat)[1:]
    y_fin_tp = np.array(y_i_tot_tp)[1:]

    y = y_fin_tp

    ret = csp.fit_transform(X, y)
    svc.fit(ret, y)
    print("SVC score: ")
    print(svc.score(ret, y))

    print(ret.shape)
    print(csp.patterns_.shape)

    # plot.ax_scalp(csp.filters_[0], ch_names)

    plot_topomap(csp.patterns_[0], positions)
    plot_topomap(csp.patterns_[1], positions)
    plot_topomap(csp.patterns_[2], positions)
    plot_topomap(csp.patterns_[3], positions)

    plot_topomap(csp.filters_[0], positions)
    plot_topomap(csp.filters_[1], positions)
    plot_topomap(csp.filters_[2], positions)
    plot_topomap(csp.filters_[3], positions)


