import numpy as np
import pandas as pd
import os

from mne.channels import read_montage
from mne import create_info, find_events, Epochs
from mne.io import RawArray, read_raw_fif
# from mne.epochs import concatenate_epochs
from mne.decoding import CSP
from mne.viz.topomap import _prepare_topo_plot, plot_topomap
from mne.preprocessing import ICA

from scipy.signal import welch

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

# from braindecode.visualization import plot
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
    global timestamps

    """ Create a mne raw instance from csv file """
    data_in = pd.read_csv(fname)
    # data_in = np.genfromtxt(fname)
    timestamps = data_in.values[:,0]
    data = np.transpose(np.asarray(data_in.values[:, 1:]))

    montage = read_montage('standard_1020', ch_names)
    ch_type = ['eeg']*len(ch_names)
    info = create_info(ch_names, sfreq=250.0, ch_types=ch_type, montage=montage)

    # print(data.shape)
    # print(len(info))

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
                ts = (timestamps[start] + timestamps[end])/2
                # ts = timestamps[end]
            else:
                try:
                    unique, counts = np.unique(label[start:end], return_counts=True)
                    labels = np.append(labels, unique[np.argmax(counts)])
                    segments = np.vstack([segments, x1_F])
                    ts_i = (timestamps[start] + timestamps[end])/2
                    # ts_i = timestamps[end]
                    ts = np.append(ts, ts_i)
                except ValueError:
                    continue

    segments = segments.reshape((-1, window_size, 32))
    # print(segments.shape)
    # print(labels.shape)

    return segments, labels, ts


def windows(data, size, overlap=1):
    start = 0
    while (start + size) < data.shape[0]:
        yield int(start), int(start + size)
        start += (size / overlap)


def run_csp(X_i_tot, y_i_tot_lat, y_i_tot_tp):
    csp = CSP()

    X = np.transpose(X_i_tot, axes=[0, 2, 1])
    X = X[1:]
    y_fin_lat = np.array(y_i_tot_lat)[1:]
    y_fin_tp = np.array(y_i_tot_tp)[1:]

    y = y_fin_tp

    ret = csp.fit_transform(X, y)

    plot_topomap(csp.patterns_[0], positions)
    plot_topomap(csp.patterns_[1], positions)
    plot_topomap(csp.patterns_[2], positions)
    plot_topomap(csp.patterns_[3], positions)

    plot_topomap(csp.filters_[0], positions)
    plot_topomap(csp.filters_[1], positions)
    plot_topomap(csp.filters_[2], positions)
    plot_topomap(csp.filters_[3], positions)

    return ret, y


def plot_power(data_in_th, data_in_al, data_in_beta, ts):

    seg_power_th = np.zeros((data_in_th.shape[0], 32))
    segments_th = np.transpose(data_in_th, axes=[0, 2, 1])
    seg_power_al = np.zeros((data_in_al.shape[0], 32))
    segments_al = np.transpose(data_in_al, axes=[0, 2, 1])
    seg_power_beta = np.zeros((data_in_beta.shape[0], 32))
    segments_beta = np.transpose(data_in_beta, axes=[0, 2, 1])
    # segments = data_in

    for k in range(len(segments)):
        seg_power_th[k] = np.sqrt(np.mean(np.square(segments_th[k]), axis=1))
        seg_power_al[k] = np.sqrt(np.mean(np.square(segments_al[k]), axis=1))
        seg_power_beta[k] = np.sqrt(np.mean(np.square(segments_beta[k]), axis=1))
        # seg_power_th[k] = np.mean(segments_th[k], axis=1)
        # seg_power_al[k] = np.mean(segments_al[k], axis=1)
        # seg_power_beta[k] = np.mean(segments_beta[k], axis=1)

    frontal_lobe_power_ar_th = seg_power_th[:, :10]
    frontal_lobe_power_th = np.mean(frontal_lobe_power_ar_th, axis=1)
    frontal_lobe_power_ar_al = seg_power_al[:, :10]
    frontal_lobe_power_al = np.mean(frontal_lobe_power_ar_al, axis=1)
    frontal_lobe_power_ar_beta = seg_power_beta[:, :10]
    frontal_lobe_power_beta = np.mean(frontal_lobe_power_ar_beta, axis=1)

    # trial_channel_pow_th = np.mean(seg_power_th, axis=0)
    # print(trial_channel_pow_th)
    # plot_topomap(trial_channel_pow_th, positions)
    # trial_channel_pow_al = np.mean(seg_power_al, axis=0)
    # print(trial_channel_pow_al)
    # plot_topomap(trial_channel_pow_al, positions, vmax=10)
    # trial_channel_pow_beta = np.mean(seg_power_beta, axis=0)
    # print(trial_channel_pow_beta)
    # plot_topomap(trial_channel_pow_beta, positions, vmax=10)

    flpt_smoothed = smooth_plot(frontal_lobe_power_th)
    flpa_smoothed = smooth_plot(frontal_lobe_power_al)
    flpb_smoothed = smooth_plot(frontal_lobe_power_beta)

    ts = ts[3:]

    plt.figure()
    axes = plt.gca()
    axes.set_ylim([0, 20])
    # axes.set_xlim([ts[0], markers_ar[-1, -1]])
    plt.plot(ts, flpt_smoothed)
    plt.plot(ts, flpa_smoothed)
    plt.plot(ts, flpb_smoothed)
    for mrkr, event in markers_ar:
        if mrkr > 0:
            if int(event) == -51 or int(event) == -52 or int(event) == -53 or int(event) == -55:
                plt.axvline(x=mrkr, color='red')
            else:
                plt.axvline(x=mrkr, color='black')
    plt.show()


def smooth_plot(data_in, sliding_win=6):

    data_out = np.zeros(data_in.shape[0])

    for i in range(sliding_win, data_in.shape[0]):
        num_mean = np.mean(data_in[i-sliding_win:i])
        # data_out[i-sliding_win] = num_mean
        data_out[i-3] = num_mean

    data_out = data_out[3:]

    return data_out


def z_normalize_data(data_in):
    mean = np.mean(data_in, axis=0)
    stddev = np.std(data_in, axis=0)

    data_out = (data_in - mean)/stddev

    return data_out


def clean_EEG_ICA(raw):
    print('apply fastica ICA')
    method = 'fastica'
    n_components = 25
    decim = 3
    random_state = 23
    reject = dict(eeg=120, grad=4000e-13)

    raw.filter(1.0, 40.0, method='iir', verbose=False)
    ica = ICA(n_components=n_components, method=method, random_state=random_state)
    # ica.fit(raw, picks='eeg', decim=decim, reject=reject)
    ica.fit(raw)

    ica.detect_artifacts(raw, start_find=None, stop_find=None, ecg_ch=None, ecg_score_func='pearsonr',
                         ecg_criterion=0.1, eog_ch='FP1', eog_score_func='pearsonr', eog_criterion=0.09,
                         skew_criterion=-1, kurt_criterion=-1, var_criterion=0, add_nodes=None)
    ica.apply(raw)

    # raw.filter(12.5, 30, method='iir')

    return (raw)


def plot_combined_topomaps(data_folder):
    print('plotting combined topomaps')

    X_final = np.zeros((1, 32))
    y_final = np.zeros(1)

    norm_by_seg = False
    norm_by_sub = True
    norm_by_seg_2 = False
    show_sub_topo = False

    for s in ["A/", "B/", "C/", "D/"]:
    # for s in ["A/", "C/", "D/"]:
        print(s)
        folder_name = data_folder + s

        X_fin_sub = np.zeros((1, 32))
        X_sub_rms_i = np.zeros((1, 32))

        X_i_tot = np.zeros((1, win_size, 32))
        y_i_tot = np.zeros(1)

        for filename in os.listdir(folder_name):
            if filename[-8] == 'a':

                try:
                    raw = read_raw_fif('/data2/SpaeTrial092019/clean_EEG_Data/' + s + filename[:-4] + '_raw.fif')
                except FileNotFoundError:
                    raw = create_mne_raw_object(folder_name + filename)

                    # raw.filter(7, 30, method='iir', verbose=False)
                    raw = clean_EEG_ICA(raw)

                    raw.save('/data2/SpaceTrial092019/clean_EEG_Data/' + s + filename[:-4] + '_raw.fif', overwrite=True)

                y = [filename[-6:-4]]
                print(y)
                y_i = y * (raw.get_data().shape[1])

                segments, labels, ts = segment_signal_without_transition(raw.get_data(), np.asarray(y_i), win_size)
                X_i_tot = np.append(X_i_tot, segments, axis=0)
                y_i_tot = np.append(y_i_tot, labels, axis=0)

                X_tr_rms = np.sqrt(np.mean(np.square(segments), axis=1))
                X_sub_rms_i = np.append(X_sub_rms_i, X_tr_rms, axis=0)

                if (norm_by_seg):

                    X_tr_mean = np.mean(X_tr_rms, axis=0)
                    X_tr_sub = np.std(X_tr_rms, axis=0)

                    X_tr_norm = (X_tr_rms - X_tr_mean)/X_tr_sub

                    X_fin_sub = np.append(X_fin_sub, X_tr_norm, axis=0)
                    y_i_tot = np.append(y_i_tot, labels, axis=0)

                    X_final = np.append(X_final, X_tr_norm, axis=0)
                    y_final = np.append(y_final, labels, axis=0)

        # # If normalizing per subject
        if (norm_by_sub):

            X_sub = X_i_tot[1:]
            y_sub = y_i_tot[1:]
            if not norm_by_seg_2:
                X_sub_rms = np.sqrt(np.mean(np.square(X_sub), axis=1))
            else:
                X_sub_rms = X_sub_rms_i[1:]
            X_sub_mean = np.mean(X_sub_rms, axis=0)
            X_sub_stddev = np.std(X_sub_rms, axis=0)

            X_sub_norm = (X_sub_rms - X_sub_mean)/X_sub_stddev

            X_final = np.append(X_final, X_sub_norm, axis=0)
            y_final = np.append(y_final, y_sub, axis=0)

        # # If you want to show topographic data for each subject
        if (show_sub_topo):
            if (norm_by_seg):
                X_sub_norm = X_fin_sub[1:]

            for clss in ['00', '10', '01', '02', '03', '11', '12', '13']:
                min0 = np.min(X_sub_norm[y_sub=='00'])
                max0 = np.max(X_sub_norm[y_sub=='13'])

                X_sub_cl = X_sub_norm[y_sub == clss]
                ch_f_cl = np.mean(X_sub_cl, axis=0)
                print(ch_f_cl)
                plot_topomap(ch_f_cl, positions, vmin=min0, vmax=max0)


    X_final = X_final[1:]
    y_final = y_final[1:]
    print(X_final.shape)
    print(y_final.shape)

    for clss in ['00', '10', '01', '02', '03', '11', '12', '13']:
        print(clss)

        # min0 = np.min(np.mean(X_final[y_final=='00'], axis=0))
        # max0 = np.max(np.mean(X_final[y_final=='00'], axis=0))

        # min0 = np.min(X_final[y_final == '13'])
        # max0 = np.max(X_final[y_final == '13'])

        # print(min0, max0)

        X_f_cl = X_final[y_final==clss]
        y_f_cl = y_final[y_final==clss]

        ch_f_cl = np.median(X_f_cl, axis=0)
        print(np.min(ch_f_cl), np.max(ch_f_cl))
        plot_topomap(ch_f_cl, positions, vmin=-1, vmax=0.5)




if __name__ == "__main__":

    global markers_ar

    win_size = 500
    i = 0
    y_label_A_tp = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
    y_label_A_lat = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 3, 2, 1, 3, 2]
    y_label_B_tp = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1]
    y_label_B_lat = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 2, 3, 1, 2, 3, 1]

    X_i_tot = np.zeros((1, win_size, 32))
    y_i_tot_lat = np.zeros(1)
    y_i_tot_tp = np.zeros(1)

    # plot_combined_topomaps(EEG_data_folder)

    band = 'theta'
    # band = 'alpha'

    # folder_name = EEG_data_folder + "All_Data/"
    folder_name = EEG_data_folder + "B/"

    for filename in os.listdir(folder_name):
        if filename[-8] == 'a':
            markers_ar = [[0, 0]]

            print(filename)

            raw_theta = create_mne_raw_object(folder_name + filename)
            raw_alpha = create_mne_raw_object(folder_name + filename)
            raw_beta = create_mne_raw_object(folder_name + filename)

            basename = filename[:-15]
            marker_filename = basename + "markers.csv"
            try:
                # markers_in = np.genfromtxt(folder_name + marker_filename)
                markers_in = pd.read_csv(folder_name + marker_filename, header=None)
                markers_ar = np.asarray(markers_in)
            except:
                markers_ar = [[0, 0]]

            # raw_theta.plot_psd(area_mode='range')
            raw_theta.filter(4, 7, method='iir', verbose=False)
            # plt.figure()
            # plt.plot(raw_theta.get_data()[7, :])
            # plt.show()
            # raw_theta.plot_psd(area_mode='range')
            raw_alpha.filter(8, 15, method='iir', verbose=False)
            # raw_alpha.plot_psd(area_mode='range')
            raw_beta.filter(16, 30, method='iir', verbose=False)

            y_lat = [int(filename[-5])]
            y_tp = [int(filename[-6])]
            y_i_lat = y_lat * (raw_theta.get_data().shape[1])
            y_i_tp = y_tp * (raw_alpha.get_data().shape[1])

            segments_theta, labels_lat, ts = segment_signal_without_transition(raw_theta.get_data(), np.asarray(y_i_lat), win_size)
            segments_alpha, labels_tp, ts = segment_signal_without_transition(raw_alpha.get_data(), np.asarray(y_i_tp), win_size)
            segments_beta, labels_tp, ts = segment_signal_without_transition(raw_beta.get_data(), np.asarray(y_i_tp), win_size)

            segments = segments_theta

            X_i_tot = np.append(X_i_tot, segments, axis=0)
            y_i_tot_lat = np.append(y_i_tot_lat, labels_lat, axis=0)
            y_i_tot_tp = np.append(y_i_tot_tp, labels_tp, axis=0)

            i += 1

            plot_power(segments_theta, segments_alpha, segments_beta, ts)


    ret, y = run_csp()

    svc = SVC()

    svc.fit(ret, y)
    print("SVC score: ")
    print(svc.score(ret, y))

    # plot.ax_scalp(csp.filters_[0], ch_names)




