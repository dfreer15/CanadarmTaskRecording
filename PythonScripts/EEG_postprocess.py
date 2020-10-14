import numpy as np
import pandas as pd
import os

from mne.channels import read_montage
from mne import create_info, find_events, Epochs
from mne.io import RawArray, read_raw_fif
# from mne.epochs import concatenate_epochs
from mne.decoding import CSP
from mne.viz.topomap import _prepare_topo_plot, plot_topomap
from mne.filter import notch_filter
from mne.preprocessing import ICA
from mne.time_frequency.tfr import cwt, morlet      # also tfr_morlet
import pywt
import copy

# from scipy.signal import welch

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import pyriemann
from pyriemann.utils.distance import distance
from pyriemann.utils.mean import mean_covariance

from nearest_correlation import nearcorr
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# from braindecode.visualization import plot
from braindecode.datasets.sensor_positions import get_channelpos, CHANNEL_10_20_APPROX

from sklearn.svm import SVC


np.set_printoptions(suppress=True)

# EEG_data_folder = "/data2/SpaceTrial092019/EEG_Data/"
EEG_data_folder = "D:/PhD_Data/CanadarmTask/"
ch_names = list(['FP1', 'FP2', 'AF3', 'AF4', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5',
                    'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'CZ', 'T8', 'CP5',
                    'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO7', 'PO3',
                    'PO4', 'PO8', 'OZ'])
positions = [get_channelpos(name, CHANNEL_10_20_APPROX) for name in ch_names]
positions = np.array(positions)

pp_data_folder = "D:/PhD_Data/CanadarmTask/Processed_Data/"


def read_data_from_file(filename, s, num_classes=5):
    data_folder = EEG_data_folder
    folder_name = data_folder + s + "EEGData/"
    try:
        raw = read_raw_fif('D:/PhD_Data/CanadarmTask/' + s + 'clean_EEG_Data/' + filename[:-4] + '_rawasldfkj.fif')
        raw_np = raw.get_data()
    except FileNotFoundError:
        print("File not found!")
        raw, ts = create_mne_raw_object(folder_name + filename)
        # raw = clean_EEG_ICA(raw)
        raw_np = clean_EEG(raw)
        # # If you want to save prefiltered data, uncomment the following line. Can only save this way if it is an MNE raw object
        # raw.save('D:/PhD_Data/CanadarmTask/' + s + 'clean_EEG_Data/' + filename[:-4] + '_raw.fif', overwrite=True)
        # np.save(raw_np)    # This is probably not correct - fix later if you want to save

    # Get labels for time pressure and latency
    y_s = filename[-6:-4]
    consider_action = False

    y = relabel_data(y_s, num_classes=num_classes)
    y_i = y * np.ones(raw_np.shape[1])
    if consider_action:
        relabel_data_ca(y_i, ts, filename, s, num_classes=num_classes)

    return raw_np, y_i


def relabel_data(y_s, num_classes=5):
    # paradigm = 'time_pressure'
    # paradigm = 'latency'
    paradigm = None

    print('# # # # # # # # # # # y_s is: {}'.format(y_s))
    if paradigm == 'time_pressure':
        y = int(y_s[0])
    elif paradigm == 'latency':
        if int(y_s[1]) < 2:
            y = int(y_s[1])
        else:
            y = 1
    elif num_classes == 5:
        if y_s == '00':
            y = 0
        elif y_s == '01':
            y = 1
        elif y_s == '10':
            y = 2
        elif y_s == '11':
            y = 3
        else:
            y = 4
    elif num_classes == 6:
        if y_s == '00':
            y = 0
        elif y_s == '01':
            y = 1
        elif y_s == '10':
            y = 2
        elif y_s == '11':
            y = 3
        elif y_s[0] == '0':
            y = 4
        else:
            y = 5
    elif num_classes == 2:
        if y_s == '00':
            y = 0
        else:
            y = 1

    return y


def relabel_data_ca(y_i, ts, eeg_filename, s, num_classes=5):

    data_in_ct, data_in_EE_m, data_in_EE_o = get_matching_Unity_data(eeg_filename, s)

    rel_ts_EEG = ts - ts[0]
    ts_Unity = data_in_ct[:, 0]
    i_EEG = 0
    i_Unity = 0

    data_ct = data_in_ct[:, 1:]
    data_EE_m = data_in_EE_m[:, 1:]
    data_EE_o = data_in_EE_o[:, 1:]

    print(rel_ts_EEG.shape)
    print(y_i.shape)
    print(ts_Unity.shape)

    unique, counts = np.unique(y_i, return_counts=True)
    print("Starting Labels: ", unique, counts)

    for i in range(len(y_i)):
        try:
            while ts_Unity[i_Unity] < rel_ts_EEG[i]:
                # print(ts_Unity[i_Unity], rel_ts_EEG[i])
                i_Unity += 1
            if np.sum(np.abs(data_ct[i_Unity])) > 0.95:
                y_i[i] += num_classes
            if np.sum(np.abs(data_EE_m[i_Unity])) > 0.95:
                y_i[i] += num_classes
            if np.sum(np.abs(data_EE_o[i_Unity])) > 0.95:
                y_i[i] += num_classes
        except IndexError:
            unique, counts = np.unique(y_i, return_counts=True)
            print("Ending Labels: ", unique, counts)
            return
        except TypeError:
            continue

    unique, counts = np.unique(y_i, return_counts=True)
    print("Ending Labels: ", unique, counts)

    return


def is_obstacles(eeg_fname, s):
    act_data_folder = 'D:/PhD_Data/CanadarmTask/' + s + 'UnityData/'
    act_files = get_matching_Unity_file(act_data_folder, eeg_fname)

    if act_files[0][20] == 'o':
        return True
    elif act_files[0][23] == 'o':
        return True

    print(act_files[0])

    return False


def get_matching_Unity_file(act_data_folder, eeg_filename):
    filenames = os.listdir(act_data_folder)
    act_files = []
    act_file_time = []
    for fname in filenames:
        # Find if data was collected in the same hour as EEG data
        if int(fname[11:13]) == int(eeg_filename[11:13]):
            # Find if data was collected between 10 and 1 minute before the EEG data
            if int(fname[14:16]) > int(eeg_filename[14:16]) - 15 and int(fname[14:16]) < int(eeg_filename[14:16]) - 1:
                act_files.append(fname)
                act_file_time.append(fname[11:16])
        # Find if data was collected the hour before the EEG data
        elif int(fname[11:13]) == int(eeg_filename[11:13]) - 1:
            # Find if data was collected near the end of the hour
            if int(fname[14:16]) > 50:
                act_files.append(fname)
                act_file_time.append(fname[11:16])

    while len(act_files) > 5:
        # Remove files that fit above conditions, but do not match EEG data
        # Ensure only closest files remain
        max_hour = 0
        max_min = 0
        for act_file in act_files:
            if int(act_file[11:13]) > max_hour:
                max_hour = int(act_file[11:13])

        for i in reversed(range(len(act_files))):
            act_file = act_files[i]
            if not int(act_file[11:13]) == max_hour:
                act_files.remove(act_file)

        for act_file in act_files:
            if int(act_file[14:16]) > max_min:
                max_min = int(act_file[14:16])

        for i in reversed(range(len(act_files))):
            act_file = act_files[i]
            if not int(act_file[14:16]) == max_min:
                act_files.remove(act_file)

    return act_files


def get_matching_Unity_data(eeg_filename, s):
    print("GETTING MATCHING UNITY DATA")
    # Get "action" data
    # Unity data time is earlier than the eeg data
    act_data_folder = 'D:/PhD_Data/CanadarmTask/' + s + 'UnityData/'

    act_files = get_matching_Unity_file(act_data_folder, eeg_filename)

    for act_file in act_files:
        if act_file[-12:-10] == 'CT':
            data_in_ct = pd.read_csv(act_data_folder + act_file)
        elif act_file[-17:-13] == 'EE_m':
            data_in_EE_m = pd.read_csv(act_data_folder + act_file)
        elif act_file[-19:-15] == 'EE_o':
            data_in_EE_o = pd.read_csv(act_data_folder + act_file)

    return np.asarray(data_in_ct), np.asarray(data_in_EE_m), np.asarray(data_in_EE_o)


def create_mne_raw_object(fname):
    global timestamps

    """ Create a mne raw instance from csv file """
    data_in = pd.read_csv(fname)
    timestamps = data_in.values[:, 0]
    data = np.transpose(np.asarray(data_in.values[:, 1:]))

    montage = read_montage('standard_1020', ch_names)
    ch_type = ['eeg']*len(ch_names)
    info = create_info(ch_names, sfreq=250.0, ch_types=ch_type, montage=montage)

    raw = RawArray(data, info, verbose=False)

    return raw, timestamps


def segment_signal_without_transition(data_in, label, window_size, overlap=1, ts_in=None):
    global timestamps

    if ts_in is not None:
        timestamps = ts_in

    data = np.transpose(data_in)
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
                    segments = np.vstack([segments, x1_F])
                    unique, counts = np.unique(label[start:end], return_counts=True)
                    labels = np.append(labels, unique[np.argmax(counts)])
                    ts_i = (timestamps[start] + timestamps[end])/2
                    # ts_i = timestamps[end]
                    ts = np.append(ts, ts_i)
                except ValueError:
                    continue

    segments = segments.reshape((-1, window_size, 32))

    return segments, labels, ts


def unsegment_signal(data_in, window_size, overlap=1):
    data_out = np.zeros((1, data_in.shape[2]))
    win_int = int(window_size/overlap)

    for data_t in data_in:
        data_out = np.append(data_out, data_t[:win_int], axis=0)

    data_out = np.transpose(data_out[1:])

    return data_out


def windows(data, size, overlap=1):
    start = 0
    while (start + size) < data.shape[0]:
        yield int(start), int(start + size)
        start += (size / overlap)


def select_channels(data, method='midline'):
    method = 'central_diamond'
    if method == 'parallel_bars':
        data_out = data[:, :, :2]                                   # FP1 and FP2
        data_out = np.append(data_out, data[:, :, 5:6], axis=2)     # F3
        data_out = np.append(data_out, data[:, :, 7:8], axis=2)     # F4
        data_out = np.append(data_out, data[:, :, 14:15], axis=2)   # C3
        data_out = np.append(data_out, data[:, :, 15:16], axis=2)   # C4
        data_out = np.append(data_out, data[:, :, 23:24], axis=2)   # P3
        data_out = np.append(data_out, data[:, :, 25:26], axis=2)   # P4
    elif method == 'frontal':
        data_out = data[:, :, :9]
    elif method == 'central_diamond':
        data_out = data[:, :, 6:8]
        data_out = np.append(data_out, data[:, :, 10:12], axis=2)
        data_out = np.append(data_out, data[:, :, 14:17], axis=2)
        data_out = np.append(data_out, data[:, :, 19:20], axis=2)
        data_out = np.append(data_out, data[:, :, 20:21], axis=2)
        data_out = np.append(data_out, data[:, :, 24:25], axis=2)
        data_out = np.delete(data_out, 1, axis=2)
    elif method == 'central_x':
        data_out = data[:, :, 5:8]
        data_out = np.append(data_out, data[:, :, 10:11], axis=2)
        data_out = np.append(data_out, data[:, :, 11:12], axis=2)
        data_out = np.append(data_out, data[:, :, 16:17], axis=2)
        data_out = np.append(data_out, data[:, :, 19:21], axis=2)
        data_out = np.append(data_out, data[:, :, 23:24], axis=2)
        data_out = np.append(data_out, data[:, :, 25:26], axis=2)
        data_out = np.delete(data_out, 1, axis=2)
    elif method == 'parietal':
        data_out = data[:, :, 18:27]
    elif method == 'parietal2':
        data_out = data[:, :, 22:31]
    elif method == 'rocket':
        data_out = data[:, :, 5:8]
        data_out = np.append(data_out, data[:, :, 10:12], axis=2)
        data_out = np.append(data_out, data[:, :, 16:17], axis=2)
        data_out = np.append(data_out, data[:, :, 19:20], axis=2)
        data_out = np.append(data_out, data[:, :, 20:21], axis=2)
        data_out = np.append(data_out, data[:, :, 24:25], axis=2)
    elif method == 'rocket2':
        data_out = data[:, :, 2:4]
        data_out = np.append(data_out, data[:, :, 6:7], axis=2)
        data_out = np.append(data_out, data[:, :, 10:12], axis=2)
        data_out = np.append(data_out, data[:, :, 16:17], axis=2)
        data_out = np.append(data_out, data[:, :, 19:20], axis=2)
        data_out = np.append(data_out, data[:, :, 20:21], axis=2)
        data_out = np.append(data_out, data[:, :, 24:25], axis=2)

    return data_out


def norm_dataset(dataset_1D):
    norm_dataset_1D = np.zeros(dataset_1D.shape)
    for i in range(dataset_1D.shape[0]):
        norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
    return norm_dataset_1D


def feature_normalize(data):
    # # # Z-normalises one segment of data (timepoints, channels) by its entire mean and
    # # # standard deviation
    # Returns: normalised data of the same shape as the input

    mean = data.mean()
    sigma = data.std()
    data_normalized = data
    data_normalized = (data_normalized - mean) / sigma
    data_normalized = (data_normalized - np.min(data_normalized)) / np.ptp(data_normalized)
    mean_dn = data_normalized.mean()
    sigma_dn = data_normalized.std()

    return data_normalized


def feature_normalize2(data):
    # # # Z-normalises data along the time axis. Each channel has a different mean and standard deviation
    # Returns: normalised data of the same shape as the input
    data_norm = np.zeros(data.shape)

    for i in range(data.shape[1]):
        data_norm[:, i] = feature_normalize(data[:, i])

    return data_norm


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


def clean_EEG(raw, use_ica=True, use_balert=False, use_bandpass=False, use_beta=True):
    # raw.plot_psd(area_mode='range')
    # Notch filter
    raw.notch_filter(50)  # The electrical peaks appear to be at 50 Hz and 100 Hz
    raw.notch_filter(100)
    # raw.plot_psd(area_mode='range')
    if use_bandpass:
        raw.filter(2.0, 64.0, method='iir', verbose=False)
        data = raw.get_data()
        # raw.plot_psd(area_mode='range')
    if use_ica:
        raw = clean_EEG_ICA(raw)
        # raw.plot_psd(area_mode='range')
        data = raw.get_data()
    if use_balert:
        data = wt_full_run(raw.get_data())
        # data = wavelet_transform(raw.get_data())
        # raw.plot_psd(area_mode='range')

    return data


def clean_EEG_ICA(raw):
    method = 'fastica'
    n_components = 25
    decim = 3
    random_state = 23
    reject = dict(eeg=120, grad=4000e-13)

    # raw.filter(1.0, 40.0, method='iir', verbose=False)  # Is this filter good/necessary? Maybe not...
    ica = ICA(n_components=n_components, method=method, random_state=random_state)
    # ica.fit(raw, picks='eeg', decim=decim, reject=reject)
    ica.fit(raw)

    ica.detect_artifacts(raw, start_find=None, stop_find=None, ecg_ch=None, ecg_score_func='pearsonr',
                         ecg_criterion=0.1, eog_ch='FP1', eog_score_func='pearsonr', eog_criterion=0.09,
                         skew_criterion=-1, kurt_criterion=-1, var_criterion=0, add_nodes=None)
    ica.apply(raw)

    # raw.filter(4, 7.5, method='iir')  # theta band
    # raw.filter(7.5, 12.5, method='iir')  # alpha band
    # raw.filter(12.5, 30, method='iir')  # beta band

    return raw


def remove_EMG():
    return


def preprocess_cleaned_eeg(segments, X_sub, rms=False, average=False, mav=False, power=False):

    if rms:
        X_tr = np.sqrt(np.mean(np.square(segments), axis=1))
        X_sub = np.append(X_sub, X_tr, axis=0)
    elif average:
        X_tr = np.mean(segments, axis=1)
        X_sub = np.append(X_sub, X_tr, axis=0)
    elif mav:
        X_tr = np.mean(np.abs(segments), axis=1)
        X_sub = np.append(X_sub, X_tr, axis=0)
    elif power:
        X_tr = np.mean(np.square(segments), axis=1)
        X_sub = np.append(X_sub, X_tr, axis=0)
    # Can also do WL, AUC, TSE (temporal spectral evolution) etc.
    # Could do wavelet or Fourier analysis

    return X_tr, X_sub


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

    for k in range(len(data_in_th)):
        seg_power_th[k] = np.sqrt(np.mean(np.square(segments_th[k]), axis=1))
        seg_power_al[k] = np.sqrt(np.mean(np.square(segments_al[k]), axis=1))
        seg_power_beta[k] = np.sqrt(np.mean(np.square(segments_beta[k]), axis=1))

    frontal_lobe_power_ar_th = seg_power_th[:, :10]
    frontal_lobe_power_th = np.mean(frontal_lobe_power_ar_th, axis=1)
    frontal_lobe_power_ar_al = seg_power_al[:, :10]
    frontal_lobe_power_al = np.mean(frontal_lobe_power_ar_al, axis=1)
    frontal_lobe_power_ar_beta = seg_power_beta[:, :10]
    frontal_lobe_power_beta = np.mean(frontal_lobe_power_ar_beta, axis=1)

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


def plot_combined_topomaps(data_folder):
    print('plotting combined topomaps')

    global folder_name, s

    X_final = np.zeros((1, 32))
    y_final = np.zeros(1)

    norm_by_sub = True
    norm_by_seg = False
    show_sub_topo = False

    for s in ["A/", "B/", "C/", "D/", "E/", "F/", "G/", "H/", "I/", "Z/"]:
    # for s in ["G/"]:
        print(s)
        folder_name = data_folder + s + "EEGData/"

        # Initialize arrays to zero for each subject
        X_fin_sub = np.zeros((1, 32))
        X_sub_rms_i = np.zeros((1, 32))
        X_i_tot = np.zeros((1, win_size, 32))
        y_i_tot = np.zeros(1)

        for filename in os.listdir(folder_name):
            print(filename)
            if filename[-8] == 'a':

                # Read eeg data from file
                raw, y_i = read_data_from_file(filename, s)

                # Break signal into parts
                segments, labels, ts = segment_signal_without_transition(raw.get_data(), np.asarray(y_i), win_size)

                # Append to combined data array including all trials for all subjects)
                X_i_tot = np.append(X_i_tot, segments, axis=0)
                y_i_tot = np.append(y_i_tot, labels, axis=0)

                # Take different values of the signal
                X_tr_rms, X_sub_rms_i = preprocess_cleaned_eeg(segments, X_sub_rms_i, rms=True)

        # # If normalizing per subject
        if (norm_by_sub):
            X_sub = X_i_tot[1:]
            y_sub = y_i_tot[1:]
            if not norm_by_seg:
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
                plot_topomap(ch_f_cl, positions, vmin=min0, vmax=max0)

    X_final = X_final[1:]
    y_final = y_final[1:]

    # for clss in ['00', '10', '01', '02', '03', '11', '12', '13']:
    for clss in ['00', '10', '01', '11']:

        X_f_cl = X_final[y_final==clss]
        y_f_cl = y_final[y_final==clss]
        ch_f_cl = np.median(X_f_cl, axis=0)

        print(clss)
        print(np.min(ch_f_cl), np.max(ch_f_cl))
        plot_topomap(ch_f_cl, positions, vmin=-0.4, vmax=0.4)


def plot_single_trial_time():
    # folder_name = EEG_data_folder + "All_Data/"
    folder_name = EEG_data_folder + "B/"

    i = 0
    for filename in os.listdir(folder_name):
        print(filename)
        if filename[-8] == 'a':
            markers_ar = [[0, 0]]

            print(filename)

            raw_theta, ts = create_mne_raw_object(folder_name + filename)
            raw_alpha, ts = create_mne_raw_object(folder_name + filename)
            raw_beta, ts = create_mne_raw_object(folder_name + filename)

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

    return


def wt_full_run(data):
    win_size_0 = 1000     # 4 second window

    # Read eeg data from file
    # raw, y_i = read_data_from_file(filename, s)

    # Break signal into parts
    segments, _, _ = segment_signal_without_transition(data, np.zeros(len(data)), win_size_0)
    # print(segments.shape)

    for i in range(len(segments)):
        segments[i] = wavelet_transform(segments[i])

    data_out = unsegment_signal(segments, win_size_0)

    return data_out


def wavelet_transform(X_t):

    # Continuous wavelet transform can be used to generate a time-frequency plot, but doesn't denoise
    # Ws = morlet(250, [2, 4, 7.5, 15, 30])      # Generates wavelets with which to transform the signal
    # t_signal = cwt(np.transpose(X), Ws)     # Transforms the signal, returning 1 transformed signal for each wavelet

    X = np.transpose(X_t)

    db1 = pywt.Wavelet('db1')
    coeffs = pywt.wavedec(X, db1, level=6)
    for coeff in coeffs:
        coeff[np.abs(coeff) < 1] = 0
    coeffs[0] = np.zeros((32, coeffs[0].shape[1]))
    coeffs[6] = np.zeros((32, coeffs[6].shape[1]))

    t_sig = pywt.waverec(coeffs, db1)

    # plt.subplot(221)
    # plt.plot(X[0])
    # plt.subplot(222)
    # plt.psd(X[0], Fs=250)
    # plt.subplot(223)
    # plt.plot(t_sig[0])
    # plt.subplot(224)
    # plt.psd(t_sig[0], Fs=250)
    # plt.show()

    return np.transpose(t_sig)


# Maybe try a Riemannian transform? Can also just use FFT. Do a comparison of classification accuracy
# need some way to remove EMG artifacts from the signal - follow B-Alert
# Test using leave-one-out (by subject) - train on 9, test on 1

def leave_one_out(num_classes=5):
    trial = False
    act = False

    all_subs = ["A/", "B/", "C/", "D/", "E/", "F/", "G/", "H/", "I/", "Z/"]
    # all_subs = ["G/", "H/", "I/", "Z/"]

    # This loop performs "Leave One Out" cross-validation
    for s_num in range(len(all_subs)):
        training_subs = ["A/", "B/", "C/", "D/", "E/", "F/", "G/", "H/", "I/", "Z/"]
        # training_subs = ["G/", "H/", "I/", "Z/"]
        training_subs.pop(s_num)
        test_sub = all_subs[s_num]
        print(test_sub)

        if trial:
            train_classifier_trial(training_subs)
            test_classifier_trial(test_sub)
        elif act:
            mean_cov_0, mean_cov_1, mean_cov_2, conf_m_train_i = train_classifier_act(training_subs, test_sub)
            conf_m_test_i = test_classifier_act(test_sub, mean_cov_0, mean_cov_1, mean_cov_2)
        else:
            mean_cov_n, conf_m_train_i = train_classifier(training_subs, test_sub, num_classes=num_classes, win_size=win_size)
            conf_m_test_i = test_classifier(test_sub, mean_cov_n, num_classes=num_classes, win_size=win_size)

        if s_num == 0:
            conf_m_train = conf_m_train_i
            conf_m_test = conf_m_test_i
        else:
            conf_m_train += conf_m_train_i
            conf_m_test += conf_m_test_i

        print("Cumulative Training Confusion Matrix: ")
        print(conf_m_train)
        print("Cumulative Testing Confusion Matrix: ")
        print(conf_m_test)

    print("Final Training Confusion Matrix: ")
    print(conf_m_train)
    print("Final Testing Confusion Matrix: ")
    print(conf_m_test)


def get_training_data(training_subs, test_sub, num_classes=5, win_size=500):

    print(win_size)
    try:
        X_i_train = np.load('All_training_data_' + test_sub[0] + '.npy')
        y_i_train = np.load('All_labels_' + test_sub[0] + '.npy')
    except FileNotFoundError:
        X_i_train = np.zeros((1, win_size, 32))
        y_i_train = np.zeros(1)
        for s in training_subs:
            print(test_sub, s)
            print(win_size)
            X_i_train_s = np.zeros((1, win_size, 32))
            y_i_train_s = np.zeros(1)
            folder_name = EEG_data_folder + s + "EEGData/"

            try:
                X_i_train_s = np.load(pp_data_folder + s[0] + '_data_2s_ntb.npy')
                y_i_train_s = np.load(pp_data_folder + s[0] + '_labels_2s_ntb.npy')
                if num_classes == 2:
                    y_i_train_s[y_i_train_s > 0] = 1
            except (FileNotFoundError, NameError):
                for filename in os.listdir(folder_name):
                    print(filename)
                    if filename[-8] == 'a' and is_obstacles(filename, s):

                        # Read eeg data from file - includes cleaning EEG data
                        raw_np, y_i = read_data_from_file(filename, s, num_classes=num_classes)

                        # Break signal into parts
                        print(win_size)
                        segments, labels, ts = segment_signal_without_transition(raw_np, np.asarray(y_i), win_size, overlap=1)

                        # Append to combined data array including all trials for all subjects)
                        X_i_train_s = np.append(X_i_train_s, segments, axis=0)
                        y_i_train_s = np.append(y_i_train_s, labels, axis=0)

                X_i_train_s = X_i_train_s[1:]
                y_i_train_s = y_i_train_s[1:]

                np.save(pp_data_folder + s[0] + '_data_2s_ntb.npy', X_i_train_s)
                np.save(pp_data_folder + s[0] + '_labels_2s_ntb.npy', y_i_train_s)

            X_i_train = np.append(X_i_train, X_i_train_s, axis=0)
            y_i_train = np.append(y_i_train, y_i_train_s, axis=0)

        X_i_train = X_i_train[1:]
        y_i_train = y_i_train[1:]

        # np.save('All_training_data_' + test_sub[0], X_i_train)
        # np.save('All_labels_' + test_sub[0], y_i_train)

    return X_i_train, y_i_train


def get_testing_data(testing_sub, num_classes=5, win_size=500):
    X_i_test = np.zeros((1, win_size, 32))
    y_i_test = np.zeros(1)

    folder_name = EEG_data_folder + testing_sub + "EEGData/"

    try:
        X_i_test = np.load(pp_data_folder + testing_sub[0] + '_data_2s_ntb.npy')
        y_i_test = np.load(pp_data_folder + testing_sub[0] + '_labels_2s_ntb.npy')
        if num_classes == 2:
            y_i_test[y_i_test > 0] = 1
    except FileNotFoundError:
        for filename in os.listdir(folder_name):
            if filename[-8] == 'a' and is_obstacles(filename, testing_sub):
                # Read eeg data from file
                raw_np, y_i = read_data_from_file(filename, testing_sub, num_classes=num_classes)

                # Break signal into parts
                segments, labels, ts = segment_signal_without_transition(raw_np, np.asarray(y_i), win_size, overlap=1)

                # Append to combined data array including all trials for all subjects)
                X_i_test = np.append(X_i_test, segments, axis=0)
                y_i_test = np.append(y_i_test, labels, axis=0)

        X_i_test = X_i_test[1:]
        y_i_test = y_i_test[1:]

        np.save(pp_data_folder + testing_sub[0] + '_data_2s_ntb.npy', X_i_test)
        np.save(pp_data_folder + testing_sub[0] + '_labels_2s_ntb.npy', y_i_test)

    return X_i_test, y_i_test


def train_classifier(training_subs, test_sub, num_classes=5, win_size=500):

    X_i_train, y_i_train = get_training_data(training_subs, test_sub, win_size=win_size)

    unique, counts = np.unique(y_i_train, return_counts=True)
    print("Training Labels: ", unique, counts)

    mean_cov_n, conf_m = train_Riemann(X_i_train, y_i_train, num_classes=num_classes)

    return mean_cov_n, conf_m


def train_classifier_act(training_subs, test_sub, num_classes=5):
    X_i_train, y_i_train = get_training_data(training_subs, test_sub, win_size=win_size)

    unique, counts = np.unique(y_i_train, return_counts=True)
    print("Training Labels: ", unique, counts)

    mean_cov_0, mean_cov_1, mean_cov_2, conf_m = train_Riemann_act(X_i_train, y_i_train)

    return mean_cov_0, mean_cov_1, mean_cov_2, conf_m


def train_classifier_trial(training_subs):
    trial_num = 0
    training_data = []
    trial_labels = []
    for s in training_subs:
        folder_name = EEG_data_folder + s + "EEGData/"
        for filename in os.listdir(folder_name):
            print(filename)
            if filename[-8] == 'a':
                # Read eeg data from file - includes cleaning EEG data
                raw_np, y_i = read_data_from_file(filename, s)

                # Break signal into windows
                segments, labels, ts = segment_signal_without_transition(raw_np, np.asarray(y_i), win_size)

                train_data = select_channels(segments)
                train_data = norm_dataset(train_data)

                cov_trial = pyriemann.estimation.Covariances().fit_transform(np.transpose(train_data, axes=[0, 2, 1]))
                mean_cov_trial = mean_covariance(cov_trial)

                training_data.append(mean_cov_trial)
                trial_labels.append(labels[0])

        # May want to do subject-based normalisation

    training_data = np.asarray(training_data)
    trial_labels = np.asarray(trial_labels, dtype=int)
    mean_cov_n = full_calc_mean_cov(training_data, trial_labels)
    pred_train, cert_train = predict_Riemann(training_data, mean_cov_n)
    print("Training Conf Matrix")
    print(confusion_matrix(trial_labels, pred_train))

    return


def check_SPD_all(data_cov):
    for i in range(len(data_cov)):
        dci=data_cov[i]
        if(is_pos_def(data_cov[i])):
            # print('SPD matrix! ({})'.format(i))
            x = i
        else:
            print('Not SPD ({})'.format(i))
            # data_cov[i] = fix_SPD(data_cov[i])
            # data_cov[i] = nearcorr(data_cov[i])
    return data_cov


def fix_SPD(cov):
    print('fixing...')
    A_eig, A_eig_v = np.linalg.eig(cov)
    A_eig[A_eig < 0] = 1e-10
    # A_eig[A_eig < 0] = 0
    eig_v_real = np.real(A_eig_v)

    # spd_cov = eig_v_real * np.real(A_eig) * np.transpose(eig_v_real)
    spd_cov = A_eig_v * A_eig * np.transpose(A_eig_v)
    is_pos_def(spd_cov)

    return spd_cov


def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            # print("Is SPD!")
            return True
        except np.linalg.LinAlgError:
            # print("Not PD!")
            return False
    else:
        print("Not symmetric!")
        return False


def train_Riemann(train_data, train_labels, num_classes=5):
    fgda = pyriemann.tangentspace.FGDA()
    train_data = select_channels(train_data)
    train_data = norm_dataset(train_data)
    cov_train = pyriemann.estimation.Covariances().fit_transform(np.transpose(train_data, axes=[0, 2, 1]))
    # clf.fit_transform(cov_train, train_labels)    # if certainty is not needed
    # cov_train = fgda.fit_transform(cov_train, train_labels)
    # cov_train = norm_dataset(cov_train)
    cov_train = check_SPD_all(cov_train)
    mean_cov_n = full_calc_mean_cov(cov_train, train_labels, num_classes=num_classes)

    pred_train, cert_train = predict_Riemann(cov_train, mean_cov_n)
    print("Training Conf Matrix")
    conf_m = confusion_matrix(train_labels, pred_train)
    print(conf_m)
    conf_m_out = fix_conf_m(conf_m, train_labels, pred_train)

    return mean_cov_n, conf_m_out


def train_Riemann_act(train_data, train_labels, num_classes=5):
    fgda = pyriemann.tangentspace.FGDA()
    train_data = select_channels(train_data)
    train_data = norm_dataset(train_data)
    cov_train = pyriemann.estimation.Covariances().fit_transform(np.transpose(train_data, axes=[0, 2, 1]))
    cov_train = check_SPD_all(cov_train)

    cov_train_no_act = cov_train[train_labels < num_classes]
    train_labels_no_act = train_labels[train_labels < num_classes]
    cov_train_act = cov_train[train_labels > num_classes - 1]
    train_labels_act = train_labels[train_labels > num_classes - 1]
    cov_train_1 = cov_train_act[train_labels_act < 2*num_classes]
    train_labels_1 = train_labels_act[train_labels_act < 2*num_classes]
    cov_train_2 = cov_train_act[train_labels_act > 2*num_classes - 1]
    train_labels_2 = train_labels_act[train_labels_act > 2*num_classes - 1]

    mean_cov_no_act = full_calc_mean_cov(cov_train_no_act, train_labels_no_act, num_classes=num_classes)
    mean_cov_1_act = full_calc_mean_cov(cov_train_1, train_labels_1 - num_classes, num_classes=num_classes)
    mean_cov_2_act = full_calc_mean_cov(cov_train_2, train_labels_2 - 2*num_classes, num_classes=num_classes)

    pred_train, cert_train = predict_Riemann_act(cov_train, train_labels, mean_cov_no_act, mean_cov_1_act, mean_cov_2_act)
    print("Training Conf Matrix")
    conf_m = confusion_matrix(train_labels, pred_train)
    print(conf_m)
    conf_m_out = fix_conf_m(conf_m, train_labels, pred_train, num_classes=3*num_classes)

    return mean_cov_no_act, mean_cov_1_act, mean_cov_2_act, conf_m_out


def fix_conf_m(conf_m, labels_t, labels_p, num_classes=5):
    conf_m_out = copy.copy(conf_m)
    if conf_m_out.shape[0] < num_classes:
        unique_tl, counts = np.unique(labels_t, return_counts=True)
        unique_pl, counts = np.unique(labels_p, return_counts=True)
        for i in range(num_classes):
            if i not in unique_tl and i not in unique_pl:
                conf_m_out = np.r_[conf_m_out[:i], np.zeros((1, conf_m_out.shape[1]), dtype=int), conf_m_out[i:]]
                conf_m_out = np.c_[conf_m_out[:, :i], np.zeros((conf_m_out.shape[0], 1), dtype=int), conf_m_out[:, i:]]

    return conf_m_out


def full_calc_mean_cov(cov_train_i, label_train_i, num_classes=5):

    num_classes = int(np.max(label_train_i)) + 1
    mean_cov_n = np.zeros((num_classes, cov_train_i.shape[1], cov_train_i.shape[2]))

    for l in range(num_classes):
        # try:
        # print(l)
        mean_cov_n[l] = mean_covariance(cov_train_i[label_train_i == l], metric='riemann',
                                            sample_weight=None)
        # print_distances(l, cov_train_i[label_train_i == l], mean_cov_n)

    return mean_cov_n


def print_distances(l, cov_mat, mean_cov_n, num_classes=5):
    print(l)
    class_len = cov_mat.shape[0]
    list_distances = np.zeros(class_len)
    list_distances_own = np.zeros(class_len)
    for i in range(class_len):
        list_distances[i] = distance(mean_cov_n[0], cov_mat[i], metric='riemann')
        list_distances_own[i] = distance(mean_cov_n[l], cov_mat[i], metric='riemann')

    """this_dist = distance(mean_cov_n[0], mean_cov_n[l], metric='riemann')
    print("Distance between averages: ")
    print(this_dist)
    print("Average of distances to class 0:")
    stderr = np.std(list_distances) / np.sqrt(list_distances.shape[0])
    print(np.mean(list_distances), np.std(list_distances), stderr)
    print("Median:")
    print(np.median(list_distances))
    print("Average of distances to own class: ")
    stderr_own = np.std(list_distances_own) / np.sqrt(list_distances_own.shape[0])
    print(np.mean(list_distances_own), np.std(list_distances_own), stderr_own)"""
    print("Median of distances to own class: ")
    print(np.median(list_distances_own))


def test_classifier(testing_sub, mean_cov_n, num_classes=5, win_size=500):

    X_i_test, y_i_test = get_testing_data(testing_sub, win_size=win_size)

    # Get "certainty" value for each time point (can be plotted over time)
    # Determine final classification for each file

    # # CSP > LDA classifier - choosing class 0 every time
    # clf_preds = clf.predict(np.transpose(X_i_test, axes=[0, 2, 1]))
    # print(confusion_matrix(y_i_test, clf_preds))

    # Print classification accuracy (confusion matrix) for all files for the given test subject

    unique, counts = np.unique(y_i_test, return_counts=True)
    print("Test Labels: ", unique, counts)
    X_i_test = select_channels(X_i_test)
    X_i_test = norm_dataset(X_i_test)
    cov_test = pyriemann.estimation.Covariances().fit_transform(np.transpose(X_i_test, axes=[0, 2, 1]))
    pred, cert = predict_Riemann(cov_test, mean_cov_n)

    conf_m = confusion_matrix(y_i_test, pred)
    print(conf_m)
    conf_m_out = fix_conf_m(conf_m, y_i_test, pred, num_classes=num_classes)

    return conf_m_out


def test_classifier_trial(testing_sub, mean_cov_n):
    folder_name = EEG_data_folder + testing_sub + "EEGData/"
    testing_data = []
    trial_labels = []

    for filename in os.listdir(folder_name):
        if filename[-8] == 'a':
            # Read eeg data from file
            raw_np, y_i = read_data_from_file(filename, testing_sub)

            # Break signal into parts
            segments, labels, ts = segment_signal_without_transition(raw_np, np.asarray(y_i), win_size)

            test_data = select_channels(segments)
            test_data = norm_dataset(test_data)

            cov_trial = pyriemann.estimation.Covariances().fit_transform(np.transpose(test_data, axes=[0, 2, 1]))
            mean_cov_trial = mean_covariance(cov_trial)

            testing_data.append(mean_cov_trial)
            trial_labels.append(labels[0])

    testing_data = np.asarray(testing_data)
    trial_labels = np.asarray(trial_labels, dtype=int)

    # Print classification accuracy (confusion matrix) for all files for the given test subject
    unique, counts = np.unique(trial_labels, return_counts=True)
    print("Test Labels: ", unique, counts)
    X_i_test = select_channels(testing_data)
    X_i_test = norm_dataset(X_i_test)
    cov_test = pyriemann.estimation.Covariances().fit_transform(np.transpose(X_i_test, axes=[0, 2, 1]))
    pred, cert = predict_Riemann(cov_test, mean_cov_n)

    print(confusion_matrix(trial_labels, pred))

    return


def test_classifier_act(testing_sub, mean_cov_no_act, mean_cov_1_act, mean_cov_2_act, num_classes=5):

    X_i_test, y_i_test = get_testing_data(testing_sub, win_size=win_size)

    unique, counts = np.unique(y_i_test, return_counts=True)
    print("Test Labels: ", unique, counts)
    X_i_test = select_channels(X_i_test)
    X_i_test = norm_dataset(X_i_test)
    cov_test = pyriemann.estimation.Covariances().fit_transform(np.transpose(X_i_test, axes=[0, 2, 1]))
    pred, cert = predict_Riemann_act(cov_test, y_i_test, mean_cov_no_act, mean_cov_1_act, mean_cov_2_act)

    print("Test Confusion Matrix: ")
    conf_m = confusion_matrix(y_i_test, pred)
    print(conf_m)

    conf_m_out = fix_conf_m(conf_m, y_i_test, pred, num_classes=3 * num_classes)

    return conf_m_out


def predict_Riemann(covtest, mean_cov_n):
    dist = predict_distances_own(covtest, mean_cov_n)
    cert = (dist.mean(axis=1) - dist.min(axis=1))*4.0

    return dist.argmin(axis=1), cert


def predict_Riemann_act(covtest, y_test, mean_cov_no_act, mean_cov_1_act, mean_cov_2_act, num_classes=5):

    pred = np.zeros(y_test.shape[0])
    cert = np.zeros(y_test.shape[0])

    for i in range(len(y_test)):
        if y_test[i] < num_classes:
            dist = predict_distances_own(covtest[i], mean_cov_no_act)
            pred[i] = dist.argmin()
            cert[i] = (dist.mean() - dist.min()) * 4.0
        elif y_test[i] < 2*num_classes:
            dist = predict_distances_own(covtest[i], mean_cov_1_act)
            pred[i] = dist.argmin(axis=0) + num_classes
            cert[i] = (dist.mean() - dist.min()) * 4.0
        else:
            dist = predict_distances_own(covtest[i], mean_cov_2_act)
            pred[i] = dist.argmin(axis=0) + 2 * num_classes
            cert[i] = (dist.mean() - dist.min()) * 4.0

    return pred, cert


def predict_distances_own(covtest, mean_cov_n):

    covmeans = mean_cov_n
    Nc = len(covmeans)
    dist = [distance(covtest, covmeans[m], 'riemann') for m in range(Nc)]
    dist = np.asarray(dist)
    if np.ndim(dist) > 1:
        dist = np.concatenate(dist, axis=1)

    return dist


def calc_all(num_classes=5):
    training_subs = ["A/", "B/", "C/", "D/", "E/", "F/", "G/", "H/", "I/", "Z/"]
    test_sub = "A/"
    train_classifier(training_subs, test_sub, num_classes=num_classes)


def print_distances_by_trial(num_classes=5):
    training_subs = ["A/", "B/", "C/", "D/", "E/", "F/", "G/", "H/", "I/", "Z/"]
    test_sub = "A/"
    mean_cov_n, conf_m = train_classifier(training_subs, test_sub, num_classes=num_classes)

    # X_i_test, y_i_test = get_testing_data(test_sub)
    for s in training_subs:
        folder_name = EEG_data_folder + s + "EEGData/"
        for filename in os.listdir(folder_name):
            if filename[-8] == 'a' and is_obstacles(filename, s):
                # Read eeg data from file
                raw_np, y_i = read_data_from_file(filename, s)

                # Break signal into parts
                segments, labels, ts = segment_signal_without_transition(raw_np, np.asarray(y_i), win_size, overlap=1)

                # Convert to SPD matrices
                train_data = select_channels(segments)
                train_data = norm_dataset(train_data)
                cov_mat = pyriemann.estimation.Covariances().fit_transform(np.transpose(train_data, axes=[0, 2, 1]))

                mean_cov_trial = mean_covariance(cov_mat, metric='riemann', sample_weight=None)

                class_len = cov_mat.shape[0]
                list_distances = np.zeros(class_len)
                list_distances_own = np.zeros(class_len)
                list_distances_trial = np.zeros(class_len)

                for i in range(class_len):
                    list_distances[i] = distance(mean_cov_n[0], cov_mat[i], metric='riemann')
                    list_distances_own[i] = distance(mean_cov_n[int(labels[i])], cov_mat[i], metric='riemann')
                    list_distances_trial[i] = distance(mean_cov_trial, cov_mat[i])

                dist_trial = distance(mean_cov_n[0], mean_cov_trial, metric='riemann')

                print(s)
                print(filename[11:16])
                # print("Distance of trial average to class 0 average: ")
                # print(dist_trial)
                # print("Average distance to trial average: ")
                # print(np.mean(list_distances_trial))
                print("Median distance to trial average: ")
                print(np.median(list_distances_trial))
                """print("Average of distances to class 0:")
                stderr = np.std(list_distances) / np.sqrt(list_distances.shape[0])
                print(np.mean(list_distances), np.std(list_distances), stderr)
                print("Median:")
                print(np.median(list_distances))
                print("Average of distances to own class: ")
                stderr_own = np.std(list_distances_own) / np.sqrt(list_distances_own.shape[0])
                print(np.mean(list_distances_own), np.std(list_distances_own), stderr_own)"""


if __name__ == "__main__":

    global markers_ar
    pp_data_folder = 'D:/PhD_Data/CanadarmTask/Processed_Data/'

    win_size = 500
    # win_size = 250

    X_i_tot = np.zeros((1, win_size, 32))
    y_i_tot_lat = np.zeros(1)
    y_i_tot_tp = np.zeros(1)

    leave_one_out(num_classes=2)
    # calc_all(num_classes=5)
    # print_distances_by_trial(num_classes=5)

    # EEG_data_folder = "D:/PhD_Data/CanadarmTask/"
    # s = "G/"
    # folder_name = EEG_data_folder + s + "EEGData/"
    # test_fn = os.listdir(folder_name)[0]

    # # Read eeg data from file
    # raw, y_i = read_data_from_file(test_fn)
    # data = clean_EEG(raw)

    # # Break signal into parts
    # segments, labels, ts = segment_signal_without_transition(data, np.asarray(y_i), win_size)

    # wt_full_run(test_fn)
    # plot_combined_topomaps(EEG_data_folder)
    # plot_single_trial_time()

    # ret, y = run_csp()
    # svc = SVC()
    # svc.fit(ret, y)
    # print("SVC score: ")
    # print(svc.score(ret, y))

    # plot.ax_scalp(csp.filters_[0], ch_names)
