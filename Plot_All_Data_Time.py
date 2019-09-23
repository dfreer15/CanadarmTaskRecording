import numpy as np
import pandas as pd
import EEG_postprocess
import matplotlib.pyplot as plt
import random
import csv


def preprocess_eeg(eeg):
    # # Pre-process eeg signals
    # print(eeg)
    raw_theta = EEG_postprocess.create_mne_raw_object(eeg)
    raw_alpha = EEG_postprocess.create_mne_raw_object(eeg)
    raw_beta = EEG_postprocess.create_mne_raw_object(eeg)

    raw_theta = EEG_postprocess.clean_EEG_ICA(raw_theta)
    raw_alpha = EEG_postprocess.clean_EEG_ICA(raw_alpha)
    raw_beta = EEG_postprocess.clean_EEG_ICA(raw_beta)

    raw_theta.filter(4, 7.5, method='iir', verbose=False)
    raw_alpha.filter(7.5, 12.5, method='iir', verbose=False)
    raw_beta.filter(12.5, 30, method='iir', verbose=False)

    y = [0]
    y_i = y * (raw_alpha.get_data().shape[1])

    win_size_eeg = 500
    segments_theta, labels_lat, ts = EEG_postprocess.segment_signal_without_transition(raw_theta.get_data(), np.asarray(y_i), win_size_eeg)
    segments_alpha, labels_tp, ts = EEG_postprocess.segment_signal_without_transition(raw_alpha.get_data(), np.asarray(y_i), win_size_eeg)
    segments_beta, labels_tp, ts = EEG_postprocess.segment_signal_without_transition(raw_beta.get_data(), np.asarray(y_i), win_size_eeg)



    return segments_theta, segments_alpha, segments_beta, ts


def calc_power(data_in_th, data_in_al, data_in_beta, ts):
    seg_power_th = np.zeros((data_in_th.shape[0], 32))
    segments_th = np.transpose(data_in_th, axes=[0, 2, 1])
    seg_power_al = np.zeros((data_in_al.shape[0], 32))
    segments_al = np.transpose(data_in_al, axes=[0, 2, 1])
    seg_power_beta = np.zeros((data_in_beta.shape[0], 32))
    segments_beta = np.transpose(data_in_beta, axes=[0, 2, 1])

    for k in range(len(ts)):
        seg_power_th[k] = np.sqrt(np.mean(np.square(segments_th[k]), axis=1))
        seg_power_al[k] = np.sqrt(np.mean(np.square(segments_al[k]), axis=1))
        seg_power_beta[k] = np.sqrt(np.mean(np.square(segments_beta[k]), axis=1))

    frontal_lobe_power_ar_th = seg_power_th[:, :10]
    frontal_lobe_power_th = np.mean(frontal_lobe_power_ar_th, axis=1)
    frontal_lobe_power_ar_al = seg_power_al[:, :10]
    frontal_lobe_power_al = np.mean(frontal_lobe_power_ar_al, axis=1)
    frontal_lobe_power_ar_beta = seg_power_beta[:, :10]
    frontal_lobe_power_beta = np.mean(frontal_lobe_power_ar_beta, axis=1)

    flpt_smoothed = EEG_postprocess.smooth_plot(frontal_lobe_power_th)
    flpa_smoothed = EEG_postprocess.smooth_plot(frontal_lobe_power_al)
    flpb_smoothed = EEG_postprocess.smooth_plot(frontal_lobe_power_beta)

    ts = ts[3:]

    data_out = np.append(ts, flpt_smoothed)
    data_out = np.append(data_out, flpa_smoothed)
    data_out = np.append(data_out, flpb_smoothed)
    data_out = data_out.reshape((4, len(ts)))
    data_out = np.transpose(data_out)

    print(data_out.shape)

    with open("/data2/processed_EEG_data.csv", 'a') as out_file:
        out_writer = csv.writer(out_file, delimiter=',', quotechar='"')

        out_writer.writerows(data_out)

    return flpt_smoothed, flpa_smoothed, flpb_smoothed, ts


def plot_markers():
    for mrkr, event in markers_ar:
        if mrkr > 0:
            if int(event) == -51 or int(event) == -52 or int(event) == -53 or int(event) == -55:
                plt.axvline(x=mrkr, color='red')
            elif int(event) == markers_ar[-1,1]:
                plt.axvline(x=mrkr-9, color='black')
            else:
                plt.axvline(x=mrkr, color='black')


def add_noise(data, scale=0.1):
    noise = np.random.normal(0,scale,len(data))
    data = data + noise

    return data




if __name__ == "__main__":

    rt_folder = '/data2/SpaceTrial092019/Real_Time_Data/'
    jp = pd.read_csv(rt_folder + 'Yao_rt_JPA.csv')
    bf = pd.read_csv(rt_folder + 'Yao_blink_freq.csv', header=None)
    ff = pd.read_csv(rt_folder + 'Yao_fixation_freq.csv', header=None)
    eeg_path = rt_folder + 'Yao_rt_EEG_data.csv'
    # eeg = pd.read_csv(rt_folder + 'Yao_rt_EEG_data.csv', header=None)
    phys = pd.read_csv(rt_folder + 'Yao_trial_phys_data.csv')
    mrkrs = pd.read_csv(rt_folder + 'Yao_rt_markers.csv', header=None)
    markers_ar = np.asarray(mrkrs)

    # Process Joint Position Data
    jp_ts = jp.values[:, 0]
    joint_pos = jp.values[:, 22:25]

    # Process EEG data
    segments_theta, segments_alpha, segments_beta, ts = preprocess_eeg(eeg_path)
    flpt, flpa, flpb, ts = calc_power(segments_theta, segments_alpha, segments_beta, ts)

    # Process gaze data
    blink_freq = bf.values[10:, 0]
    fix_freq = ff.values[10:, 0]
    gaze_ts = bf.values[10:, 1]
    gaze_ts_0 = gaze_ts[0]
    gaze_ts = gaze_ts - gaze_ts_0
    ts = ts - gaze_ts_0
    markers_ar[:,0] = markers_ar[:,0] - gaze_ts_0

    # Process Physiological data
    time_phys = phys.values[:,0]
    time_phys = time_phys - time_phys[17]
    time_phys = time_phys[3:]
    heartrate = phys.values[:,1]
    heartrate = add_noise(heartrate)
    heartrate = EEG_postprocess.smooth_plot(heartrate, sliding_win=10)
    temp = phys.values[:,2]
    temp = add_noise(temp)
    temp = EEG_postprocess.smooth_plot(temp, sliding_win=10)
    gsr = phys.values[:,3]
    gsr = add_noise(gsr, scale=1)
    gsr = EEG_postprocess.smooth_plot(gsr, sliding_win=10)
    acc = phys.values[:,4:]
    acc_x = EEG_postprocess.smooth_plot(acc[:,0])
    acc_y = EEG_postprocess.smooth_plot(acc[:,1])
    acc_z = EEG_postprocess.smooth_plot(acc[:,2])

    plt.subplot(711)
    axes = plt.gca()
    # axes.set_ylim([1, 3])
    axes.set_xlim([jp_ts[0], markers_ar[-1, 0]])
    p = plt.plot(jp_ts, joint_pos)
    plt.ylabel('EE position (m)')
    plt.legend((p[0], p[1], p[2]), ('x', 'y', 'z'), loc='lower right')
    plot_markers()

    plt.subplot(712)
    axes = plt.gca()
    axes.set_ylim([1.5, 3])
    axes.set_xlim([ts[0], markers_ar[-1, 0]])
    p_th = plt.plot(ts, flpt)
    p_al = plt.plot(ts, flpa)
    p_be = plt.plot(ts, flpb)
    plt.ylabel('EEG RMS power')
    plt.legend((p_th[0], p_al[0], p_be[0]), ('theta', 'alpha', 'beta'), loc='lower right')
    plot_markers()

    plt.subplot(713)
    axes=plt.gca()
    axes.set_ylim([0, 1])
    axes.set_xlim([gaze_ts[0], markers_ar[-1, 0]])
    p_bf = plt.plot(gaze_ts, blink_freq)
    p_ff = plt.plot(gaze_ts, fix_freq)
    plt.ylabel('Frequency (Hz)')
    plt.legend((p_bf[0], p_ff[0]), ('blink', 'fixation'), loc='lower right')
    plot_markers()

    plt.subplot(714)
    axes=plt.gca()
    axes.set_ylim([72,80])
    axes.set_xlim([0, markers_ar[-1, 0]])
    plt.plot(time_phys, heartrate)
    plt.ylabel('Heartrate (bps)')
    plot_markers()

    plt.subplot(715)
    axes = plt.gca()
    axes.set_ylim([27.5, 29])
    axes.set_xlim([0, markers_ar[-1, 0]])
    plt.plot(time_phys, temp)
    plt.ylabel('Temperature ($^\circ$C)')
    plot_markers()

    plt.subplot(716)
    axes = plt.gca()
    axes.set_ylim([565, 575])
    axes.set_xlim([0, markers_ar[-1, 0]])
    plt.plot(time_phys, gsr)
    plt.ylabel('Resistance (k$\Omega$)')
    plot_markers()

    plt.subplot(717)
    axes = plt.gca()
    # axes.set_ylim([70, 80])
    axes.set_xlim([0, markers_ar[-1, 0]])
    p_x = plt.plot(time_phys, acc_x)
    p_y = plt.plot(time_phys, acc_y)
    p_z = plt.plot(time_phys, acc_z)
    plt.xlabel('Time from Start of Task (s)')
    plt.ylabel('Human Accel. ($m/s^2$)')
    plt.legend((p_x[0], p_y[0], p_z[0]), ('x', 'y', 'z'), loc='lower right')
    plot_markers()

    plt.show()