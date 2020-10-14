import EEG_postprocess as EEG_p
import pandas as pd
import os
import numpy as np
import pyriemann
import matplotlib.pyplot as plt


def read_segment_channels_norm(fname):
    raw, y = EEG_p.read_data_from_file(fname, test_sub)
    segments, labels, ts = EEG_p.segment_signal_without_transition(raw, np.asarray(y), win_size, overlap=1)
    data = EEG_p.select_channels(segments)
    data = EEG_p.norm_dataset(data)

    return data, ts


def create_beta_wave(fname, w_size=500):
    beta, ts = EEG_p.create_mne_raw_object(fname)
    beta = EEG_p.clean_EEG_ICA(beta)
    beta.filter(12.5, 30, method='iir', verbose=False)
    segments, labels, ts_s = EEG_p.segment_signal_without_transition(beta.get_data(), np.zeros(beta.get_data().shape[1]), w_size, overlap=1)
    data = EEG_p.select_channels(segments)
    power = calculate_power(data)
    # power = EEG_p.smooth_plot(power, sliding_win=4)

    return power


def calculate_power(data):
    power = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        power[i] = np.sqrt(np.mean(np.power(data[i], 2)))

    return power


def calc_distance_time(data, mean_cov_n):
    distance = np.zeros(data.shape[0])
    cov_mats = pyriemann.estimation.Covariances().fit_transform(np.transpose(data, axes=[0, 2, 1]))
    for i in range(data.shape[0]):
        distance[i] = EEG_p.distance(cov_mats[i], mean_cov_n[0], metric='riemann')

    # distance = EEG_p.smooth_plot(distance)

    return distance


def plot_markers(markers_ar):
    for mrkr, event in markers_ar:
        if mrkr > 0:
            if int(event) == -51 or int(event) == -52 or int(event) == -53 or int(event) == -55:
                plt.axvline(x=mrkr, color='red')
            elif int(event) == markers_ar[-1,1]:
                plt.axvline(x=mrkr-9, color='black')
            else:
                plt.axvline(x=mrkr, color='black')


def plot_beta_Rdist():
    print(EEG_data_folder + lw_trial[:-15] + 'markers.csv')
    mrkrs_lw = np.asarray(pd.read_csv(folder_name + lw_trial[:-15] + 'markers.csv', header=None))
    mrkrs_tp = np.asarray(pd.read_csv(folder_name + tp_trial[:-15] + 'markers.csv', header=None))
    mrkrs_lat = np.asarray(pd.read_csv(folder_name + lat_trial[:-15] + 'markers.csv', header=None))
    mrkrs_tp_lat = np.asarray(pd.read_csv(folder_name + tp_lat_trial[:-15] + 'markers.csv', header=None))

    lw_beta = create_beta_wave(folder_name + lw_trial)

    data_lw, ts_lw = read_segment_channels_norm(lw_trial)
    data_tp, ts_tp = read_segment_channels_norm(tp_trial)
    data_lat, ts_lat = read_segment_channels_norm(lat_trial)
    data_tp_lat, ts_tp_lat = read_segment_channels_norm(tp_lat_trial)

    mrkrs_lw[:, 0] -= ts_lw[0]
    mrkrs_tp[:, 0] -= ts_tp[0]
    mrkrs_lat[:, 0] -= ts_lat[0]
    mrkrs_tp_lat[:, 0] -= ts_tp_lat[0]

    ts_lw = ts_lw - ts_lw[0]
    ts_tp = ts_tp - ts_tp[0]
    ts_lat = ts_lat - ts_lat[0]
    ts_tp_lat = ts_tp_lat - ts_tp_lat[0]

    # Shorten the time array if smoothing
    ts_lw = ts_lw[3:]
    ts_tp = ts_tp[3:]
    ts_lat = ts_lat[3:]
    ts_tp_lat = ts_tp_lat[3:]

    lw_beta = create_beta_wave(folder_name + lw_trial)
    tp_beta = create_beta_wave(folder_name + tp_trial)
    lat_beta = create_beta_wave(folder_name + lat_trial)
    tp_lat_beta = create_beta_wave(folder_name + tp_lat_trial)

    mean_cov_n, conf_m = EEG_p.train_classifier(training_subs, test_sub, num_classes=num_classes, win_size=500)

    dist_lw = calc_distance_time(data_lw, mean_cov_n)
    dist_tp = calc_distance_time(data_tp, mean_cov_n)
    dist_lat = calc_distance_time(data_lat, mean_cov_n)
    dist_tp_lat = calc_distance_time(data_tp_lat, mean_cov_n)

    plt.subplot(411)
    axes = plt.gca()
    axes.set_ylim([0, 12])
    axes.set_xlim([ts_lw[0], mrkrs_lw[-1, 0] + 3])
    p = plt.plot(ts_lw, lw_beta, ts_lw, dist_lw)
    # p = plt.plot(ts_lw, dist_lw)
    plt.title('Low Workload')
    # plt.ylabel('Low Workload')
    plt.legend((p[0], p[1]), ('beta', 'Rdist'), loc='lower right')
    plot_markers(mrkrs_lw)

    plt.subplot(412)
    axes = plt.gca()
    axes.set_ylim([0, 12])
    axes.set_xlim([ts_tp[0], mrkrs_tp[-1, 0] + 3])
    p = plt.plot(ts_tp, tp_beta, ts_tp, dist_tp)
    # p = plt.plot(ts_tp, dist_tp)
    plt.title('Time Pressure')
    # plt.ylabel('EE position (m)')
    plt.legend((p[0], p[1]), ('beta', 'Rdist'), loc='lower right')
    plot_markers(mrkrs_tp)

    plt.subplot(413)
    axes = plt.gca()
    axes.set_ylim([0, 12])
    axes.set_xlim([ts_lat[0], mrkrs_lat[-1, 0] + 3])
    p = plt.plot(ts_lat, lat_beta, ts_lat, dist_lat)
    # p = plt.plot(ts_lat, dist_lat)
    plt.title('Latency (0.5 s)')
    # plt.ylabel('EE position (m)')
    plt.legend((p[0], p[1]), ('beta', 'Rdist'), loc='lower right')
    plot_markers(mrkrs_lat)

    plt.subplot(414)
    axes = plt.gca()
    axes.set_ylim([0, 12])
    axes.set_xlim([ts_tp_lat[0], mrkrs_tp_lat[-1, 0] + 3])
    p = plt.plot(ts_tp_lat, tp_lat_beta, ts_tp_lat, dist_tp_lat)
    # p = plt.plot(ts_tp_lat, dist_tp_lat)
    plt.title('Time Pressure + Latency (0.5 s)')
    # plt.ylabel('EE position (m)')
    plt.legend((p[0], p[1]), ('beta', 'Rdist'), loc='lower right')
    plot_markers(mrkrs_tp_lat)

    plt.tight_layout()
    plt.show()


def beta_eval():
    global folder_name

    training_subs = ["A/", "B/", "C/", "D/", "E/", "F/", "G/", "H/", "I/", "Z/"]
    test_sub = "A/"

    for s in training_subs:
        folder_name = EEG_data_folder + s + "EEGData/"
        for filename in os.listdir(folder_name):
            if filename[-8] == 'a':
                # Read eeg data from file
                beta_pow = create_beta_wave(folder_name + filename)

                print(s)
                print(filename[11:16])
                print("Beta Power: ")
                print(np.mean(beta_pow), np.std(beta_pow))


def plot_beta_Rdist_trials_ot():
    beta_pow_trial = [.729954, .735052, .440238, .24249, .3422, .158675, .415261, .247687, .210361, .18922, .16885,
                      .215335, .369673, .427318, .381162, .190167, .138874, .223788, .290054, .050904, .101202]
    beta_pow_stddev = [.293827, .312416, .249727, .170191, .266355, .114009, .265347, .296154, .181418, .166891,
                       .127342, .104749, .253275, .34137, .22832, .150485, .076541, .160436, .225022, .050904, .013524]
    Rdist_trial = [.477835, .533841, .440494, .411098, .386497, .378413, .578539, .479262, .375602, .355359, .4725,
                   .373104, .334085, .228522, .44886, .405715, .383381, .385045, .611897, .404137, .208156]
    Rdist_stddev = [.32252, .364171, .332694, .375375, .27884, .260682, .264248, .35831, .351442, .298106, .299514,
                    .284749, .28455, .259174, .280657, .365843, .306067, .217292, .236461, .171885, .081587]
    PM_trial = [.382509, .450562, .282605, .535927, .489159, .561206, .475742, .488333, .463676, .535178, .483714,
                .447501, .651394, .719946, .516354, .641422, .557263, .488096, .367015, .726965, .859308]
    PM_stddev = [.353887, .261472, .221678, .28251, .235443, .295029, .287084, .234614, .245549, .275825, .202941,
                .21077, .195779, .271804, .343471, .208113, .315043, .145252, .582174, .049906, .00804]
    divide_vec = np.sqrt([10., 10., 10., 10., 10., 10., 10., 10., 10., 9., 9., 9., 8., 7., 6., 6., 6., 6., 5., 2., 2.])

    bpt = np.asarray(beta_pow_trial)
    bps = 2 * np.asarray(beta_pow_stddev)/divide_vec
    rdt = np.asarray(Rdist_trial)
    rds = 2 * np.asarray(Rdist_stddev)/divide_vec
    pmt = 1 - np.asarray(PM_trial)
    pms = 2 * np.asarray(PM_stddev)/divide_vec

    x = np.linspace(0, bpt.shape[0], num=bpt.shape[0])

    p1 = plt.plot(x, bpt, color='r')
    plt.fill_between(x, bpt - bps, bpt+bps, color='r', alpha=0.2)
    p2 = plt.plot(x, rdt, color='b')
    plt.fill_between(x, rdt - rds, rdt + rds, color='b', alpha=0.2)
    p3 = plt.plot(x, pmt, color='k')
    plt.fill_between(x, pmt + pms, pmt - pms, color='k', alpha=0.1)
    plt.xlabel("Trial Number")
    plt.ylabel("Normalised Beta Power / $d_0$ / Score")
    plt.legend((p1[0], p2[0], p3[0]), ('Beta', '$d_0$', '1-Score'))

    plt.show()

    return


if __name__ == "__main__":

    EEG_data_folder = "D:/PhD_Data/CanadarmTask/"
    num_classes = 5
    training_subs = ["A/", "B/", "C/", "D/", "E/", "F/", "G/", "H/", "I/", "Z/"]
    test_sub = "B/"
    win_size = 500

    folder_name = EEG_data_folder + test_sub + "EEGData/"

    lw_trial = '2019-09-03_15-25-34-725450_EEG_data_00.csv'
    tp_trial = '2019-09-03_15-36-32-731913_EEG_data_10.csv'
    lat_trial = '2019-09-03_15-46-44-246919_EEG_data_01.csv'
    tp_lat_trial = '2019-09-03_16-13-06-356586_EEG_data_11.csv'

    # plot_beta_Rdist()
    plot_beta_Rdist_trials_ot()
    # beta_eval()



