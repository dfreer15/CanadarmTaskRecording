import real_time_BCI_train as real_time_BCI
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
import numpy as np
import atexit
import signal

import time

if __name__ == '__main__':
    freq = 250
    window_size = 0.5

    num_channels = 32
    buffer_size = int(freq*window_size)

    vrep = False
    feedback = False
    markers = True
    experiment_going = True  # How to indicate when the experiment has stopped?
    filter_rt = False
    
    subject_name = 'A'
    mode = 'space'

    real_time_BCI.init_globals_2(user_name=subject_name)
    real_time_BCI.get_test_data()  # if using prerecorded data
    real_time_BCI.init_receiver()
    if markers:
        real_time_BCI.init_marker_receiver()

    buffer_data = np.zeros((buffer_size, num_channels))
    iter_n = 0

    atexit.register(real_time_BCI.save_data, feedback=feedback, vrep=vrep, markers=markers, mode=mode)

    t_start = time.perf_counter()

    while experiment_going:
        label = []
        buffer_data = real_time_BCI.iter_bci_buffer(buffer_data, iter_n)
        if markers:
            real_time_BCI.read_bci_markers()

        iter_n += 1

