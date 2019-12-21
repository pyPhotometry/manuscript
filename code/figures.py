import os
import numpy as np 
import pylab as plt 
from functools import partial
from matplotlib.gridspec import GridSpec
from scipy.signal import windows, butter, filtfilt
from scipy.io import loadmat

import data_import as di
import pyControl_import as ci
import rsync as rs

plt.rcParams['pdf.fonttype'] = 42
plt.rc("axes.spines", top=False, right=False)

data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data')

# ----------------------------------------------------------------------------------------
# Functions for generating figures.
# ----------------------------------------------------------------------------------------

# Figure 3

def Time_division_figure():
    '''Generate figure 3 panels B-D. Showing: B) photoreciever voltage response to 4ms light
    light pulse. C) Acquisition sequence for time division illumination.  D) Response of 
    baseline subtracted time division readout to in phase, anti-phase and continous light.'''
 
    # Photoreciever response to 4ms light pulse.
    def pulse_response_plot(datafolder):
        scope_trace(datafolder, trace_index=2, baseline=True)
        scope_trace(datafolder, trace_index=1, color='k', baseline=True,
                       y_offset=3.5, scale_y=0.3)
        plt.xlim(-1,8)
        plt.plot([-1,8],[0,0], 'k--', linewidth=1)
    traces_4ms   = os.path.join(data_dir, 'Time division','pulse response', '4ms')
    plt.figure(2, figsize=[9, 3]).clf()
    plt.subplot(1,3,1)
    pulse_response_plot(traces_4ms)

    # Acqusition sequence plot.
    sequence_dir = os.path.join(data_dir, 'Time division','acquisition sequence')
    LED_1_2    = os.path.join(sequence_dir, 'LED 1, LED 2')
    baseline_1 = os.path.join(sequence_dir, 'LED 1, baseline ADC 1')
    baseline_2 = os.path.join(sequence_dir, 'LED 1, baseline ADC 2')
    sample_1   = os.path.join(sequence_dir, 'LED 1, sample ADC 1')
    sample_2   = os.path.join(sequence_dir, 'LED 1, sample ADC 2')
    signal_1   = os.path.join(sequence_dir, 'LED 1, signal 1')
    signal_2   = os.path.join(sequence_dir, 'LED 1, signal 2')
    plt.subplot(1,3,2)
    scope_trace(signal_1  , trace_index=2, color='b', scale_y=1.  , y_offset=0   , baseline=False)
    scope_trace(sample_1  , trace_index=2, color='k', scale_y=0.15, y_offset=3.5 , baseline=False)
    scope_trace(baseline_1, trace_index=2, color='k', scale_y=0.15, y_offset=4.2 , baseline=False)
    scope_trace(LED_1_2   , trace_index=1, color='k', scale_y=0.15, y_offset=4.9 , baseline=False)
    scope_trace(signal_2  , trace_index=2, color='r', scale_y=1.  , y_offset=-6  , baseline=False)
    scope_trace(sample_2  , trace_index=2, color='k', scale_y=0.15, y_offset=-2.5, baseline=False)
    scope_trace(baseline_2, trace_index=2, color='k', scale_y=0.15, y_offset=-1.8, baseline=False)
    scope_trace(LED_1_2   , trace_index=2, color='k', scale_y=0.15, y_offset=-1.1, baseline=False)
    
    # Linearity plot
    linearity_datafolder = os.path.join(data_dir, 'Time division','linearity')
    plt.subplot(1,3,3)
    Time_division_linearity(linearity_datafolder)
    plt.tight_layout()


def Time_frequency_division_comparison():
    '''Figure 3 panels E-F comparing time and frequency division illumination.'''
    plt.figure(4, figsize=[6,3]).clf()
    plt.subplot(1,2,1)
    frequency_division_simulation()
    plt.subplot(1,2,2)
    FD_TD_noise_comparison(lowpass=20)
    plt.tight_layout()    

# Figure 4 ------------------------------------------------------------------------

def LED_driver_figure():
    '''Figure 4 panels A-C showing LED driver characterisation. '''

    # LED current plot vs DAC voltage plot.
    datafile = os.path.join(data_dir,'LED driver','DAC_value_vs_voltage.csv')
    plt.figure(1, figsize=[9, 3.5]).clf()
    DAC_LED_current_plot(datafile)

    # LED current pulse waveform plot.
    traces_dir = os.path.join(data_dir, 'LED driver','pulse response')
    LED_current_trace = partial(scope_trace, scale_y=1000/4.7, 
                               ylabel='LED current (mA)', ylim=[-5, 120])
    plt.subplot2grid((2, 4), (0, 2), colspan=2)
    LED_current_trace(os.path.join(traces_dir, '100mA 1ms pulse 200us_div'),
                     xlim=[-0.2,1.2])
    plt.subplot2grid((2, 4), (1, 2))
    LED_current_trace(os.path.join(traces_dir, '100mA 1ms pulse rising edge 1us_div'), 
                     xlim=[-1,2], x_units='us')
    plt.subplot2grid((2, 4), (1, 3))
    LED_current_trace(os.path.join(traces_dir, '100mA 1ms pulse falling edge 1us_div'),
                     xlim=[-1,5], x_units='us')
    plt.tight_layout()


def Analog_input_figure():
    '''Figure 4 panels D-E showing analog input characterisation.'''

    # Load data and calcualate means and noise SD for each input and voltage.
    datafolder = os.path.join(data_dir, 'Analog inputs')
    boards = [1,2,3,4]
    input_volts = np.arange(0,2.2,0.2)
    file_names  = ['{:.1f}V.ppd'.format(v) for v in input_volts]
    signal_mean = np.zeros([len(input_volts), 2*len(boards)])
    noise_SD  = np.zeros([len(input_volts), 2*len(boards)])
    for b in boards:
        for i, file_name in enumerate(file_names):
            data = di.import_ppd(
                os.path.join(datafolder, 'board {}'.format(b), file_name))
            data
            signal_mean[i, 2*(1-b)  ] = np.mean(data['analog_1'])
            signal_mean[i, 2*(1-b)+1] = np.mean(data['analog_2'])
            noise_SD[i, 2*(1-b)  ] = np.std( data['analog_1'])
            noise_SD[i, 2*(1-b)+1] = np.std( data['analog_2'])
    mean_signal = np.mean(signal_mean,1)    # Mean meansurement across inputs.
    cross_input_SD = np.std(signal_mean,1)  # Standard deviation across inputs.
    noise_SD  = np.mean(noise_SD,1)         # Mean of noise standard deviation.
    error = mean_signal - input_volts       # Error bewteen cross input mean and input signal.
    means_fit = np.polyfit(input_volts, mean_signal, 1) # Linear fit to cross input mean
    print(means_fit)

    # Evaluate linear fit seperately for each input and computer residuals
    per_input_fits = np.zeros([len(input_volts), 2*len(boards)])
    for i in range(2*len(boards)):
        input_fit = np.polyfit(input_volts, signal_mean[:,i], 1)
        per_input_fits[:,i] = np.poly1d(input_fit)(input_volts)
    per_input_residuals = signal_mean - per_input_fits
    residuals_SDs  = np.std( per_input_residuals,1)
    residuals_mean = np.mean(per_input_residuals,1)
    plt.figure(3, figsize=[6,3]).clf()
    plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    plt.scatter(input_volts, mean_signal, color='b', s=12)
    plt.plot([0,2.2],np.poly1d(means_fit)([0,2.2]), 'b', linewidth=1)
    plt.xlim(0,input_volts[-1]*1.1)
    plt.ylim(0,input_volts[-1]*1.1)
    plt.ylabel('Measured voltage (V)')
    plt.xlabel('Input voltage (V)')
    plt.subplot2grid((3, 2), (0, 1))
    plt.errorbar(input_volts, error*1000, cross_input_SD*1000, color='b', 
                 linestyle='none', marker='o', markersize=4)
    plt.axhline(0,color='k', linestyle=':')
    plt.ylabel('Error')
    plt.subplot2grid((3, 2), (1, 1))
    plt.errorbar(input_volts, residuals_mean*1000,residuals_SDs*1000, color='b', 
                 linestyle='none', marker='o', markersize=4)
    plt.axhline(0,color='k', linestyle=':')
    plt.xlim(0,input_volts[-1]*1.1)
    plt.ylabel('Deviation')
    plt.subplot2grid((3, 2), (2, 1))
    plt.scatter(input_volts, noise_SD*1000, color='b', s=12)
    plt.xlim(0,input_volts[-1]*1.1)
    plt.ylabel('Noise')
    plt.xlabel('Input voltage (V)')
    plt.tight_layout()


def Dopamine_recordings_figure():
    '''Figure 5 showing in vivo recordings in dopamine system.'''
    plt.figure(5, figsize=[10,6]).clf()
    Reward_DA_response(behaviour_file ='P10V_16-2018-08-16-085121.txt', 
                       photometry_file='P10V_16-2018-08-16-085115.ppd',
                       plot_window =[10, 150], plot_row=0)
    Reward_DA_response(behaviour_file='P14-NAc-L-2018-11-29-143413.txt', 
                       photometry_file='P14-NAc-L-2018-11-29-143403.ppd',
                       plot_window=[2280, 2420], ctrl_col='purple', plot_row=1)

# ----------------------------------------------------------------------------------------
# Plotting functions called by figure functions.
# ----------------------------------------------------------------------------------------

def scope_trace(datafolder, fig_no=None, color='b', x_units='ms', ylabel='Volts',
                scale_y=1., xlim=None, ylim=None, trace_index=1, baseline=False,
                y_offset=0.):
    '''Load the set of scope traces in a datafolder, plot the mean and standard deviation.'''
    traces = []
    for filename in os.listdir(datafolder):
        data = np.loadtxt(os.path.join(datafolder,filename),
                          delimiter=',', skiprows=3)
        traces.append(data[:,trace_index])
    traces = np.array(traces)*scale_y
    time = data[:,0]
    trace_mean = np.mean(traces,0)
    if baseline:
        trace_mean = trace_mean - np.mean(trace_mean[time<0])
    if y_offset:
        trace_mean += y_offset
    trace_SD   = np.std(traces,0)
    if fig_no: plt.figure(fig_no).clf()
    plt.fill_between(time, trace_mean-trace_SD, trace_mean+trace_SD, facecolor=color, alpha=0.5)
    plt.plot(time, trace_mean, color=color, linewidth=1)
    if xlim:
        plt.xlim(*xlim)
    else:
        plt.xlim(time[0], time[-1])
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel('Time ({})'.format(x_units))
    plt.ylabel(ylabel)


def DAC_LED_current_plot(datafile):
    '''Load data file with sense resistor voltage measurements as function of 
    DAC values. Plot LED current and standard deviation across drivers as
    function of DAC values.'''
    data = np.loadtxt(datafile, delimiter=',', skiprows=3)
    DAC_values = data[:,0]
    sense_resistor_voltages = data[:,1:]
    LED_currents = sense_resistor_voltages/4.7
    current_mean = np.mean(LED_currents,1)
    current_SD   = np.std(LED_currents,1)
    linear_fit = np.polyfit(current_mean, DAC_values, 1)
    def current_plot(xlim,ylim, ylabel=True):
        plt.scatter(DAC_values, current_mean, color='b', s=12)
        plt.plot(np.poly1d(linear_fit)([0,110]), [0,110], color='b', linewidth=1)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        if ylabel: plt.ylabel('LED current (mA)')
    def current_SD_plot(xlim, ylabel=True):
        plt.scatter(DAC_values, current_SD, color='b', s=12)
        plt.xlim(*xlim)
        plt.xlabel('DAC value')
        if ylabel: plt.ylabel('Current SD (mA)')
    gs = GridSpec(2, 4, height_ratios=[3, 1])
    plt.subplot(gs[0])
    current_plot(xlim=[0,4200],ylim=[0,110])
    plt.subplot(gs[1])
    current_plot(xlim=[0,110],ylim=[0,3], ylabel=False)
    plt.subplot(gs[4])
    current_SD_plot(xlim=[0,4200])
    plt.subplot(gs[5])
    current_SD_plot(xlim=[0,110], ylabel=False)
    print('Δ DAC value per mA LED current: {:.2f}'.format(linear_fit[0]))
    print('DAC value for 0 mA LED current: {:.2f}'.format(linear_fit[1]))


def Time_division_linearity(datafolder):
    '''Plot the mean signal acquired with time division illumination as a function of 
    LED current for in-phase, anti-phase and continuous illumination.'''
    def plot_response(traces_dir, color='b', label=None, marker='o', s=30):
        input_mA = np.arange(10,110,10)
        file_names  = ['{:}mA.ppd'.format(v) for v in input_mA]
        signal_mean = np.zeros(len(file_names))
        for i, file_name in enumerate(file_names):
            data = di.import_ppd(os.path.join(traces_dir, file_name))
            signal_mean[i] = np.mean(data['analog_2'])
        plt.scatter(input_mA, signal_mean, color=color, s=s, marker=marker)
        linear_fit = np.polyfit(input_mA, signal_mean, 1)
        plt.plot([-5,115], np.poly1d(linear_fit)([-5,115]),  color=color, linewidth=1, label=label, marker=marker)
    plot_response(os.path.join(datafolder,'in phase LED'    ), 'b', 'In phase')
    plot_response(os.path.join(datafolder,'out of phase LED'), 'r', 'Out of phase')
    plot_response(os.path.join(datafolder,'continuous LED'  ), 'g', 'Continous', marker='x',s=20)
    plt.xlim(0,110)
    plt.ylim(-0.1,2.5)
    plt.legend()
    plt.ylabel('Photoreceiver signal (V)')
    plt.xlabel('LED current (mA)')
    plt.tight_layout()

def Reward_DA_response(behaviour_file, photometry_file, plot_window, plot_row,
                       peri_reward_window=[-1,4], ctrl_col='r'):
    # Import behavioural and photometry data.
    datafolder  = os.path.join(data_dir, 'dopamine data')
    session =    ci.Session(os.path.join(datafolder, behaviour_file))
    DA_data = di.import_ppd(os.path.join(datafolder, photometry_file), 
                            low_pass=20, high_pass=0.01)
    # Setup syncronisation bewteen photometry and behavioural data.
    sync_signal = DA_data['digital_2'].astype(int) # Signal from digital input 2 which has sync pulses.
    pulse_times_pho = (1 + np.where(np.diff(sync_signal) == 1)[0] # Photometry sync pulse times (ms).
                       * 1000/DA_data['sampling_rate']) 
    pulse_times_pyc = session.times['Rsync'] # pyControl sync pulse times (ms).
    aligner = rs.Rsync_aligner(pulse_times_A=pulse_times_pyc, pulse_times_B=pulse_times_pho)
    # Convert behaviour system reward delivery times into photometry system time reference.
    reward_times = aligner.A_to_B(session.times['reward'])
    reward_times = reward_times[~np.isnan(reward_times)]
    reward_times = reward_times[(reward_times > (plot_window[0]*1000)) &
                                (reward_times < (plot_window[1]*1000))]
    # Extract traces round rewards.
    n_pre_post_samples = (np.array(peri_reward_window)*DA_data['sampling_rate']).astype(int)
    reward_inds = np.argmax(DA_data['time']*1000 > reward_times[:,None], 1)
    GCaMP_traces = np.vstack([DA_data['analog_1_filt']
        [i+n_pre_post_samples[0]:i+n_pre_post_samples[1]] for i in reward_inds])
    TdTom_traces = np.vstack([DA_data['analog_2_filt']
        [i+n_pre_post_samples[0]:i+n_pre_post_samples[1]] for i in reward_inds])
    # Calculate means and standard errors.
    GCaMP_mean = np.mean(GCaMP_traces,0)
    TdTom_mean = np.mean(TdTom_traces,0) - 0.05 # Constant added to offset from GCaMP
    GCaMP_sem  = np.std( GCaMP_traces,0)/np.sqrt(GCaMP_traces.shape[0])
    TdTom_sem  = np.std( TdTom_traces,0)/np.sqrt(GCaMP_traces.shape[0])
    time = np.arange(peri_reward_window[0], peri_reward_window[1], 1/DA_data['sampling_rate'])
    # Plotting
    plt.subplot2grid((2, 4), (plot_row, 0), colspan=3)
    s = plot_window[0]*DA_data['sampling_rate']
    f = plot_window[1]*DA_data['sampling_rate']
    t_0 = DA_data['time'][s]
    plt.plot(DA_data['time'][s:f]-t_0,DA_data['analog_1_filt'][s:f],'g', linewidth=1.2)
    plt.plot(DA_data['time'][s:f]-t_0,DA_data['analog_2_filt'][s:f]-0.1, ctrl_col, linewidth=1.2)
    plt.scatter(DA_data['time'][reward_inds]-t_0, np.ones(len(reward_inds))*
                np.max(DA_data['analog_1_filt'][s:f])*1.1, marker='v', color='k') 
    plt.xlim(DA_data['time'][s]-t_0,DA_data['time'][f]-t_0)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Photoreceiver signal (Volts)')
    plt.subplot2grid((2, 4), (plot_row, 3))
    plt.fill_between(time, GCaMP_mean-GCaMP_sem, GCaMP_mean+GCaMP_sem, facecolor='g', alpha=0.5)
    plt.fill_between(time, TdTom_mean-TdTom_sem, TdTom_mean+TdTom_sem, facecolor=ctrl_col, alpha=0.5)
    plt.plot(time, GCaMP_mean, 'g', linewidth=1.2)
    plt.plot(time, TdTom_mean, ctrl_col, linewidth=1.2)
    plt.xlim(*peri_reward_window)
    plt.xticks([0,2,4])
    plt.xlabel('Time relative to reward delivery (seconds)')
    plt.ylabel('Photoreceiver signal (Volts)')
    plt.tight_layout()


def frequency_division_simulation(f1=211,f2=531,fs=10000,T=100, window='chebwin'):
    assert window in ['gauss', 'blackman', 'boxcar', 'ACG','chebwin'], 'Invalid window'
    t = np.arange(fs*T)/fs     # Time (seconds).
    s1 = np.sin(2*np.pi*f1*t)  # Modulation 1
    s2 = np.sin(2*np.pi*f2*t)  # Modulation 2
    s1sq = s1**2               # Overlap of modulation 1 with itself.
    s1s2 = s1*s2               # Overlap of modulations 1 and 2.
    signal_amplitude = np.mean(s1sq)
    window_durs = np.arange(2,22,2)/1000
    rel_crosstalk_amps = np.zeros(window_durs.shape)
    DC_crosstalk_amps  = np.zeros(window_durs.shape)
    for i, window_dur in enumerate(window_durs):
        win_len = int(fs*window_dur)
        if window == 'gauss':
            w = windows.gaussian(win_len,win_len*0.11)
        elif window == 'blackman':
            w = np.blackman(win_len)
        elif window == 'boxcar':
            w = np.ones(win_len)
        elif window == 'hann':
            w = np.hanning(win_len)
        elif window == 'chebwin':
            w = windows.chebwin(win_len,100)
        elif window == 'ACG':
            w = ACGwindow(win_len,3.5)
        w = w/np.sum(w)
        s1s2_filt = np.convolve(s1s2, w,'valid')
        s1_filt   = np.convolve(s1  , w,'valid')
        rel_crosstalk_amps[i] = np.std(s1s2_filt)/signal_amplitude
        DC_crosstalk_amps[i]  = np.std(s1_filt)  /signal_amplitude
    plt.semilogy(window_durs*1000, rel_crosstalk_amps,'ro-', label='{}Hz - {}Hz'.format(f1,f2))
    #plt.semilogy(window_durs*1000, DC_crosstalk_amps ,'co-', label='   DC - {}Hz'.format(f1))
    plt.xlabel('Window duration (ms)')
    plt.ylabel('Overlap standard deviation')
    plt.legend()


def ACGwindow(N,alf):
    'Approximate confined Gaussian window.'
    G = lambda x: np.exp(-(1/2)*(alf*(x-(N-1)/2)/((N-1)/2))**2)
    p = np.arange(N)
    if alf < 7:
        w= G(p) - G(-1/2)*(G(p+N) + G(p-N))/(G(-1/2+N)+G(1/2-N))
    else:
        w = G(p)
    return w


def FD_TD_noise_comparison(lowpass=False):
    datafolder = os.path.join(data_dir, 'TD FD noise comparison')
    # Load pyPhotometry time division data and calculate coeficient of variation.
    def time_division_CV(filename):
        td_data = di.import_ppd(os.path.join(datafolder,'time division signals', filename))
        signal = td_data['analog_2']
        if lowpass:
            # Calculate coeficcient of variation for signal lowpassed at 20Hz.
            b, a = butter(2, lowpass/(0.5*td_data['sampling_rate']), 'lowpass')
            td_sig_lowpass = filtfilt(b,a,signal)
            td_CV = np.std(td_sig_lowpass)/np.mean(td_sig_lowpass)
        else:
            td_CV = np.std(signal)/np.mean(signal)
        return td_CV
    LED_on_currents = np.arange(20,120,20)
    fs = 130 # Sampling rate.
    td_CVs = [time_division_CV('{}mA_{}Hz.ppd'.format(c,fs)) for c in LED_on_currents]
    duty_cycle = 0.743/(1000/fs) # Fraction of time LED is on for.
    LED_mean_currents_td = np.array(LED_on_currents)*duty_cycle
    # Load sinusoidally modulated data and calculate coefficient of variation.
    def freq_division_CV(filename):
        # Load  data.
        sm_data = loadmat(os.path.join(datafolder,'sinusoidal signals', filename))
        fs = 10000 # Sampling rate
        f1 = 260   # Modulation frequency.
        modulation = sm_data['Y'][:,0]
        fluor_sig  = sm_data['Y'][:,1]
        # Bandpass filter modulation and signal around modulation frequency.
        b, a = butter(2, np.array([0.8*f1, 1.2*f1])/(0.5*fs), 'bandpass')
        filtered_mod = filtfilt(b, a, modulation)
        filtered_sig = filtfilt(b, a, fluor_sig)
        # Find lag which maximises correlation between modulation and signal.
        xcorr = lambda i: np.correlate(np.roll(filtered_mod,i), filtered_sig)
        lags = np.arange(int(-0.5*fs/f1),int(0.5*fs/f1))
        lag = lags[np.argmax([xcorr(s) for s in lags])]
        # Demodulate signal by multiplication with lagged modulation then lowpass filtering.
        mixed_sig = np.roll(filtered_mod,lag)*filtered_sig
        if lowpass: # Lowpass signal
            b, a = butter(2, lowpass/(0.5*fs), 'lowpass')
            demod_signal = filtfilt(b, a, mixed_sig)
        else:
            win_len = np.ceil(fs/130).astype(int)
            window = np.ones(win_len)
            #window = windows.chebwin(win_len,100)
            demod_signal = np.convolve(mixed_sig, window, 'valid')
        # Trim start and end as modulation not present there.
        demod_signal = demod_signal[5000:95000]
        # Evaluate CV.
        fm_CV = np.std(demod_signal)/np.mean(demod_signal)
        return fm_CV
    modulation_mV = np.arange(50,550,50)
    fd_CVs = [freq_division_CV('{}mV_amplitude_260Hz_10s'.format(c)) for c in modulation_mV]
    mA_per_mV = 0.04
    LED_mean_currents_fd = np.array(modulation_mV)*mA_per_mV
    plt.plot(LED_mean_currents_td, td_CVs,'o-', label='Time  division')
    plt.plot(LED_mean_currents_fd, fd_CVs,'o-', label='Freq. division')
    plt.xlabel('Average LED current (mA)')
    plt.ylabel('Signal coef. of variation.')
    plt.xticks(np.arange(0,22,4))
    plt.ylim(ymin=0)
    plt.legend()



def make_modulation(fs=10000, amplitude=250, freq=260, T=10, plot=False, filename=None):
    t = np.arange(fs*T)/fs 
    modulation = amplitude+amplitude*np.sin(2*np.pi*freq*t)
    if plot:
        plt.figure(1).clf()
        plt.plot(t[:1000], modulation[:1000])
    if filename:
        np.savetxt(filename,modulation, fmt='%4.6f')
    else:
        return modulation
