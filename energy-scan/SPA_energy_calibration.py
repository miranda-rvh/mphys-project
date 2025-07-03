import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import sys
from scipy.optimize import curve_fit
import mplhep
mplhep.style.use("LHCb2")
import progressbar

import pickle

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


#FLAG="PLOT"
#FLAG = 'PARSE'
FLAG = 'PLOT'

print("FLAG: ", FLAG)

#SAMPLE = 'NEIL'
SAMPLE = 'CIS'
#SAMPLE = 'SiC'
print('SAMPLE: ', SAMPLE)

DEPTH_SCAN = False
wheelstepisize = 1
if "FAST" in FLAG: wheelstepisize = 50

#determine the baseline start and end indices for a given sample by looking at the plots

if 'NEIL' in SAMPLE: 
    spa_channel = 3

    picklefilename = "NeilSampleData.pickle"
    PREFIX = 'plots/neil_sample/'
    f = h5py.File('exp_data/Calibration_100KHZz_new_wheel_scan_23-05-23_09-44.mat')
    baseline_start = 10
    baseline_end = 800
    calibration_start = 40
    calibration_end = 120

    depth_scan_data = h5py.File('exp_data/Depth_scan_ND2_only_115angle_10KHz_scan_23-05-23 10-38.mat')
    depth_scan_baseline_start = 10
    depth_scan_baseline_end = 800

if 'CIS' in SAMPLE: 
    spa_channel = 3

    picklefilename = "CISDiodeData.pickle"
    PREFIX = 'plots/cis_diode/'
    f = h5py.File('exp_data/Power_calibration_scan_varfilter_f100kHz_OD2PT2_stepOnerange400_maxpower60uW_spagain20TPA30_06-03-24_10-14.mat')
    baseline_start = 1000
    baseline_end = 3000
    calibration_start = 10
    calibration_end = 260

    depth_scan_data = h5py.File('exp_data/depth_data.mat')
    depth_scan_baseline_start = 10
    depth_scan_baseline_end = 2000


if 'SiC' in SAMPLE:
    spa_channel = 2

    picklefilename = "SiCData.pickle"
    PREFIX = 'plots/SiC/'
    f = h5py.File('exp_data/energy_calibration_SiC_100kHz_720nm.mat')
    baseline_start = 10
    baseline_end = 3000
    calibration_start = 58
    calibration_start = 68
    calibration_end = 300

    '''
    energy_picklefilename = "SiCTCTEnergyScanData.pickle"
    ENERGY_PREFIX = 'plots/SiC/energy_scan_TCT/'
    data = h5py.File('exp_data/energy_scan_SiC_720nm_100kHz_1000avg_-200V_TCT_0to400_4umbelowsurface.mat')
    energy_baseline_start = 10
    energy_baseline_end = 2000
    energy_calibration_start = 80
    energy_calibration_end = 300
    signal_index_start = 0
    signal_index_end = 10000
    '''
    depth_scan_data = h5py.File('exp_data/Zscan_SiC_x3_y3_-160V_05-11-24_21-36.mat')
    #depth_scan_data = h5py.File('exp_data/Zscan_SiC_x3_y3_-180V_05-11-24_21-38.mat')
    #depth_scan_data = h5py.File('exp_data/TCTZscan_SiC_x3_y3_-200V_06-11-24_13-21.mat')
    #depth_scan_data = h5py.File('exp_data/TCTZscan_SiC_x2_y4_-200V_06-11-24_13-28.mat')
    #depth_scan_data = h5py.File('exp_data/depth_voltage_scan_SiC_720nm_100kHz_1000avg_0to-200V_TCT_zenergy-30+40.mat')
    depth_scan_baseline_start = 10
    depth_scan_baseline_end = 2000

def linear(x, m, c):
    return m*x + c

def parse_power_calibration():

    picklefile = open(picklefilename,"wb")

    f1 = f['data']
    print('Shape of data: ', f1.shape)

    all_powers = []
    all_rawpowermeter = []
    all_power_meter_data = []
    all_max_SPAs = []
    all_SPA_errors = []
    all_charge_SPAs = []

    all_wheel_steps = []
    # for i in progressbar.progressbar(range(f["data"].shape[3]), redirect_stdout=True):

    pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()], maxval=f["data"].shape[3])
    pbar.start()

    for i in range(0, f["data"].shape[3], wheelstepisize): #iterate from zero to total number of waveforms

        all_wheel_steps.append(i)

        rawpowermeter = np.average(f['data'][baseline_start:baseline_end,0,0,i,0,0,0,0])
        power = (60e-6/2.)*np.average(f['data'][baseline_start:baseline_end,0,0,i,0,0,0,0]) + 0.15e-6 # Enoch, (MaxPower/MaxVoltage)*PowerMeter reading + bkg

        power_meter_data = np.abs(f['power_meter_data'][0][i])
        
        offset_SPA = np.mean(f['data'][baseline_start:baseline_end,spa_channel,0,i,0,0,0,0])
        max_SPA = np.max(f['data'][:,spa_channel,0,i,0,0,0,0] - offset_SPA)
        SPA_error = np.std(f['data'][baseline_start:baseline_end,spa_channel,0,i,0,0,0,0])
        charge_SPA = max_SPA*0.1e-9/(10*50) #account for gain, amplifier resistance and time in ns

        #energy = power * 0.7214 * max_SPA #why?

        all_powers.append(power)
        all_rawpowermeter.append(rawpowermeter)
        all_power_meter_data.append(power_meter_data)
        all_max_SPAs.append(max_SPA)
        all_SPA_errors.append(SPA_error)
        all_charge_SPAs.append(charge_SPA)

        #if power>0.02 and power < 0.14 and max_SPA>0.7*power+0.0:
        #    powers.append(power)
        #    max_SPAs.append(max_SPA)
        if int(i/10) == i/10:
            pbar.update(i)
    
    pbar.finish()

    wheel_steps = np.array(all_wheel_steps)
    
    powers = np.array(all_powers)
    all_Ep = powers/100000. # divided by the pulse frequency

    power_meter_datas = np.array(all_power_meter_data)
    all_power_meter_Ep = power_meter_datas/100000 # divided by the pulse frequency
    
    max_SPAs = np.array(all_max_SPAs)
    SPA_errors = np.array(all_SPA_errors)
    charge_SPAs = np.array(all_charge_SPAs)

    pickle.dump([wheel_steps, powers, all_Ep, power_meter_datas, all_power_meter_Ep, max_SPAs, SPA_errors, charge_SPAs], picklefile)
    return None

def plot_power_calibration():

    print('Number of waveforms: {:e}'.format(f["data"].shape[3]))
    print('Wheel step size: {:e}'.format(wheelstepisize))

    plt.clf()
    plt.plot(f['data'][:,spa_channel,0,0,0,0,0,0], label = 'SPA (0th waveform)')
    plt.plot(f['data'][:,spa_channel,0,-1,0,0,0,0], label = 'SPA (final waveform)')
    plt.plot(f['data'][baseline_start:baseline_end,spa_channel,0,0,0,0,0,0], label = 'SPA baseline region')
    plt.xlabel('Index')
    plt.ylabel('Voltage [V]')
    plt.legend()
    plt.savefig(PREFIX + 'power_calibration/SPA_waveform.png')
    #plt.show()
    plt.clf()

    picklefile = open(picklefilename,"rb")

    wheel_steps, powers, all_Ep, power_meter_datas, all_power_meter_Ep, max_SPAs, half_SPA_errors, charge_SPAs = pickle.load(picklefile)

    if 'CIS' in SAMPLE or 'SiC' in SAMPLE:
        power = power_meter_datas
        energy = all_power_meter_Ep
    else: 
        power = powers
        energy = all_Ep

    energy_errors = 0.005*energy #*pulse frequency energy
    SPA_errors = np.sqrt(2)*half_SPA_errors

    plt.clf()
    #plt.semilogy(True)
    plt.grid()
    plt.scatter(wheel_steps, power, label = 'Power',  s = 5)
    plt.xlabel(r"Wheel step (arb.)")
    plt.ylabel(r"Power")
    #plt.ylim(1e-7, 1e-6)
    plt.legend()
    plt.savefig(PREFIX + "power_calibration/full_power_wheel_step.png")

    plt.clf()
    plt.semilogy(True)
    plt.grid()
    plt.scatter(power[calibration_start:calibration_end], max_SPAs[calibration_start:calibration_end], label = 'SPA', s = 5)
    plt.xlabel(r"Power")
    plt.ylabel(r"Max SPA [V]")
    plt.legend()
    plt.savefig(PREFIX + "power_calibration/power_vs_maxSPA.png")

    max_SPA_params, pcov = curve_fit(linear, max_SPAs[calibration_start:calibration_end], energy[calibration_start:calibration_end]*1e12, p0=[1,0], sigma = energy_errors[calibration_start:calibration_end]*1e12, absolute_sigma = False)
    max_SPA_errors = np.sqrt(np.diag(pcov))

    plt.clf()
    fig, ax = plt.subplots()
    ax.errorbar(max_SPAs[calibration_start:calibration_end]*1e3, energy[calibration_start:calibration_end]*1e12, xerr=SPA_errors[calibration_start:calibration_end], yerr= energy_errors[calibration_start:calibration_end]*1e12, label="Max. SPA voltage", markersize=3, fmt='o', color = 'b')
    plt.plot(max_SPAs[calibration_start:calibration_end]*1e3, linear(max_SPAs[calibration_start:calibration_end], max_SPA_params[0], max_SPA_params[1]), label="Linear fit", color = 'r')
    ax.set_ylabel(r"Pulse energy [pJ]")
    ax.set_xlabel(r"Maximum SPA voltage [mV]")
    ax.legend(fontsize=20)
    plt.savefig(PREFIX + "power_calibration/Ep_vs_maxSPA_"+SAMPLE+".png")
    plt.clf()

    print("Max SPA calibration parameters: m = {:e} +/- {:e}, c = {:e} +/- {:e}".format(max_SPA_params[0], max_SPA_errors[0], max_SPA_params[1], max_SPA_errors[1]) )

    return None

if 'PARSE' in FLAG:
    parse_power_calibration()

if 'PLOT' in FLAG:
    plot_power_calibration()

def quadratic_fit(x, a, b, c):
    return a*x**2 + b*x + c

def compute_energy_depth_scan():
    picklefile = open(picklefilename,"rb")

    wheel_steps, powers, all_Ep, power_meter_datas, all_power_meter_Ep, max_SPAs, half_SPA_errors, charge_SPAs = pickle.load(picklefile)

    if 'CIS' in SAMPLE or 'SiC' in SAMPLE:
        power = power_meter_datas
        energy = all_power_meter_Ep
    else: 
        power = powers
        energy = all_Ep

    energy_errors = 0.005*energy
    SPA_errors = np.sqrt(2)*half_SPA_errors

    
    print('Number of waveforms: {:e}'.format(depth_scan_data["data"].shape[3]))

    plt.clf()
    plt.plot(depth_scan_data['data'][:,spa_channel,0,0,0,0,0,0], label = 'SPA (0th waveform)')
    plt.plot(depth_scan_data['data'][depth_scan_baseline_start:depth_scan_baseline_end,spa_channel,0,0,0,0,0,0], label = 'SPA baseline (0th waveform)')
    plt.xlabel('Index')
    plt.ylabel('Voltage [V]')
    plt.legend()
    plt.savefig(PREFIX + 'depth_scan/SPA_waveform.png')
    #plt.show()
    plt.clf()

    max_SPA_params, pcov = curve_fit(linear, max_SPAs[calibration_start:calibration_end], energy[calibration_start:calibration_end]*1e12, p0=[1,0], sigma = energy_errors[calibration_start:calibration_end]*1e12, absolute_sigma = False)
    max_SPA_errors = np.sqrt(np.diag(pcov))

    print("Max SPA calibration parameters: m = {:e} +/- {:e}, c = {:e} +/- {:e}".format(max_SPA_params[0], max_SPA_errors[0], max_SPA_params[1], max_SPA_errors[1]) )

    offset_SPA = np.mean(depth_scan_data['data'][depth_scan_baseline_start:depth_scan_baseline_end,spa_channel,0,0,0,0,0,0])
    max_SPA = np.max(depth_scan_data['data'][:,spa_channel,0,0,0,0,0,0] - offset_SPA)
    error_max_SPA = np.std(depth_scan_data['data'][depth_scan_baseline_start:depth_scan_baseline_end,spa_channel,0,0,0,0,0,0])

    calc_energy = max_SPA_params[0]*max_SPA + max_SPA_params[1] #pJ
    energy_error = np.sqrt((max_SPA**2)*max_SPA_errors[0]**2 + (max_SPA_params[0]**2)*error_max_SPA**2 + max_SPA_errors[1]**2)

    print('Max SPA: {:e} +/- {:e}'.format(max_SPA, error_max_SPA))
    print('Pulse energy [pJ]: {:e} +/- {:e}'.format(calc_energy, energy_error))

if DEPTH_SCAN:
    compute_energy_depth_scan()
