import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import sys
from scipy.optimize import curve_fit
from scipy import integrate
import mplhep
mplhep.style.use("LHCb2")
import progressbar
import pickle

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

import multiprocessing as mp
from numba import jit, njit

import warnings
warnings.filterwarnings("ignore")

SAMPLE = 'CIS' #can be SiC or CIS
print('SAMPLE: ', SAMPLE)

#AMP = 'CXL'
AMP = 'TCT'
print('AMP: ', AMP)

ENERGYFLAG = ''
print('ENERGYFLAG: ', ENERGYFLAG)
#controls what happens inside the find energy function

@jit(nopython=True)

def linear(x, m, c):
    return m*x + c

def quadratic_fit(x, a, b, c):
    return a*x**2 + b*x + c

def cubic_fit(x, a):
    return a*x**3

def effectiveNA_z0(wavelength, n, z0):
    return np.sqrt(n*wavelength/(z0*np.pi))

def rayleigh_length(wavelength, n, NA):
    return (n*wavelength)/(np.pi*(NA**2))

def refractive_index_wavelength(wavelength_m): #wang (used)
    wavelength = wavelength_m * 1e6
    return np.sqrt(1 + (0.20075*wavelength**2)/(wavelength**2 + 12.07224) + (5.54861*wavelength**2)/(wavelength**2 - 0.02641) \
        + (35.65066*wavelength**2)/(wavelength**2 - 1268.24708))

def reduced_chi_squared(exp_charges, exp_errors, sim_charges, dof):
    residuals = exp_charges - sim_charges
    chisq = np.sum((residuals/exp_errors)**2)
    dof = len(exp_charges - dof)
 
    reduced_chi = chisq/dof

    return reduced_chi


def refractive_index_model_2(wavelength_m): #naftaly
    wavelength = wavelength_m * 1e6
    return np.sqrt(9.90 + 0.1364/(wavelength**2-0.0334)+545/(wavelength**2-163.69))

def refractive_index_model_3(wavelength_m): #shaffer
    A = 2.5610
    B=0.0340
    wavelength = wavelength_m * 1e3
    return A+(B/wavelength**2)*10**-6

if 'SiC' in SAMPLE:
    #key variables
    WAVELENGTH = 720*10**(-9) #m
    PULSE_DURATION = 160*10**(-15) #s
    REFRACTIVE_INDEX = refractive_index_model_2(WAVELENGTH)
    H = 6.62607015*10**(-34)
    LENS_APERTURE = 0.5
    EFFECTIVE_NA = 0.42
    #Z_0 = rayleigh_length(WAVELENGTH, REFRACTIVE_INDEX, NUMERICAL_APERTURE) #using NA = 0.4

    R_LIM = 25*10**(-6)
    THICKNESS = 50*10**(-6)
    POSITION = 0.5

    REF_INDEX_AIR = 1.0003
    DELAY = (2*REFRACTIVE_INDEX)/(3*10**8) #*VOXEL_POSITION
    REFLECTANCE = np.abs((REFRACTIVE_INDEX-REF_INDEX_AIR)/(REFRACTIVE_INDEX+REF_INDEX_AIR))**2 #check if there is a model for this

    #I will vary pulse energy and fit Beta2
    BETA_2 = 2.1*10**(-14)

if 'CIS' in SAMPLE:
    WAVELENGTH = 1550*10**(-9)
    PULSE_DURATION = 194*10**(-15) #s
    BETA_2 = 0.79*10**(-11) #m/W
    REFRACTIVE_INDEX = 3.4757
    H = 6.62607015*10**(-34)

    LENS_APERTURE = 0.5
    EFFECTIVE_NA = 0.42

    THICKNESS = 162.77*10**(-6)
    R_LIM = 25*10**(-6)
    POSITION = 0.5

    REF_INDEX_AIR = 1.0003
    DELAY = (2*REFRACTIVE_INDEX)/(3*10**8) #*VOXEL_POSITION
    REFLECTANCE = np.abs((REFRACTIVE_INDEX-REF_INDEX_AIR)/(REFRACTIVE_INDEX+REF_INDEX_AIR))**2 #check if there is a model for this

wheelstepisize = 1

if 'CIS' in SAMPLE: 
    spa_channel = 3
    SAVE_FOLDER = "CIS_TCT/"

    picklefilename = SAVE_FOLDER+"CISDiodeData.pickle"
    PREFIX = 'plots/cis_diode/'
    f = h5py.File('exp_data/Power_calibration_scan_varfilter_f100kHz_OD2PT2_stepOnerange400_maxpower60uW_spagain20TPA30_06-03-24_10-14.mat')
    baseline_start = 1000
    baseline_end = 3000
    calibration_start = 10
    calibration_end = 260

    energy_picklefilename = SAVE_FOLDER+"CISEnergyScanData.pickle"
    ENERGY_PREFIX = 'plots/cis_diode/energy_scan/'
    data = h5py.File('exp_data/Energy_scan_varfilter_f100kHz_OD2PT2_stepOnerange400_spagain20TPA30_06-03-24_11-42.mat')
    energy_baseline_start = 10
    energy_baseline_end = 3000
    energy_calibration_start = 5
    energy_calibration_end = 280
    signal_index_start = 0
    signal_index_end = 20000

    blip_index_start = 0
    blip_index_end = 1


if 'SiC' in SAMPLE:
    spa_channel = 2

    picklefilename = "SiCData.pickle"
    PREFIX = 'plots/SiC/'
    f = h5py.File('exp_data/energy_calibration_SiC_100kHz_720nm.mat')
    baseline_start = 10
    baseline_end = 3000
    calibration_start = 70
    calibration_end = 300

    if 'TCT' in AMP:
        SAVE_FOLDER = "SiC_TCT/"

        energy_picklefilename = SAVE_FOLDER+"SiCTCTEnergyScanData.pickle"
        reflection_picklefilename = SAVE_FOLDER+"SiCTCTEnergyScanReflection.pickle"
        envelope_picklefilename = SAVE_FOLDER+"SiCTCTEnvelope.pickle"
        ENERGY_PREFIX = 'plots/SiC/energy_scan_TCT/'
        data = h5py.File('exp_data/energy_scan_SiC_720nm_100kHz_1000avg_-200V_TCT_0to400_4umbelowsurface.mat')
        energy_baseline_start = 10
        energy_baseline_end = 2000
        energy_calibration_start = 75
        energy_calibration_end = 300
        signal_index_start = 0
        signal_index_end = 10000
                
        blip_index_start = 82
        blip_index_end = 90

    if 'CXL' in AMP:
        SAVE_FOLDER = "SiC_CXL/"
    
        energy_picklefilename = SAVE_FOLDER+"SiCXEnergyScanData.pickle" 
        reflection_picklefilename = SAVE_FOLDER+"SiCCXEnergyScanReflection.pickle"
        envelope_picklefilename = SAVE_FOLDER+"SiCCXLEnvelope.pickle"
        ENERGY_PREFIX = 'plots/SiC/energy_scan_CX/'
        data = h5py.File('exp_data/energy_scan_SiC_Cxchargeamp_720nm_100kHz_0to400.mat')
        energy_baseline_start = 10
        energy_baseline_end = 3000
        energy_calibration_start = 85
        energy_calibration_end = 300
        signal_index_start = 0
        signal_index_end = 30000

        blip_index_start = 127
        blip_index_end = 140

def direct_charge_density_integrand(z, r, V, sigma, Ep, beta2, nr, NAv, tau, l):
    h = H
    freq = (3*10**8)/WAVELENGTH
    deltat = DELAY*V
    R = REFLECTANCE
    if l != WAVELENGTH:
        nr = refractive_index_wavelength(l)


    ndirect = (4 * beta2 * Ep**2 * NAv**4 * nr**4 * np.pi**1.5 * np.log(2)) / (
    np.exp((4 * NAv**2 * nr**2 * np.pi**2 * r**2) /
           (l**2 * nr**2 + NAv**2 * np.pi**2 * (4 * nr**2 * sigma**2 + NAv**2 * (V - z)**2))) *
    freq * h * tau *
    (l**2 * nr**2 + NAv**2 * np.pi**2 * (4 * nr**2 * sigma**2 + NAv**2 * (V - z)**2)) *
    (l**2 * nr**2 + NAv**4 * np.pi**2 * (V - z)**2) *
    np.sqrt(np.log(4)))
    
    return ndirect*2*np.pi*r

def reflected_charge_density_integrand(z, r, V, sigma, Ep, beta2, nr, NAv, tau, l):
    h = H
    freq = (3*10**8)/WAVELENGTH
    deltat = DELAY*V
    R = REFLECTANCE
    if l != WAVELENGTH:
        nr = refractive_index_wavelength(l)


    nreflected = (4 * beta2 * Ep**2 * NAv**4 * nr**4 * np.pi**1.5 * R**2 * np.log(2)) / (
    np.exp((4 * NAv**2 * nr**2 * np.pi**2 * r**2) /
           (l**2 * nr**2 + NAv**2 * np.pi**2 * (4 * nr**2 * sigma**2 + NAv**2 * (V + z)**2))) *
    freq * h * tau *
    (l**2 * nr**2 + NAv**4 * np.pi**2 * (V + z)**2) *
    (l**2 * nr**2 + NAv**2 * np.pi**2 * (4 * nr**2 * sigma**2 + NAv**2 * (V + z)**2)) *
    np.sqrt(np.log(4)))

    return nreflected*2*np.pi*r

def interference_charge_density_integrand(z, r, V, sigma, Ep, beta2, nr, NAv, tau, l):
    h = H
    freq = (3*10**8)/WAVELENGTH
    deltat = DELAY*V
    R = REFLECTANCE
    if l != WAVELENGTH:
        nr = refractive_index_wavelength(l)

    ninterference = (
    (2**(2.5 - (2 * deltat**2) / tau**2) * beta2 * np.exp(
        (-4 * NAv**2 * nr**2 * np.pi**2 * r**2) /
        (l**2 * nr**2 + NAv**2 * np.pi**2 * (4 * nr**2 * sigma**2 + NAv**2 * (V - z)**2)) -
        (4 * NAv**2 * nr**2 * np.pi**2 * r**2) /
        (l**2 * nr**2 + NAv**2 * np.pi**2 * (4 * nr**2 * sigma**2 + NAv**2 * (V + z)**2))
    ) / 2) *
    Ep**2 * NAv**4 * nr**4 * np.pi**1.5 * R**2 * np.sqrt(np.log(2))) / (
    freq * h * tau *
    np.sqrt(l**2 * nr**2 + NAv**2 * np.pi**2 * (4 * nr**2 * sigma**2 + NAv**2 * (V - z)**2)) *
    np.sqrt(l**2 * nr**2 + NAv**4 * np.pi**2 * (V - z)**2) *
    np.sqrt(l**2 * nr**2 + NAv**4 * np.pi**2 * (V + z)**2) *
    np.sqrt(l**2 * nr**2 + NAv**2 * np.pi**2 * (4 * nr**2 * sigma**2 + NAv**2 * (V + z)**2)))

    return ninterference*2*np.pi*r

def charge_density_normalisation_with_reflection_and_smearing(Ep, beta2, nr, NAv, tau, l):
    if l != WAVELENGTH:
        nr = refractive_index_wavelength(l)
    ref_index_correction = 1/np.sqrt((nr**2-NAv**2)/(1-NAv**2))
    Z_LIM = THICKNESS*ref_index_correction
    sigma = 0
    r_lim = R_LIM
    z_lim = Z_LIM
    V = POSITION*Z_LIM

    normalisations = np.zeros(len(Ep))

    for i in range(len(Ep)):

        direct, error = integrate.dblquad(direct_charge_density_integrand, 0, r_lim, 0, z_lim, args=(V, sigma, Ep[i], beta2, nr, NAv, tau, l))

        reflected, error = integrate.dblquad(reflected_charge_density_integrand, 0, r_lim, 0, z_lim, args=(V, sigma, Ep[i], beta2, nr, NAv, tau, l))

        interference, error = integrate.dblquad(interference_charge_density_integrand, 0, r_lim, 0, z_lim, args=(V, sigma, Ep[i], beta2, nr, NAv, tau, l))

        total_normalisation = direct + reflected + interference

        normalisations[i] = total_normalisation
        
    return normalisations

def parallel_charge_density_normalisation_with_reflection_and_smearing(Ep, beta2, nr, NAv, tau, l):
    if l != WAVELENGTH:
        nr = refractive_index_wavelength(l)
    
    ref_index_correction = 1/np.sqrt((nr**2-NAv**2)/(1-NAv**2))
    Z_LIM = THICKNESS*ref_index_correction
    sigma = 0
    r_lim = R_LIM
    z_lim = Z_LIM
    V = POSITION*Z_LIM

    direct, error = integrate.dblquad(direct_charge_density_integrand, 0, r_lim, 0, z_lim, args=(V, sigma, Ep, beta2, nr, NAv, tau, l))

    reflected, error = integrate.dblquad(reflected_charge_density_integrand, 0, r_lim, 0, z_lim, args=(V, sigma, Ep, beta2, nr, NAv, tau, l))

    interference, error = integrate.dblquad(interference_charge_density_integrand, 0, r_lim, 0, z_lim, args=(V, sigma, Ep, beta2, nr, NAv, tau, l))

    total_normalisation = direct + reflected + interference
        
    return total_normalisation

def parallel_charge_density(Ep_values, beta2, nr, NAv, tau, l):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(parallel_charge_density_normalisation_with_reflection_and_smearing, [(Ep, beta2, nr, NAv, tau, l) for Ep in Ep_values])
    results = np.array(results)
    if results.ndim > 1:
        results = results.flatten()
 
    return results
 
def fit_function(Ep, beta2, nr, NAv, tau, l):
    #print('Refractive index: ', nr)
    result = parallel_charge_density(Ep, beta2, nr, NAv, tau, l)
    
    return np.array(result)

def wrapper_function(nr_test, NAv_test, tau_test, l_test):
    def tempfunc(Ep, beta2, nr=nr_test, NAv=NAv_test, tau=tau_test, l=l_test):
        return fit_function(Ep, beta2, nr_test, NAv_test, tau_test, l_test)
    return tempfunc

def find_energy():
    picklefile = open(picklefilename,"rb")

    wheel_steps, powers, all_Ep, power_meter_datas, all_power_meter_Ep, max_SPAs, SPA_errors, charge_SPAs = pickle.load(picklefile)
    #loads the power calibration from the pickle file

    if 'CIS' in SAMPLE or 'SiC' in SAMPLE:
        power = power_meter_datas
        energy = all_power_meter_Ep
    else: 
        power = powers
        energy = all_Ep
    
    energy_errors = 0.005*energy
    
    max_SPA_params, pcov = curve_fit(linear, max_SPAs[calibration_start:calibration_end], energy[calibration_start:calibration_end]*1e12, p0=[1,0], sigma = energy_errors[calibration_start:calibration_end]*1e12, absolute_sigma = False)
    max_SPA_errors = np.sqrt(np.diag(pcov))
    #curve fit to find the SPA calibration parameters

   # print("Max SPA calibration parameters: ", max_SPA_params)
   # print("Max SPA errors: ", max_SPA_errors)
    
   # print('Indices: ',data["data"].shape[0])
   # print('Number of waveforms: {:e}'.format(data["data"].shape[3]))

    if 'PLOT_WAVEFORMS' in ENERGYFLAG:
        #check plots make sense

        plt.clf()
        plt.plot(data['data'][:,spa_channel,0,0,0,0,0,0], label = 'SPA (waveform 0)')
        plt.plot(data['data'][:,spa_channel,0,-1,0,0,0,0], label = 'SPA (final waveform)')
        plt.plot(data['data'][energy_baseline_start:energy_baseline_end,spa_channel,0,0,0,0,0,0], label = 'Energy Baseline (waveform 0)', color = 'k')
        plt.xlabel('Index')
        plt.ylabel('Voltage [V]')
        plt.legend()
        plt.savefig(ENERGY_PREFIX + 'SPA_waveforms.png')
        plt.clf()

        plt.clf()
        plt.plot(data['data'][:,0,0,100,0,0,0,0]/np.max(data['data'][:,0,0,100,0,0,0,0]), label = 'Signal 100 (low energy)')
        plt.plot(data['data'][:,0,0,60,0,0,0,0]/np.max(data['data'][:,0,0,60,0,0,0,0]), label = 'Signal 60 (high energy)')
        plt.plot(data['data'][energy_baseline_start:energy_baseline_end,0,0,0,0,0,0,0], label = 'Baseline (signal 0)', color = 'k')
        plt.xlabel('Index')
        plt.ylabel('Voltage [V]')
        plt.legend()
        plt.savefig(ENERGY_PREFIX + 'signals.png')
        plt.clf()

        plt.clf()
        plt.plot(data['data'][:,-1,0,100,0,0,0,0], label = 'Signal 100 (low energy)')
        plt.plot(data['data'][:,-1,0,60,0,0,0,0], label = 'Signal 60 (high energy)')
        plt.xlabel('Index')
        plt.ylabel('Voltage [V]')
        plt.legend()
        plt.savefig(ENERGY_PREFIX + 'reflection_waveforms.png')
        plt.clf()

    if 'PARSE' in ENERGYFLAG:
        #take a data file and extract all numbers needed into a pickle

        picklefile = open(energy_picklefilename,"wb")

        all_sum_charges = []
        all_TCT_charge_errors = []
        all_cxl_charges = []
        all_cxl_signal_errors = []
        all_max_SPA = []
        all_max_SPA_error = []
        all_energy = []
        all_energy_error = []

        #num = [1, 25, 75, 125, 173, 225, 270, 304]

        pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()], maxval=f["data"].shape[3])
        pbar.start()

        for i in range(0, data["data"].shape[3], wheelstepisize):
            #if i not in num: continue
            
            offset_Signal = np.mean(data['data'][energy_baseline_start:energy_baseline_end,0,0,i,0,0,0,0])
            signal = data['data'][signal_index_start:signal_index_end,0,0,i,0,0,0,0]-offset_Signal
            signal_deviation = np.sqrt(2)*np.std(data["data"][energy_baseline_start:energy_baseline_end,0,0,i,0,0,0,0]) #in volts

            sum_charge = np.abs(np.sum(signal)*0.1*1e-9/ (50*100)) #0.1 ns steps/ amplifier resistance * gain factor (40db aka 100)
            signal_deviation_amps = signal_deviation/(50*100)
            TCT_charge_error = signal_deviation_amps*0.1*1e-9 #multiplying the deviation by dt

            gain_factor = 12.5 * 1e-3 * 1e15 #12.5 mV/fC
            cxl_charge = np.abs(np.min(signal/gain_factor))
            cxl_signal_error = signal_deviation

            offset_SPA = np.mean(data['data'][energy_baseline_start:energy_baseline_end,spa_channel,0,i,0,0,0,0])
            max_SPA = np.max(data['data'][:,spa_channel,0,i,0,0,0,0] - offset_SPA)
            error_max_SPA = np.sqrt(2)*np.std(data['data'][energy_baseline_start:energy_baseline_end,spa_channel,0,i,0,0,0,0])
            
            energy = max_SPA_params[0]*max_SPA + max_SPA_params[1] #pJ
            energy_error = np.sqrt((max_SPA**2)*max_SPA_errors[0]**2 + (max_SPA_params[0]**2)*error_max_SPA**2 + max_SPA_errors[1]**2)
        
            all_sum_charges.append(sum_charge)
            all_TCT_charge_errors.append(TCT_charge_error)
            all_cxl_charges.append(cxl_charge)
            all_cxl_signal_errors.append(cxl_signal_error)
            all_energy.append(energy)
            all_energy_error.append(energy_error)
            all_max_SPA.append(max_SPA)
            all_max_SPA_error.append(error_max_SPA)
            
            if int(i/10) == i/10:
                pbar.update(i)
        
        pbar.finish()

        sum_charges = np.array(all_sum_charges)
        TCT_charge_errors = np.array(all_TCT_charge_errors)
        cxl_charges = np.array(all_cxl_charges)
        cxl_signal_errors = np.array(all_cxl_signal_errors)
        energies = np.array(all_energy)
        energy_errors = np.array(all_energy_error)
        max_SPAs = np.array(all_max_SPA)
        error_max_SPAs = np.array(all_max_SPA_error)


        pickle.dump([sum_charges, TCT_charge_errors, cxl_charges, cxl_signal_errors, energies, energy_errors, max_SPAs, error_max_SPAs], picklefile)
    
    energy_picklefile = open(energy_picklefilename,"rb")

    full_sum_charges, full_TCT_charge_errors, full_cxl_charges, full_cxl_signal_errors, full_energies, full_energy_errors, full_max_SPAs, full_error_max_SPAs = pickle.load(energy_picklefile)

    if 'CXL' in AMP:
        full_charges = full_cxl_charges
    else:
        full_charges = full_sum_charges

    #removes energy blips

    blip_linspace = np.linspace(blip_index_start, blip_index_end, (blip_index_end-blip_index_start+1), dtype=int)

    charges = np.delete(full_charges, blip_linspace)[energy_calibration_start:energy_calibration_end]
    energies = np.delete(full_energies, blip_linspace)[energy_calibration_start:energy_calibration_end]
    energy_errors = np.delete(full_energy_errors, blip_linspace)[energy_calibration_start:energy_calibration_end]
    max_SPAs = np.delete(full_max_SPAs, blip_linspace)[energy_calibration_start:energy_calibration_end]
    error_max_SPAs = np.delete(full_error_max_SPAs, blip_linspace)[energy_calibration_start:energy_calibration_end]

    #calulates errors for each amp

    if 'CXL' in AMP:
        signal_errors = np.delete(full_cxl_signal_errors, blip_linspace)[energy_calibration_start:energy_calibration_end]
        gain_factor = 12.5 * 1e-3 * 1e15 #12.5 mV/fC
        gain_error = 0.033333*gain_factor
        signal = charges*gain_factor

        charge_errors = np.sqrt((signal_errors**2)/(gain_factor**2)+(gain_error**2)*(signal**2)/((gain_factor)**4)) #propagation of y errors

    if 'TCT' in AMP:
        signal_errors = np.delete(full_TCT_charge_errors, blip_linspace)[energy_calibration_start:energy_calibration_end]
        charge_errors = np.sqrt(np.abs(energy_calibration_start-energy_calibration_end)*signal_errors**2)

    if 'COMPUTE_REFLECTION' in ENERGYFLAG:
        picklefile = open(reflection_picklefilename,"wb")

        all_reflection = []
        all_reflection_deviation = []

        pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()], maxval=energy_calibration_end)
        pbar.start()

        for i in range(energy_calibration_start, energy_calibration_end, wheelstepisize):
            #if i not in num: continue

            reflected_signal = np.mean(data['data'][:,-1,0,i,0,0,0,0])
            reflected_signal_deviation = np.sqrt(2)*np.std(data["data"][:,-1,0,i,0,0,0,0]) #in volts
        
            all_reflection.append(reflected_signal)
            all_reflection_deviation.append(reflected_signal_deviation)
            
            if int(i/10) == i/10:
                pbar.update(i)
        
        pbar.finish()

        reflected_signals = np.array(all_reflection)
        reflected_signal_errors = np.array(all_reflection_deviation)

        pickle.dump([reflected_signals, reflected_signal_errors], picklefile)

        reflection_picklefile = open(reflection_picklefilename,"rb")

        full_reflected_signals, full_reflected_signal_errors = pickle.load(reflection_picklefile)

        plt.clf()
        fig, ax = plt.subplots()
        ax.errorbar(energies, full_reflected_signals, full_reflected_signal_errors, s=7, color = 'b', label = 'Experimental data')
        ax.set_xlabel(r"E$_{{\rm p}}$ [pJ]")
        ax.set_ylabel(r"Reflected Signal [V]")
        ax.legend()
        plt.savefig(ENERGY_PREFIX + "reflection_vs_energy.png")
        plt.clf()


    if 'PLOT_ENERGY' in ENERGYFLAG:

        indices = np.linspace(0, len(full_max_SPAs), len(full_max_SPAs))

        plt.clf()
        fig, ax = plt.subplots()
        plt.plot(indices, full_max_SPAs)
        plt.fill_betweenx([np.min(full_max_SPAs), np.max(full_max_SPAs)], blip_index_start, blip_index_end, alpha=0.2)
        ax.set_ylabel(r"max SPA [V]")
        ax.set_xlabel(r"Index")
        plt.savefig(ENERGY_PREFIX + "max_SPA_variation.png")
        plt.clf()

        energy_fit_params, *_ = curve_fit(quadratic_fit, energies, charges, p0=[1,0,0])
        print('Energy fit parameters:', energy_fit_params)

        a, b, c = energy_fit_params[0], energy_fit_params[1], energy_fit_params[2]

        '''
        plt.clf()
        fig, ax = plt.subplots()
        ax.scatter(max_SPAs, energies, s=7, color = 'b')
        plt.plot(max_SPAs, linear(max_SPAs, max_SPA_params[0], max_SPA_params[1]), label="Linear fit", color = 'r')
        ax.set_ylabel(r"E$_{{\rm p}}$ [pJ]")
        ax.set_xlabel(r"max SPA [V]")
        ax.legend()
        plt.savefig(ENERGY_PREFIX + "max_SPA_vs_energy.png")
        plt.clf()
        '''

        ordered_energies = np.sort(energies)

        plt.clf()
        fig, ax = plt.subplots()
        ax.scatter(energies, charges, s=7, color = 'b', label = 'Experimental data')
        ax.plot(ordered_energies, quadratic_fit(ordered_energies, a, b, c), label="Quadratic fit", color = 'r')
        ax.set_xlabel(r"E$_{{\rm p}}$ [pJ]")
        ax.set_ylabel(r"Charge [C]")
        ax.legend()
        plt.savefig(ENERGY_PREFIX + "charge_vs_energy.png")
        plt.clf()

    return energies, energy_errors, charges, charge_errors

def fit_charge_parameter(fit_parameter, parse, param_min, param_max, accepted_min, accepted_max, parse_charge):
    energies_pJ, energy_errors_pJ, charges, charge_errors = find_energy()
    energies = energies_pJ*1e-12 #get everything in SI units
    energy_errors = energy_errors_pJ*1e-12
    n_carriers = charges/(1.602e-19)
    n_carriers_errors = charge_errors/(1.602e-19)

    beta_min = 1e-14
    beta_max = 1e-12

    if parse:

        param_indices = 20
        param_array = np.linspace(param_min, param_max, param_indices)

        beta_2_values = np.zeros(param_indices)
        beta_2_errors = np.zeros(param_indices)

        if 'ref_index' in fit_parameter:
            picklefile = open(SAVE_FOLDER+'SiC'+AMP+'_Beta_Nr.pickle',"wb")
        if 'numerical_aperture' in fit_parameter:
            picklefile = open(SAVE_FOLDER+'SiC'+AMP+'_Beta_z0.pickle',"wb")
        if 'pulse_duration' in fit_parameter:
            picklefile = open(SAVE_FOLDER+'SiC'+AMP+'_Beta_tau.pickle',"wb")
        if 'wavelength' in fit_parameter:
            picklefile = open(SAVE_FOLDER+'SiC'+AMP+'_Beta_lamda.pickle',"wb")

        pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()], maxval=len(param_array))
        pbar.start()

        for i in range(len(param_array)):

            if 'ref_index' in fit_parameter:
                print('Refractive index: \n', param_array[i])
                popt, pcov = curve_fit(wrapper_function(param_array[i], EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            if 'numerical_aperture' in fit_parameter:
                popt, pcov = curve_fit(wrapper_function(REFRACTIVE_INDEX, param_array[i], PULSE_DURATION, WAVELENGTH), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            if 'pulse_duration' in fit_parameter:
                popt, pcov = curve_fit(wrapper_function(REFRACTIVE_INDEX, EFFECTIVE_NA, param_array[i], WAVELENGTH), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            if 'wavelength' in fit_parameter:
                popt, pcov = curve_fit(wrapper_function(REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, param_array[i]), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            
            perr = np.sqrt(np.diag(pcov))
            beta_2_values[i] = popt[0]
            beta_2_errors[i] = perr[0]

            pbar.update(i)
            print()

        pickle.dump([param_array, beta_2_values, beta_2_errors], picklefile)
        pbar.finish()

    else:
        if 'ref_index' in fit_parameter:
            picklefile = open(SAVE_FOLDER+'SiC'+AMP+'_Beta_Nr.pickle',"rb")
        if 'numerical_aperture' in fit_parameter:
            picklefile = open(SAVE_FOLDER+'SiC'+AMP+'_Beta_z0.pickle',"rb")
        if 'pulse_duration' in fit_parameter:
            picklefile = open(SAVE_FOLDER+'SiC'+AMP+'_Beta_tau.pickle',"rb")
        if 'wavelength' in fit_parameter:
            picklefile = open(SAVE_FOLDER+'SiC'+AMP+'_Beta_lamda.pickle',"rb")

        param_array, beta_2_values, beta_2_errors =  pickle.load(picklefile)

    if parse_charge:
        if 'ref_index' in fit_parameter:
                charge_picklefile = open(SAVE_FOLDER+SAMPLE+AMP+'_Range_Beta_Nr.pickle',"wb")
        if 'numerical_aperture' in fit_parameter:
                charge_picklefile = open(SAVE_FOLDER+SAMPLE+AMP+'_Range_Beta_z0.pickle',"wb")
        if 'pulse_duration' in fit_parameter:
                charge_picklefile = open(SAVE_FOLDER+SAMPLE+AMP+'_Range_Beta_tau.pickle',"wb")
        if 'wavelength' in fit_parameter:
                charge_picklefile = open(SAVE_FOLDER+SAMPLE+AMP+'_Range_Beta_lamda.pickle',"wb")

        if 'ref_index' in fit_parameter:
            beta2_accepted_max, max_cov = curve_fit(wrapper_function(accepted_min, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            beta2_accepted_min, min_cov = curve_fit(wrapper_function(accepted_max, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            min_error = np.sqrt(np.diag(min_cov))
            max_error = np.sqrt(np.diag(max_cov))
        
        if 'numerical_aperture' in fit_parameter:
            beta2_accepted_max, max_cov = curve_fit(wrapper_function(REFRACTIVE_INDEX, accepted_min, PULSE_DURATION, WAVELENGTH), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            beta2_accepted_min, min_cov = curve_fit(wrapper_function(REFRACTIVE_INDEX, accepted_max, PULSE_DURATION, WAVELENGTH), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            min_error = np.sqrt(np.diag(min_cov))
            max_error = np.sqrt(np.diag(max_cov)) 

        if 'pulse_duration' in fit_parameter:
            beta2_accepted_min, min_cov = curve_fit(wrapper_function(REFRACTIVE_INDEX, EFFECTIVE_NA, accepted_min, WAVELENGTH), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            beta2_accepted_max, max_cov = curve_fit(wrapper_function(REFRACTIVE_INDEX, EFFECTIVE_NA, accepted_max, WAVELENGTH), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            min_error = np.sqrt(np.diag(min_cov))
            max_error = np.sqrt(np.diag(max_cov)) 

        if 'wavelength' in fit_parameter:
            beta2_accepted_min, min_cov = curve_fit(wrapper_function(refractive_index_model_2(accepted_min), EFFECTIVE_NA, PULSE_DURATION, accepted_min), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            beta2_accepted_max, max_cov = curve_fit(wrapper_function(refractive_index_model_2(accepted_max), EFFECTIVE_NA, PULSE_DURATION, accepted_max), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            min_error = np.sqrt(np.diag(min_cov))
            max_error = np.sqrt(np.diag(max_cov)) 

        pickle.dump([beta2_accepted_min, min_error, beta2_accepted_max, max_error], charge_picklefile)
        print('Curvefit completed: ' + fit_parameter)

    else:
        if 'ref_index' in fit_parameter:
            charge_picklefile = open(SAVE_FOLDER+SAMPLE+AMP+'_Range_Beta_Nr.pickle',"rb")
        if 'numerical_aperture' in fit_parameter:
            charge_picklefile = open(SAVE_FOLDER+SAMPLE+AMP+'_Range_Beta_z0.pickle',"rb")
        if 'pulse_duration' in fit_parameter:
            charge_picklefile = open(SAVE_FOLDER+SAMPLE+AMP+'_Range_Beta_tau.pickle',"rb")
        if 'wavelength' in fit_parameter:
            charge_picklefile = open(SAVE_FOLDER+SAMPLE+AMP+'_Range_Beta_lamda.pickle',"rb")
        
        beta2_accepted_min, min_error, beta2_accepted_max, max_error = pickle.load(charge_picklefile)

    y_min = np.min(beta_2_values-beta_2_errors)
    y_max = np.max(beta_2_values+beta_2_errors)
    y_linspace = np.linspace(y_min, y_max, 100)

    print('Maximum: ', beta2_accepted_max+max_error)
    print('Minimum: ', beta2_accepted_min+min_error)

    plt.clf()
    fig, ax = plt.subplots()
    ax.set_ylabel(r"$\beta_2$ [m/W]")
    if 'ref_index' in fit_parameter:
        ax.errorbar(param_array, beta_2_values, yerr=beta_2_errors, markersize=5, fmt = 'o', color = 'b')
        ax.fill_betweenx(y_linspace, accepted_min, accepted_max, color='lightskyblue', alpha=0.5, label=r"Accepted range")
        ax.axhline(beta2_accepted_min-min_error, color = 'k', linestyle = 'dashed', label=r"Minimum $\beta_2$")
        ax.axhline(beta2_accepted_max+max_error, color = 'r', linestyle = 'dashed', label=r"Maximum $\beta_2$")
        ax.set_xlabel(r"Refractive index")
        legend=plt.legend(loc= 'upper left', fontsize = 20, frameon=True)
        legend.get_frame().set_facecolor('white')  # Set solid white background
        legend.get_frame().set_edgecolor('black')
        plt.savefig(ENERGY_PREFIX + "beta2_vs_nr.png")
    if 'pulse_duration' in fit_parameter:  
        ax.errorbar(param_array*1e15, beta_2_values, yerr=beta_2_errors, markersize=5, fmt = 'o', color = 'b')
        ax.fill_betweenx(y_linspace, accepted_min*1e15, accepted_max*1e15, color='lightskyblue', alpha=0.5, label=r"Accepted range")
        ax.axhline(beta2_accepted_min-min_error, color = 'k', linestyle = 'dashed', label=r"Minimum $\beta_2$")
        ax.axhline(beta2_accepted_max+max_error, color = 'r', linestyle = 'dashed', label=r"Maximum $\beta_2$")
        ax.set_xlabel(r"Pulse duration [fs]")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-1000,1000))
        legend=plt.legend(loc= 'upper left', fontsize = 20, frameon=True)
        legend.get_frame().set_facecolor('white')  # Set solid white background
        legend.get_frame().set_edgecolor('black')
        plt.savefig(ENERGY_PREFIX + "beta2_vs_tau.png")
    if 'wavelength' in fit_parameter:
        ax.errorbar(param_array*1e9, beta_2_values, yerr=beta_2_errors, markersize=5, fmt = 'o', color = 'b')
        ax.fill_betweenx(y_linspace, accepted_min*1e9, accepted_max*1e9, color='lightskyblue', alpha=0.5, label=r"Accepted range")
        ax.axhline(beta2_accepted_min-min_error, color = 'k', linestyle = 'dashed', label=r"Minimum $\beta_2$")
        ax.axhline(beta2_accepted_max+max_error, color = 'r', linestyle = 'dashed', label=r"Maximum $\beta_2$")
        ax.set_xlabel(r"Wavelength [nm]")
        legend=plt.legend(loc= 'upper left', fontsize = 20, frameon=True)
        legend.get_frame().set_facecolor('white')  # Set solid white background
        legend.get_frame().set_edgecolor('black')
        plt.savefig(ENERGY_PREFIX + "beta2_vs_lamda.png")
    if 'numerical_aperture' in fit_parameter:
        ax.errorbar(param_array, beta_2_values, yerr=beta_2_errors, markersize=5, fmt = 'o', color = 'b')
        ax.fill_betweenx(y_linspace, accepted_min, accepted_max, color='lightskyblue', alpha=0.5, label=r"Accepted range")
        ax.axhline(beta2_accepted_min-min_error, color = 'k', linestyle = 'dashed', label=r"Minimum $\beta_2$")
        ax.axhline(beta2_accepted_max+max_error, color = 'r', linestyle = 'dashed', label=r"Maximum $\beta_2$")
        ax.set_xlabel(r"Effective numerical aperture")
        legend=plt.legend(loc= 'upper left', fontsize = 20, frameon=True)
        legend.get_frame().set_facecolor('white')  # Set solid white background
        legend.get_frame().set_edgecolor('black')
        plt.savefig(ENERGY_PREFIX + "beta2_vs_na.png")
    plt.clf()

    return param_array, beta_2_values, beta_2_errors, beta2_accepted_min, beta2_accepted_max

def confidence_levels(beta2, calculated_charges, fit_parameter, beta2_accepted_min, beta2_accepted_max, beta_values):
    print('Calcualting and plotting confidence levels')
    energies_pJ, energy_errors_pJ, charges, charge_errors = find_energy()
    energies = energies_pJ*1e-12 #get everything in SI units
    energy_errors = energy_errors_pJ*1e-12
    n_carriers = charges/(1.602e-19)
    n_carriers_errors = charge_errors/(1.602e-19)

    beta_min = 1e-14
    beta_max = 1e-12

    energy_array = np.linspace(np.min(energies), np.max(energies), 500)

    min_calculated_charges = np.array(charge_density_normalisation_with_reflection_and_smearing(energy_array, beta2_accepted_min, REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH))*1.602e-19
    max_calculated_charges = np.array(charge_density_normalisation_with_reflection_and_smearing(energy_array, beta2_accepted_max, REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH))*1.602e-19

    total_min_calculated_charges = np.array(charge_density_normalisation_with_reflection_and_smearing(energy_array, beta_values[0] , REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH))*1.602e-19
    total_max_calculated_charges = np.array(charge_density_normalisation_with_reflection_and_smearing(energy_array, beta_values[-1] , REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH))*1.602e-19

    plt.clf()
    fig, ax = plt.subplots()
    ax.errorbar(energies, charges, xerr=energy_errors, yerr=charge_errors, markersize=5, fmt = 'o', color = 'b', label = 'Experimental data')
    ax.plot(energy_array, calculated_charges, color = 'r', label ='Theoretical charge, Beta2: {:e}'.format(beta2))
    ax.fill_between(energy_array, min_calculated_charges, max_calculated_charges, color='k', alpha=0.4)
    ax.fill_between(energy_array, total_min_calculated_charges, total_max_calculated_charges, color='k', alpha=0.2)
    ax.set_xlabel(r"E$_{{\rm p}}$ [J]")
    ax.set_ylabel(r"Charge [C]")
    ax.legend(fontsize=20)
    if 'ref_index' in fit_parameter:
        plt.savefig(ENERGY_PREFIX + "ref_index_charge_vs_energy.png")
    if 'rayleigh' in fit_parameter:
        plt.savefig(ENERGY_PREFIX + "z0_charge_vs_energy.png")
    if 'pulse_duration' in fit_parameter:
        plt.savefig(ENERGY_PREFIX + "pulse_duration_charge_vs_energy.png")
    if 'wavelength' in fit_parameter:
        plt.savefig(ENERGY_PREFIX + "wavelength_charge_vs_energy.png")
    plt.clf()

    return None

def error_envelope(compute, plot, beta2, error_beta2):
    
    energies_pJ, energy_errors_pJ, charges, charge_errors = find_energy()
    energies = energies_pJ*1e-12 #get everything in SI units
    energy_errors = energy_errors_pJ*1e-12
    n_carriers = charges/(1.602e-19)
    n_carriers_errors = charge_errors/(1.602e-19)

    energy_array = np.linspace(np.min(energies), np.max(energies), 500)

    beta_min = 1e-14
    beta_max = 1e-12

    accepted_tau_min = 141e-15
    accepted_tau_max = 179e-15

    accepted_lamda_min = 714.56e-9
    accepted_lamda_max = 725.44e-9

    accepted_nr_max = refractive_index_model_3(accepted_lamda_min)
    accepted_nr_min = refractive_index_wavelength(accepted_lamda_max)

    accepted_na_min = 0.42
    accepted_na_max = 0.5

    if compute:

        envelope_picklefile = open(envelope_picklefilename,"wb")

        print('Calculating error envelope values')

        beta2_accepted_min, min_cov = curve_fit(wrapper_function(accepted_nr_max, accepted_na_min, accepted_tau_min, accepted_lamda_min), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
        beta2_accepted_max, max_cov = curve_fit(wrapper_function(accepted_nr_min, accepted_na_max, accepted_tau_max, accepted_lamda_max), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))

        min_error = np.sqrt(np.diag(min_cov))
        max_error = np.sqrt(np.diag(max_cov))

        '''
        popt, pcov = curve_fit(fit_function, energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, \
                bounds = ([beta_min, accepted_nr_min, accepted_na_min, accepted_tau_min, accepted_lamda_min], [beta_max, accepted_nr_max, accepted_na_max, accepted_tau_max, accepted_lamda_max]))
        perr = np.sqrt(np.diag(pcov))
        '''

        min_calculated_charges = np.array(charge_density_normalisation_with_reflection_and_smearing(energy_array, beta2_accepted_min, REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH))*1.602e-19
        max_calculated_charges = np.array(charge_density_normalisation_with_reflection_and_smearing(energy_array, beta2_accepted_max, REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH))*1.602e-19
        
        true_charge = np.array(charge_density_normalisation_with_reflection_and_smearing(energy_array, beta2, REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH))*1.602e-19
        true_charge_min = np.array(charge_density_normalisation_with_reflection_and_smearing(energy_array, beta2+error_beta2, REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH))*1.602e-19
        true_charge_max = np.array(charge_density_normalisation_with_reflection_and_smearing(energy_array, beta2-error_beta2, REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH))*1.602e-19

        pickle.dump([np.array(beta2_accepted_min), np.array(beta2_accepted_max), np.array(min_error), np.array(max_error), \
                np.array(min_calculated_charges), np.array(max_calculated_charges), \
                np.array(true_charge), np.array(true_charge_min), np.array(true_charge_max)], envelope_picklefile)

    envelope_picklefile = open(envelope_picklefilename,"rb")
    beta2_accepted_min, beta2_accepted_max, min_error, max_error,\
        min_calculated_charges, max_calculated_charges, true_charge, true_charge_min, true_charge_max = pickle.load(envelope_picklefile)
    
    print('Minimum (parameter range): ', beta2_accepted_min)
    print('Maximum (parameter range): ', beta2_accepted_max)

    if plot:

        print('Plotting error envelope')

        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(energy_array*1e12, true_charge*1e15, color = 'k', linestyle='dashed', lw = 2, zorder=2, label = r'Theoretical fit, $\beta_2$ ={:.2e} m/W'.format(beta2))
        ax.fill_between(energy_array*1e12, min_calculated_charges*1e15, max_calculated_charges*1e15, color='lightskyblue', alpha=0.4, label = 'Uncertainty from parameters')
        ax.fill_between(energy_array*1e12, true_charge_min*1e15, true_charge_max*1e15, color='r',hatch='///', alpha=0.5, label = 'Uncertainty from fit')
        ax.errorbar(energies*1e12, charges*1e15, xerr=energy_errors*1e12, yerr=charge_errors*1e15, markersize=4, fmt = 'o', color = 'b', label = 'Experimental data')
        ax.set_xlabel(r"E$_{{\rm p}}$ [pJ]")
        ax.set_ylabel(r"Charge [fC]")
        ax.legend(fontsize=20)
        if 'TCT' in AMP:
            plt.savefig(ENERGY_PREFIX + "error_envelope_energy_tct.png")
        else:
            plt.savefig(ENERGY_PREFIX + "error_envelope_energy_cxl.png")
        plt.clf()

    return None

def three_pa(beta2, calculated_charges):
    energies_pJ, energy_errors_pJ, charges, charge_errors = find_energy()
    energies = energies_pJ*1e-12 #get everything in SI units
    energy_errors = energy_errors_pJ*1e-12
    n_carriers = charges/(1.602e-19)
    n_carriers_errors = charge_errors/(1.602e-19)
    energy_array = np.linspace(np.min(energies), np.max(energies), 500)

    popt, pcov = curve_fit(cubic_fit, energies, n_carriers, p0 = [1e-8], sigma = n_carriers_errors, absolute_sigma = True)
    a = popt[0]
    perr = np.sqrt(np.diag(pcov))
    a_error = perr[0]

    plt.clf()
    fig, ax = plt.subplots()
    ax.errorbar(energies, charges, xerr=energy_errors, yerr=charge_errors, markersize=5, fmt = 'o', color = 'b', label = 'Experimental data')
    ax.plot(energy_array, cubic_fit(energy_array, a), color = 'r')
    '''
    ax.fill_between(energy_array, cubic_fit(energy_array, a-a_error, b-b_error, c-c_error, d-d_error), \
        cubic_fit(energy_array, a+a_error, b+b_error, c+c_error, d+d_error), color = 'r', alpha=0.5)
    '''
    #ax.plot(energy_array, calculated_charges, color = 'k', label = r'Theortical fit, $\beta_2$ ={:.2e}'.format(beta2))
    ax.set_xlabel(r"E$_{{\rm p}}$ [pJ]")
    ax.set_ylabel(r"Charge [fC]")
    ax.legend(fontsize=20)
    plt.savefig(ENERGY_PREFIX + "three_pa.png")
    plt.clf()

def main(fit_parameter, fit_actions):
    energies_pJ, energy_errors_pJ, charges, charge_errors = find_energy()
    energies = energies_pJ*1e-12 #get everything in SI units
    energy_errors = energy_errors_pJ*1e-12
    n_carriers = charges/(1.602e-19)
    n_carriers_errors = charge_errors/(1.602e-19)
    energy_array = np.linspace(np.min(energies), np.max(energies), 500)

    if 'SiC' in SAMPLE:
        beta_min = 1e-14
        beta_max = 1e-12
    
    if 'CIS' in SAMPLE:
        beta_min = 1e-12
        beta_max = 1e-11

    accepted_lamda_min = 714.56e-9
    accepted_lamda_max = 725.44e-9

    accepted_nr_max = refractive_index_model_3(accepted_lamda_min)
    accepted_nr_min = refractive_index_wavelength(accepted_lamda_max)

    accepted_na_min = 0.42
    accepted_na_max = 0.5

    accepted_tau_min = 141e-15
    accepted_tau_max = 179e-15


    '''
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(energies, energy_errors)
    ax.set_xlabel(r"E$_{{\rm p}}$ [J]")
    ax.set_ylabel(r"Energy error [J]")
    plt.show()
    plt.clf()
    '''

    #This section is the loop. It returns the number of charge carriers for a central voxel for a given pulse energy.
        
    if __name__ == '__main__':
        if 'fit_beta' in fit_actions:
            picklefile = open(SAVE_FOLDER+'SiC'+AMP+'_Beta.pickle',"wb")
            popt, pcov = curve_fit(wrapper_function(REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH), energies, n_carriers, sigma = n_carriers_errors, absolute_sigma = True, bounds = ([beta_min], [beta_max]))
            beta2, error_beta2 = popt[0], np.sqrt(np.diag(pcov))[0]
            print('Beta2: {:e} +/- {:e}'.format(beta2, error_beta2))
            calculated_charges = np.array(charge_density_normalisation_with_reflection_and_smearing(energy_array, beta2, REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH))*1.602e-19
            pickle.dump([beta2, error_beta2, calculated_charges], picklefile)
        else:
            picklefile = open(SAVE_FOLDER+'SiC'+AMP+'_Beta.pickle',"rb")
            beta2, error_beta2, calculated_charges=  pickle.load(picklefile)
            print('Beta2: {:e} +/- {:e}'.format(beta2, error_beta2))

        print('Fit parameter: ', fit_parameter)

        if 'ref_index' in fit_parameter:
            if 'plot' in fit_actions:
                if 'parse' in fit_actions:
                    nr_array, beta_2_values, beta_2_errors, beta2_accepted_min, beta2_accepted_max = fit_charge_parameter(fit_parameter, True, param_min = 2.5, param_max = 2.7, accepted_min=accepted_nr_min, accepted_max=accepted_nr_max)
                else:
                    nr_array, beta_2_values, beta_2_errors, beta2_accepted_min, beta2_accepted_max = fit_charge_parameter(fit_parameter, False, param_min = 2.5, param_max = 2.7, accepted_min=accepted_nr_min, accepted_max=accepted_nr_max, parse_charge=False)

                print(accepted_nr_min, accepted_nr_max)

                #confidence_levels(beta2, calculated_charges, 'ref_index', beta2_accepted_min, beta2_accepted_max, beta_2_values)

        if 'numerical_aperture' in fit_parameter:
            if 'plot' in fit_actions:
                if 'parse' in fit_actions:
                    na_array, beta_2_values, beta_2_errors, beta2_accepted_min, beta2_accepted_max = fit_charge_parameter(fit_parameter, True, param_min = 0.3, param_max = 0.55, accepted_min=accepted_na_min, accepted_max=accepted_na_max)
                else:
                    na_array, beta_2_values, beta_2_errors, beta2_accepted_min, beta2_accepted_max = fit_charge_parameter(fit_parameter, False, param_min = 0.3, param_max = 0.55, accepted_min=accepted_na_min, accepted_max=accepted_na_max, parse_charge=False)

                #confidence_levels(beta2, calculated_charges, 'numerical_aperture', beta2_accepted_min, beta2_accepted_max, beta_2_values)

        if 'pulse_duration' in fit_parameter:
            if 'plot' in fit_actions:
                if 'parse' in fit_actions:
                    tau_array, beta_2_values, beta_2_errors, beta2_accepted_min, beta2_accepted_max = fit_charge_parameter(fit_parameter, True, param_min = 130e-15, param_max = 190e-15, accepted_min=accepted_tau_min, accepted_max=accepted_tau_max)
                else:
                    tau_array, beta_2_values, beta_2_errors, beta2_accepted_min, beta2_accepted_max = fit_charge_parameter(fit_parameter, False, param_min = 130e-15, param_max = 190e-15, accepted_min=accepted_tau_min, accepted_max=accepted_tau_max, parse_charge=False)

                #confidence_levels(beta2, calculated_charges, 'pulse_duration', beta2_accepted_min, beta2_accepted_max, beta_2_values)

        if 'wavelength' in fit_parameter:
            if 'plot' in fit_actions:
                if 'parse' in fit_actions:
                    lamda_array, beta_2_values, beta_2_errors, beta2_accepted_min, beta2_accepted_max = fit_charge_parameter(fit_parameter, True, param_min = 650e-9, param_max = 800e-9, accepted_min=accepted_lamda_min, accepted_max=accepted_lamda_max)
                else:
                    lamda_array, beta_2_values, beta_2_errors, beta2_accepted_min, beta2_accepted_max = fit_charge_parameter(fit_parameter, False, param_min = 650e-9, param_max = 800e-9, accepted_min=accepted_lamda_min, accepted_max=accepted_lamda_max, parse_charge=False)

                #confidence_levels(beta2, calculated_charges, 'wavelength', beta2_accepted_min, beta2_accepted_max, beta_2_values)

        if 'third_order' in fit_parameter:
            three_pa(beta2, calculated_charges)

        if 'envelope' in fit_parameter:
            if 'plot' in fit_actions:
                if 'parse' in fit_actions:
                    error_envelope(compute=True, plot=True, beta2=beta2, error_beta2=error_beta2)
                else:
                    error_envelope(compute=False, plot=True, beta2=beta2, error_beta2=error_beta2)
            
        else:
            energy_picklefile = open(energy_picklefilename,"rb")
            full_sum_charges, full_TCT_charge_errors, full_cxl_charges, full_cxl_signal_errors, full_energies, full_energy_errors, full_max_SPAs, full_error_max_SPAs = pickle.load(energy_picklefile)

            if 'CXL' in AMP:
                full_charges = full_cxl_charges
            else:
                full_charges = full_sum_charges
    
            uncleaned_charges = full_charges[energy_calibration_start:energy_calibration_end]
            uncleaned_energies = full_energies[energy_calibration_start:energy_calibration_end]
            uncleaned_energy_errors = full_energy_errors[energy_calibration_start:energy_calibration_end]

            if 'CXL' in AMP:
                signal_errors = full_cxl_signal_errors[energy_calibration_start:energy_calibration_end]
                gain_factor = 12.5 * 1e-3 * 1e15 #12.5 mV/fC
                gain_error = 0.033333*gain_factor
                signal = full_charges[energy_calibration_start:energy_calibration_end]*gain_factor
                uncleaned_charge_errors = np.sqrt((signal_errors**2)/(gain_factor**2)+(gain_error**2)*(signal**2)/((gain_factor)**4)) #propagation of y errors

            if 'TCT' in AMP:
                signal_errors = full_TCT_charge_errors[energy_calibration_start:energy_calibration_end]
                uncleaned_charge_errors = np.sqrt(np.abs(energy_calibration_start-energy_calibration_end)*signal_errors**2)            

            plt.clf()
            fig, ax = plt.subplots()
            #ax.errorbar(uncleaned_energies, uncleaned_charges*1e15, xerr=uncleaned_energy_errors, yerr=uncleaned_charge_errors*1e15, markersize=5, fmt = 'o', color = 'lightblue', label = 'Uncleaned experimental data')
            ax.errorbar(energies*1e12, charges*1e15, xerr=energy_errors*1e12, yerr=charge_errors*1e15, markersize=5, fmt = 'o', color = 'b', label = 'Experimental data')
            #ax.plot(energy_array*1e12, calculated_charges*1e15, color = 'r', label = r'Theoretical fit')
            ax.plot(energy_array*1e12, calculated_charges*1e15, color = 'r', label = r'Theoretical fit, $\beta_2$ ={:.2e} m/W'.format(beta2))
            ax.set_xlabel(r"E$_{{\rm p}}$ [pJ]")
            ax.set_ylabel(r"Charge [fC]")
            ax.legend(fontsize=25)
            if 'TCT' in AMP:
                plt.savefig(ENERGY_PREFIX + "initial_energy_scan_tct.png")
            else:
                plt.savefig(ENERGY_PREFIX + "initial_energy_scan_cxl.png")
            #plt.show()
            plt.clf()
            
            energy_picklefile = open(energy_picklefilename,"rb")
            full_sum_charges, full_TCT_charge_errors, full_cxl_charges, full_cxl_signal_errors, full_energies, full_energy_errors, full_max_SPAs, full_error_max_SPAs = pickle.load(energy_picklefile)

            if 'CXL' in AMP:
                full_charges = full_cxl_charges
                print('Energy: ',full_energies[74])

            else:
                full_charges = full_sum_charges
                print('Energy: ',full_energies[74])


            plt.clf()
            fig, ax = plt.subplots()
            if 'TCT' in AMP:
                ax.plot(data['data'][:,0,0,74,0,0,0,0], color='b', label = 'Signal waveform')
                ax.set_xlim(2000, 7000)
            else:
                ax.plot(-data['data'][:,0,0,74,0,0,0,0], color='b', label = 'Signal waveform')
                ax.set_xlim(0, 25000)
            
            ax.set_xlabel('Time [arb.]')
            ax.set_ylabel('Voltage [V]')
            ax.set_xticklabels([])
            plt.legend(fontsize=25, loc='upper right')
            plt.savefig(ENERGY_PREFIX + 'signals.png')
            plt.clf()

            #sim_charges = np.array(charge_density_normalisation_with_reflection_and_smearing(energies, beta2, REFRACTIVE_INDEX, EFFECTIVE_NA, PULSE_DURATION, WAVELENGTH))*1.602e-19
            #print(reduced_chi_squared(charges, charge_errors, sim_charges, dof=3))

    #print('Ration of sim to experiment at {:e}J and Beta2={:e}: {:e}'.format(energies[11], BETA_2, (calculated_charges[11]/charges[11])))

    return None

main(fit_parameter= '', fit_actions = '')

#main(fit_parameter= 'pulse_duration', fit_actions = 'plot')
#main(fit_parameter= 'ref_index', fit_actions = 'plot')
#main(fit_parameter= 'wavelength', fit_actions = 'plot')
#main(fit_parameter= 'numerical_aperture', fit_actions = 'plot')

'''
fit_parameter options:
    'ref_index'
    'numerical_aperture'
    'wavelength'
    'pulse_duration'
    'envelope'
    otherwise it will make an energy-charge graph with a line of fitted beta2

fit_actions options: 
    fit_beta : use if no pickle file for initial beta fit
    plot : will make plots for a given fit parameter
    parse : will create a new pickle file for a given fit parameter
    otherwise leave blank
'''