import numpy as np

import sympy as sp
from sympy import *

import math as math

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import h5py
import multiprocessing as mp
from numba import jit

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy import integrate
from scipy.signal import find_peaks

from matplotlib import rcParams

import pickle

import progressbar
import mplhep
mplhep.style.use("LHCb2")
#rcParams['font.family'] = 'DejaVu Serif'


R_LIMIT = 25*10**(-6)
Z_LIMIT = 50*10**(-6)

WAVELENGTH = 720*10**(-9)

EXTRAORDINARY_REFRACTIVE_INDEX = np.sqrt(10.52+0.1701/(WAVELENGTH**2-0.0258)+729.2/(WAVELENGTH**2-194.72))
ORDINARY_REFRACTIVE_INDEX = np.sqrt(9.0+0.1364/(WAVELENGTH**2-0.0334)+545/(WAVELENGTH**2-163.69))

PULSE_DURATION = 160*10**(-15) #s
BETA_2 = 2e-14 #m/W


REFRACTIVE_INDEX = 2.6154
H = 6.62607015*10**(-34)

PULSE_ENERGY = 1.009651*10**(-9)

EFFECTIVE_NUMERICAL_APERTURE = 0.42

REF_INDEX_AIR = 1.0003
REF_INDEX_ALUMINIUM = 1.51
 #*VOXEL_POSITION
REFLECTANCE = np.abs((REFRACTIVE_INDEX-REF_INDEX_AIR)/(REFRACTIVE_INDEX+REF_INDEX_AIR))**2
#REFLECTANCE = np.abs((REFRACTIVE_INDEX-REF_INDEX_ALUMINIUM)/(REFRACTIVE_INDEX+REF_INDEX_ALUMINIUM))**2

PREFIX = '/home/miran/mphys/SiC_Diode/plots/SiC/depth_scan_CX/'

CHARGE_DATA_FILE_NAME = "200V_Charge/x3y3.txt"

SIC_DEPTH_SCAN = "/home/miran/mphys/SiC_Diode/exp_data/Zscan_SiC_x3y3_-200V.mat"

AMPLIFIER_RESISTANCE = 50
AMPLIFIER_GAIN = 100 #mV/fC

ENERGY_FILE = "/Users/celyn1/Documents/Sem2MPhys/TCT_BetaFit/energy_file.csv"

@jit(nopython=True)

def direct_charge_density_integrand(z, r, V, beta2, Ep, l):
    NAv = EFFECTIVE_NUMERICAL_APERTURE
    h = H
    freq = (3*10**8)/l
    R = REFLECTANCE
    sigma = 0

    tau = PULSE_DURATION
    l_um = l*10**6
    nr = np.sqrt(9.90 + 0.1364/(l_um**2-0.0334)+545/(l_um**2-163.69))
    
    R = np.abs((nr-REF_INDEX_AIR)/(nr+REF_INDEX_AIR))**2

    DELAY = (2* nr)/(3*10**8)
    deltat = DELAY*V


    ndirect = (4 * beta2 * Ep**2 * NAv**4 * nr**4 * np.pi**1.5 * np.log(2)) / (
    np.exp((4 * NAv**2 * nr**2 * np.pi**2 * r**2) /
           (l**2 * nr**2 + NAv**2 * np.pi**2 * (4 * nr**2 * sigma**2 + NAv**2 * (V - z)**2))) *
    freq * h * tau *
    (l**2 * nr**2 + NAv**2 * np.pi**2 * (4 * nr**2 * sigma**2 + NAv**2 * (V - z)**2)) *
    (l**2 * nr**2 + NAv**4 * np.pi**2 * (V - z)**2) *
    np.sqrt(np.log(4)))
    
    return ndirect*2*np.pi*r

def reflected_charge_density_integrand(z, r, V, beta2, Ep, l):
    NAv = EFFECTIVE_NUMERICAL_APERTURE
    h = H
    freq = (3*10**8)/l
    
    tau = PULSE_DURATION
    l_um = l*10**6
    nr = np.sqrt(9.90 + 0.1364/(l_um**2-0.0334)+545/(l_um**2-163.69))
    R = np.abs((nr-REF_INDEX_AIR)/(nr+REF_INDEX_AIR))**2
    DELAY = (2* nr)/(3*10**8)
    deltat = DELAY*V

    sigma = 0

    nreflected = (4 * beta2 * Ep**2 * NAv**4 * nr**4 * np.pi**1.5 * R**2 * np.log(2)) / (
    np.exp((4 * NAv**2 * nr**2 * np.pi**2 * r**2) /
           (l**2 * nr**2 + NAv**2 * np.pi**2 * (4 * nr**2 * sigma**2 + NAv**2 * (V + z)**2))) *
    freq * h * tau *
    (l**2 * nr**2 + NAv**4 * np.pi**2 * (V + z)**2) *
    (l**2 * nr**2 + NAv**2 * np.pi**2 * (4 * nr**2 * sigma**2 + NAv**2 * (V + z)**2)) *
    np.sqrt(np.log(4)))

    return nreflected*2*np.pi*r

def interference_charge_density_integrand(z, r, V, beta2, Ep, l):
    NAv = EFFECTIVE_NUMERICAL_APERTURE
    h = H
    freq = (3*10**8)/l
    tau = PULSE_DURATION
    l_um = l*10**6
    nr = np.sqrt(9.90 + 0.1364/(l_um**2-0.0334)+545/(l_um**2-163.69))
    DELAY = (2* nr)/(3*10**8)
    deltat = DELAY*V
    R = np.abs((nr-REF_INDEX_AIR)/(nr+REF_INDEX_AIR))**2

    sigma = 0

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



def array_charge_density_normalisation_with_reflection_and_smearing(V, beta, ep, l):
    r_lim = R_LIMIT
    l_um = l*10**6
    nr = np.sqrt(9.90 + 0.1364/(l_um**2-0.0334)+545/(l_um**2-163.69))

    REF_INDEX_CORRECTION = 1/np.sqrt((nr**2-EFFECTIVE_NUMERICAL_APERTURE**2)/(1-EFFECTIVE_NUMERICAL_APERTURE**2))

    z_lim = Z_LIMIT*REF_INDEX_CORRECTION

    V=V*REF_INDEX_CORRECTION

    direct, error = integrate.dblquad(direct_charge_density_integrand, 0, r_lim, 0, z_lim, args=(V, beta, ep, l))
    reflected, error = integrate.dblquad(reflected_charge_density_integrand, 0, r_lim, 0, z_lim, args=(V, beta, ep,l))
    interference, error = integrate.dblquad(interference_charge_density_integrand, 0, r_lim, 0, z_lim, args=(V, beta, ep, l))

    total_normalisation = direct + reflected + interference


    return np.array([total_normalisation])

def parallel_charge_density(V_values, beta, ep, l):
    """Parallel computation of charge densities for multiple V values."""

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(array_charge_density_normalisation_with_reflection_and_smearing, [(V, beta, ep, l) for V in V_values])
    results = np.array(results)  # Convert list to NumPy array

    if results.ndim > 1:
        results = results.flatten()  # Ensure it's 1D

    #print(f"parallel_charge_density output shape: {results.shape}")  # Debugging

    return results

def fit_function(V, beta, ep, l):
    result = parallel_charge_density(V, beta,ep, l)

    return np.array(result)

def wrapper_function(l_test):
    def tempfunc(V, beta,ep, l=l_test):
        return fit_function(V, beta,ep, l_test)
    return tempfunc

def charge_density_normalisation_with_reflection_and_smearing(V, beta, ep, l):
    r_lim = R_LIMIT
    l_um = l*10**6
    nr = np.sqrt(9.90 + 0.1364/(l_um**2-0.0334)+545/(l_um**2-163.69))

    REF_INDEX_CORRECTION = 1/np.sqrt((nr**2-EFFECTIVE_NUMERICAL_APERTURE**2)/(1-EFFECTIVE_NUMERICAL_APERTURE**2))

    z_lim = Z_LIMIT*REF_INDEX_CORRECTION

    V=V*REF_INDEX_CORRECTION
    
    direct, error = integrate.dblquad(direct_charge_density_integrand, 0, r_lim, 0, z_lim, args=(V, beta, ep, l))

    reflected, error = integrate.dblquad(reflected_charge_density_integrand, 0, r_lim, 0, z_lim, args=(V, beta, ep, l))

    interference, error = integrate.dblquad(interference_charge_density_integrand, 0, r_lim, 0, z_lim, args=(V, beta, ep, l))

    total_normalisation = direct + reflected + interference

    return total_normalisation


def data_file_read(detector_thickness, recalculate_exp_charges=False, plot = False, plot_sim = False, debug = False, file = SIC_DEPTH_SCAN, filename = CHARGE_DATA_FILE_NAME, beta = BETA_2, l = WAVELENGTH, ep = PULSE_ENERGY):
    f = h5py.File(file, 'r')
    
    if debug:
        plot = True
        depth_scan_depth0 = f["settings"]['range_of_motion'][0,2] # initial depth in the microscope reference system
        depth_scan_depthf = f["settings"]['range_of_motion'][1,2] # final depth in the microscope reference system
        depth_scan_stepsize = f["settings"]['step_size'][2,0] # step size in the microscope reference system
        depth_scan_nsteps = int((depth_scan_depthf-depth_scan_depth0)/depth_scan_stepsize) # number of steps

        print("experimental settings")
        print("Depth 0: ",depth_scan_depth0)
        print("Depth f: ",depth_scan_depthf)
        print("Step size: ",depth_scan_stepsize)
        print("Number of steps: ",depth_scan_nsteps)
        print(np.shape(f["data"][:,0,0,0,0,0,0,:]))

    #start positions and times for z scan data
    START_POSITION = 0
    END_POSITION = 81
    DATA_START_TIME = 2500
    DATA_END_TIME = 15000
    
    data_width = 20

    
    if debug:
        time_step_0 = f["settings"]['t_axis'][0]
        time_step_1 = f["settings"]['t_axis'][-1]
        print("SiC range: ", time_step_0, time_step_1)
        print("SiC diff = ", time_step_1-time_step_0)
        print("SiC time length", np.shape(f["settings"]['t_axis']))
        print("SiC time step: ", (time_step_1-time_step_0)/np.shape(f["settings"]['t_axis'])[0])
        print("SiC length: ", time_step_1-time_step_0 )

    experiment_time_space = np.linspace(0,5000, 50000)*10e-10
    
    experiment_time_segment = experiment_time_space[DATA_START_TIME:DATA_END_TIME]

    experimental_charges = []
    

    if debug:
        plt.plot(experiment_time_segment,f["data"][:,0,0,0,0,0,0,40][DATA_START_TIME:DATA_END_TIME])
        
        plt.plot(experiment_time_space[DATA_END_TIME+00:DATA_END_TIME+15000],f["data"][:,0,0,0,0,0,0,40][DATA_END_TIME+00:DATA_END_TIME+15000])
        plt.show()

        plt.show()

    signal_deviation = np.std(f["data"][:,0,0,0,0,0,0,0][DATA_END_TIME:DATA_END_TIME+15000])
    total_charge_error = signal_deviation*10**3/12.5*1e-15*np.sqrt(2)
    
    if recalculate_exp_charges:
        print("charge error: ", total_charge_error)
        for i in range(START_POSITION, END_POSITION):
            volt_signal = f["data"][:,0,0,0,0,0,0,i][DATA_START_TIME:DATA_END_TIME] 
            charge = np.min(volt_signal*10**3/12.5*1e-15)
            experimental_charges = np.append(experimental_charges,np.abs(charge))

        np.savetxt(filename, experimental_charges)

    else:
        experimental_charges = np.genfromtxt(filename, dtype = 'float')

    experimental_charges=experimental_charges/1.602e-19
    carrier_deviation = total_charge_error/1.602e-19

    unscaled_positions = np.linspace(START_POSITION, END_POSITION, END_POSITION-START_POSITION)
 
    #plt.plot(unscaled_positions, experimental_charges)
    #plt.show()

    charge_derivative = np.gradient(experimental_charges, range(START_POSITION, END_POSITION))
    
    thickness_indices = np.where(charge_derivative == np.min(charge_derivative))[0] - np.where(charge_derivative == np.max(charge_derivative))[0]
    
    max_derivative_indice = np.where(charge_derivative == np.max(charge_derivative))[0]
    
    scaling = detector_thickness/thickness_indices

    experiment_positions = np.linspace(0, (END_POSITION - START_POSITION)*scaling, END_POSITION-START_POSITION)\
        - max_derivative_indice * scaling 
    
    offset = np.average(experimental_charges[:10])
    experimental_charges -= offset
    
    
    if debug:
        print('INDICES:', thickness_indices)
        print("SCALING:", scaling)

    max_charge=np.max(abs(experimental_charges))
    
    if debug:
        print("MAX:" ,max_charge)
        print("DEVIATION:", carrier_deviation)
        print("DEVIATION/MAX:", carrier_deviation/max_charge)

    adjusted_experiment_data = experimental_charges[START_POSITION:END_POSITION]
    adjusted_positions = experiment_positions[START_POSITION:END_POSITION]
    
    if plot:  
        #plt.plot(experiment_positions, experimental_charges)
        plt.errorbar(experiment_positions, experimental_charges, fmt = 'r', capsize = 2, yerr = carrier_deviation)
        if debug:
            plt.axhline(max_charge)
        #plt.plot(experiment_positions[:40], experimental_charges[:40])
        #plt.plot(experiment_positions, experimental_charges-offset)

        if plot_sim:
            normalisations = np.zeros(len(experiment_positions))
            for i in range(len(experiment_positions)):
                normalisations[i] = charge_density_normalisation_with_reflection_and_smearing(experiment_positions[i], beta, ep)
            print('RATIO:',np.max(normalisations)/ max_charge)
            plt.plot(experiment_positions, normalisations)
        plt.xlabel("Position (m)")
        plt.ylabel("Number of carriers")
        plt.show()

    return experimental_charges, np.array(experiment_positions), carrier_deviation
    
def comparing_experiment(charge_files, thickness = Z_LIMIT, include_errors = True):
    position_array = 1
    for charge_file in charge_files:
        charges, positions, _ = data_file_read(thickness, filename=charge_file)
        print(charge_file[18:21])
        plt.plot(positions, charges, label = charge_file[4:8])
    plt.legend()
    plt.show()


def curve_fitting(position, range_pickle = "pickle_files/l_range_pickle.pickle", refit = True):
    if __name__ == "__main__":
        beta_data = np.genfromtxt(
            "beta_vals.txt", 
            delimiter=",", 
            dtype=[('coordinate', 'U10'), ('beta', 'f8'), ('error', 'f8')], 
            skip_header=1
        )

        energy_data = np.genfromtxt(
            "pulse_energies.txt", 
            delimiter=",", 
            dtype=[('coordinate', 'U10'), ('energy', 'f8'), ('error', 'f8')], 
            skip_header=1
        )

        mat_file_name = "/home/miran/mphys/SiC_Diode/exp_data/Zscan_SiC_{}_-200V.mat".format(position)
        charge_file_name = "200V_Charge/{}.txt".format(position)

        energy_match = energy_data[energy_data['coordinate'] == position]
        beta_match = beta_data[beta_data['coordinate'] == position]

        energy = energy_match['energy'][0]*10**-12
        energy_error = energy_match['error'][0]*10**-12

        beta = beta_match['beta'][0]
        beta_error = beta_match['error'][0]
        
        lower_bounds = [1e-14, energy-energy_error]
        upper_bounds = [1e-12, energy+energy_error]

        experiment_n_carriers, experimental_positions, errors = data_file_read(Z_LIMIT, filename = charge_file_name, file = mat_file_name)

        l_min = 650e-9
        l_max = 800e-9
        l_indices = 20

        l_array = np.linspace(l_min, l_max, l_indices)

        beta_2_values = np.zeros(l_indices)
        beta_2_errors = np.zeros(l_indices)

        if refit:
            picklefile = open(range_pickle,"wb")

            pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()], maxval=len(l_array)+10)
            pbar.start()
            for i in range(len(l_array)):
                
                popt, pcov = curve_fit(wrapper_function(l_array[i]), experimental_positions, experiment_n_carriers, bounds = (lower_bounds, upper_bounds), sigma = errors, absolute_sigma = False)
                pbar.update(i+10)
                perr = np.sqrt(np.diag(pcov))

                beta_2_values[i]= popt[0]
                beta_2_errors[i] = perr[0]
        
            pbar.finish()

            pickle.dump([l_array, beta_2_values, beta_2_errors], picklefile)
        else: 
            picklefile = open(range_pickle,"rb")
            l_array, beta_2_values, beta_2_errors = pickle.load(picklefile)

        return 0

def confidence_plot(position, max_l, min_l, median_l,range_pickle = "pickle_files_2/l_range_pickle.pickle", bound_pickle = "pickle_files_2/full_data_l_pickle.pickle", recalculate = True):
    beta_data = np.genfromtxt(
            "beta_vals.txt", 
            delimiter=",", 
            dtype=[('coordinate', 'U10'), ('beta', 'f8'), ('error', 'f8')], 
            skip_header=1
        )

    energy_data = np.genfromtxt(
        "pulse_energies.txt", 
        delimiter=",", 
        dtype=[('coordinate', 'U10'), ('energy', 'f8'), ('error', 'f8')], 
        skip_header=1
    )

    mat_file_name = "/home/miran/mphys/SiC_Diode/exp_data/Zscan_SiC_{}_-200V.mat".format(position)
    charge_file_name = "200V_Charge/{}.txt".format(position)
    
    energy_match = energy_data[energy_data['coordinate'] == position]
    beta_match = beta_data[beta_data['coordinate'] == position]

    energy = energy_match['energy'][0]*10**-12
    energy_error = energy_match['error'][0]*10**-12

    beta = beta_match['beta'][0]
    beta_error = beta_match['error'][0]
    
    lower_bounds = [1e-14, energy-energy_error]
    upper_bounds = [1e-12, energy+energy_error]

    range_pickle_file = open(range_pickle, "rb") 

    l_array, beta_2_values, beta_2_errors = pickle.load(range_pickle_file)

    charges, positions, errors = data_file_read(Z_LIMIT, filename = charge_file_name, file= mat_file_name)

    if recalculate:
        pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()], maxval=100)
        pbar.start()

        lower_depth_scan_results = np.zeros(len(positions))
        upper_depth_scan_results = np.zeros(len(positions))
        median_depth_scan_results = np.zeros(len(positions))

        abs_lower_depth_scan_results = np.zeros(len(positions))
        abs_upper_depth_scan_results = np.zeros(len(positions))

        upper_popt, upper_pcov = curve_fit(wrapper_function(max_l), positions, charges, bounds = (lower_bounds, upper_bounds), sigma = errors, absolute_sigma = False)
        lower_popt, lower_pcov = curve_fit(wrapper_function(min_l), positions, charges, bounds = (lower_bounds, upper_bounds), sigma = errors, absolute_sigma = False)
        median_popt, median_pcov = curve_fit(wrapper_function(median_l), positions, charges, bounds = (lower_bounds, upper_bounds), sigma = errors, absolute_sigma = False)

        upper_beta = upper_popt[0]
        lower_beta = lower_popt[0]
        median_beta = median_popt[0]

        upper_beta_err = np.sqrt(np.diag(upper_pcov))[0]
        lower_beta_err = np.sqrt(np.diag(lower_pcov))[0]
        median_beta_err = np.sqrt(np.diag(median_pcov))[0]


        results_pickle = open(bound_pickle, "wb")

        for i in range(len(positions)):
            lower_depth_scan_results[i] = charge_density_normalisation_with_reflection_and_smearing(positions[i], lower_popt[0], energy, median_l)
            upper_depth_scan_results[i] = charge_density_normalisation_with_reflection_and_smearing(positions[i], upper_popt[0], energy, median_l)
            median_depth_scan_results[i] = charge_density_normalisation_with_reflection_and_smearing(positions[i], median_popt[0], energy, median_l)

            abs_lower_depth_scan_results[i] = charge_density_normalisation_with_reflection_and_smearing(positions[i], min(beta_2_values), energy, median_l)
            abs_upper_depth_scan_results[i] = charge_density_normalisation_with_reflection_and_smearing(positions[i], max(beta_2_values), energy, median_l)

        pickle.dump([upper_beta, lower_beta, median_beta, upper_beta_err, lower_beta_err, median_beta_err, lower_depth_scan_results, upper_depth_scan_results, median_depth_scan_results, abs_lower_depth_scan_results, abs_upper_depth_scan_results], results_pickle)

    else:
        results_pickle = open(bound_pickle, "rb")

        upper_beta, lower_beta, median_beta, upper_beta_err, lower_beta_err, median_beta_err, lower_depth_scan_results, upper_depth_scan_results, median_depth_scan_results, abs_lower_depth_scan_results, abs_upper_depth_scan_results = pickle.load(results_pickle)
        
    positions = positions.flatten()

    '''
    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(positions*10**6, median_depth_scan_results)
    
    ax.fill_between(positions*10**6, abs_lower_depth_scan_results, abs_upper_depth_scan_results, color = 'grey', alpha = .2)
    ax.fill_between(positions*10**6, lower_depth_scan_results, upper_depth_scan_results, color = 'b', alpha = .2)

    ax.set_xlabel("Position (um)")
    ax.set_ylabel("Number of carriers")
    plt.savefig("plots/depth_scan_lambda_confidence_levels.png")
    plt.clf()
  '''

    y_min = min(beta_2_values - beta_2_errors)
    y_max = max(beta_2_values + beta_2_errors)
    y_range = np.linspace(y_min, y_max, 100)

    l_max = 725.44e-9
    l_min = 714.56e-9

    print('Maximum: ', upper_beta+upper_beta_err)
    print('Minimum: ', lower_beta-lower_beta_err)

    plt.clf()
    fig, ax = plt.subplots()
    ax.errorbar(l_array*1e9, beta_2_values, yerr=beta_2_errors, markersize=5, fmt = 'o', color='blue')
    ax.fill_betweenx(y_range, x1=l_max*1e9, x2=l_min*1e9, color='lightskyblue', alpha=0.5, label=r"Accepted range")
    ax.axhline(y=(upper_beta+upper_beta_err), color = 'r', linestyle='dashed', label=r"Maximum $\beta_2$")
    ax.axhline(y=(lower_beta-lower_beta_err), color = 'k', linestyle='dashed', label=r"Minimum $\beta_2$")
    ax.set_ylabel(r"$\beta_2$ [m/W]")
    ax.set_xlabel(r"Wavelength [nm]")
    legend=plt.legend(loc= 'upper left', fontsize = 20, frameon=True)
    legend.get_frame().set_facecolor('white')  # Set solid white background
    legend.get_frame().set_edgecolor('black')
    plt.savefig(PREFIX+"shaded_wavelength.png")
    plt.clf()
    '''
    fig, ax = plt.subplots()
    ax.errorbar(l_array*1e9, beta_2_values, yerr=beta_2_errors, markersize=5, fmt = 'o', color = 'b', label = 'Wavelength samples')
    ax.fill_betweenx(y_range, x1=(720-5.4432), x2=(720+5.4432), alpha = 0.2, color = 'r')
    
    
    ax.axhline(y=(upper_beta+upper_beta_err), color = 'r', linestyle='dashdot', alpha = 0.5)
    ax.axhline(y=(lower_beta-lower_beta_err), color = 'r', linestyle='dashdot', alpha = 0.5)
    
    
    ax.errorbar(720-5.4432, lower_beta, yerr= lower_beta_err, color = 'red', fmt = 'o', markersize=5, label = 'Accepted range')
    ax.errorbar(720+5.4432, upper_beta, yerr= upper_beta_err, color = 'red', fmt = 'o', markersize=5)

    ax.set_ylabel(r"Beta (m/W)")
    ax.set_xlabel(r"Wavelength (nm)")

    legend=plt.legend(loc= 'lower right', fontsize = 20,frameon=True)
    legend.get_frame().set_facecolor('white')  # Set solid white background
    legend.get_frame().set_edgecolor('black')

    plt.savefig("plots/shaded_lambda.png")
    plt.clf()

    '''
    print(upper_beta+upper_beta_err, lower_beta-lower_beta_err, upper_beta_err)

def main():
    if __name__ == "__main__":
        #curve_fitting("x3y3")
        confidence_plot("x3y3", (720+5.4432)*10**(-9), (720-5.4432)*10**(-9), (720)*10**(-9), recalculate=False)

main()
