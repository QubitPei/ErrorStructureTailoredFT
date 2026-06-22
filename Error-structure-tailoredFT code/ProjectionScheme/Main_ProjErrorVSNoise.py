import numpy as np
import asym_surfacecode_circuit
import util

from typing import Callable, Set, List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

import pymatching

import matplotlib.pyplot as plt

import sinter    # used for parallel Monte Carlo sampling
import copy
import json


import os
import multiprocessing
from multiprocessing import shared_memory

import time

import pickle
from collections import Counter

from scipy.optimize import curve_fit

import stim


###############################################


### Core variables and file name

# some configuration parameters
global ncpus, z_distance, x_distance, phi, num_shots_max, num_shots_min, Output_Filename

z_distance = 6
x_distance = 7
phi = 1e-3
num_shots_min = 1e7
num_shots_max = 1.5e8   #!!! maximum allowed sample number for a given noise parameter p_dep and a given error configuration
Output_Filename = 'MainProjectionNoise251016.pkl'   #!!! name of the output data file

# try:
#     ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
# except KeyError:
#     ncpus = multiprocessing.cpu_count()
ncpus = 4     #!!! number of cpus to be used


### Define some useful functions: for parallized Proj_Count_naive

def solve_theta(phi, k, delta=1e-14):
    """
    Solve for theta using the dichotomy method given phi and k.
    Improved numerical stability version.
    """
    a = 1e-10  # Avoid exactly 0 for numerical stability
    b = np.pi / 4
    target = np.sin(phi)
    
    def f(theta):
        # Use logarithms for better numerical stability with large exponents
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # For small theta, use approximations to avoid precision loss
        if theta < 1e-6:
            # Taylor expansion for small theta
            numerator = theta**k
            p_ideal = theta**(2*k) + 1.0  # cos(theta)^(2k) ≈ 1 for small theta
        else:
            # Use logarithmic form for better numerical stability
            log_sin_k = k * np.log(sin_theta)
            log_cos_k = k * np.log(cos_theta)
            
            # For the ratio: numerator/denominator = exp(log_numerator - log_denominator)
            # But we need to handle the sum in denominator carefully
            numerator = np.exp(log_sin_k)
            term1 = np.exp(2 * log_sin_k)
            term2 = np.exp(2 * log_cos_k)
            
            # Handle potential overflow/underflow
            if term1 + term2 == 0:
                return -target
            p_ideal = term1 + term2
        
        ratio = numerator / np.sqrt(p_ideal)
        return ratio - target
    
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0:
        # Try to find a better initial bracket
        if abs(fa - target) < abs(fb - target):
            return a
        else:
            return b
    
    # Dichotomy method
    iterations = 0
    max_iter = 100
    while (b - a) >= delta and iterations < max_iter:
        c = (a + b) / 2
        fc = f(c)
        
        if abs(fc) < delta:
            return c
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        iterations += 1
    
    return (a + b) / 2


def probability_sample_binary_strings(k: int, theta: float) -> dict[str, float]:
    """
    Generates the expected combined probability of k-bit binary strings based on theta.

    Returns:
        A Counter with 2^(k-1) entries, each representing a binary string and its complement.
    """
    prob_0 = np.cos(theta)**2
    prob_1 = np.sin(theta)**2

    # Generate all possible k-bit binary strings
    binary_strings = [bin(i)[2:].zfill(k) for i in range(2**k)]

    # Compute the probability of each binary string
    probabilities = np.array([
        np.prod([prob_0 if bit == '0' else prob_1 for bit in b])
        for b in binary_strings
    ])

    # Create a Counter for the expected frequencies
    expected_probabilities = {binary_strings[i]: probabilities[i] for i in range(2**k)}

    # Combine frequencies of binary strings and their complements
    combined_probabilities: dict[str, float] = {}
    seen = set()

    for binary_string, probability in expected_probabilities.items():
        complement = ''.join('1' if bit == '0' else '0' for bit in binary_string)
        key = min(binary_string, complement)
        if key not in seen:
            combined_prob = expected_probabilities[binary_string] + expected_probabilities.get(complement, 0)
            combined_probabilities[key] = combined_prob
            seen.add(key)
            seen.add(complement)

    return combined_probabilities

# subroutine: for a given b string, sample the number of failed shots
def count_logical_failure_Projection(circuit: stim.Circuit, Post_Dec_List: list[int], num_shots: int) -> int:
    '''
    For a given sample number and circuit, and all the post-select detector locations, calculate the number of failed shots
    '''

    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Step 1: first do post-selction, filtering the data points to the `kept_rows`
    selected_columns = detection_events[:, Post_Dec_List]
    abandoned_rows = np.any(selected_columns, axis=1)   # this will provide a boolean mask of the rows
    num_fail = np.sum(abandoned_rows)  # Total abandoned rows

    # kept_rows = ~abandoned_rows
    # filtered_detection_events = detection_events[kept_rows, :]
    # filtered_observable_flips = observable_flips[kept_rows, :]

    # # Step 2: then perform decoding for the filtered rows
    # # Configure a decoder using the circuit.
    # detector_error_model = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    # matcher = pymatching.Matching.from_detector_error_model(detector_error_model)    
    
    # predictions = matcher.decode_batch(filtered_detection_events)

    # mismatches = (predictions != filtered_observable_flips)
    # num_errors = np.sum(np.any(mismatches, axis=1))


    # error_std = binomial_dist_std(num_shots-num_fail, num_errors)
    # return num_errors/(num_shots-num_fail), num_fail/num_shots, error_std
    return num_fail


# main worker function to be parallized: for given p_dep, d, theta and shot number, estimate the passing probability, and the proportion of b when passing
# use direct ancilla-free dispersive coupling ZZ rotation gate
def Proj_Count_naive(rounds: int,  # number of rounds for the syndrome measurement, should >= rounds_post
        rounds_post: int, # number of rounds for the post-selection, should be >= 1
        rounds_prep: int, # number of rounds for state preparation (also post-selection), should be >= 1
        ZZtype: int, # 0: the naive ZZ rotation gate; 1: the dispersive ZZ rotation gate
        # Note that we do not implement the ZZ gate, but only append the corresponding noise channel.
        # distance: int = None,
        x_distance: int = None,
    
    
        z_distance: int = None,  # for the projection scheme, z_distance should be even, x_distance should be odd
        p_dep: float = 0.0,
        Num_shots: int = None,
        phi: float = 0.0,
        ) -> Counter[str]:

    # Step 1: sample bk, determine the histogram for the later stim circuit simulation.  

    if (z_distance % 2) != 0:
        raise ValueError("z distance should be even")
    
    k = int(z_distance/2)
    # theta = phi**(1/k)   # rough value: inaccurate when phi becomes large
    theta = solve_theta(phi, k, delta=1e-8)

    # bk_frequencies = sample_binary_strings(k = k, theta = theta, N = Num_shots)  # sample and store the bk frequency histogram

    # # run naive `sampling' of bk_frequencies to save time
    # bk_frequencies = expected_sample_binary_strings(k = k, theta = theta, N = Num_shots)

    # record the probability of bk_probabilities
    bk_probabilities = probability_sample_binary_strings(k = k, theta = theta)

    # Step 2: for each bk, sample the failure shots from the stim simulation and store it
    bk_fail_shots = Counter()  # the counter register for failure shots 
    
    # for bk, num_bk in sorted(bk_frequencies.items()):
    for bk, prob_bk in sorted(bk_probabilities.items()):
        #### Simplify the calculation: ignore the high weight terms
        count0 = bk.count('0')
        count1 = bk.count('1')
        if count0 > 1 and count1 > 1:  # high weight bit strings
            bk_fail_shots[bk] = int(Num_shots)
            continue  # Skip this string
        ####
        
        b = ''.join(bit * 2 for bit in bk)  # generate a weight-d string to indicate the location of Pauli errors
        b_list = [bit == '1' for bit in b]
        ### b_list = [bit == '0' for bit in b]
        # print(b_list)

        # generate the whole circuit
        circuit, Post_Dec_List = asym_surfacecode_circuit.ProjectionCircuit_Check_naive(b_list, 
                                                            rounds,
                                                            rounds_post,
                                                            rounds_prep, 
                                                            ZZtype, 
                                                            x_distance,
                                                        
                                                            z_distance,
                                                            after_clifford_depolarization = p_dep,
                                                            before_round_data_depolarization = p_dep,
                                                            before_measure_flip_probability = p_dep,
                                                            after_reset_flip_probability = p_dep,
                                                            top_left_2_body_meas_type = 'X',   
                                                            )
        
        # num_bk_fail = count_logical_failure_Projection(circuit, Post_Dec_List, num_bk)
        num_bk_fail = count_logical_failure_Projection(circuit, Post_Dec_List, int(Num_shots))

        bk_fail_shots[bk] = num_bk_fail
    
    # return bk_frequencies, bk_fail_shots
    return bk_probabilities, bk_fail_shots



################################################################

### main function: run the sampling

def main():
    # Initialization
    # ncpus = 3

    # fix the code parameters
    # z_distance = 6
    # x_distance = 7

    # Define the QED rounds
    rounds_post = 3   # should be larger than 2! or else there will be large overhead introduced by the measurement error!
    rounds_prep = 1
    rounds = 4      # z_distance-rounds_prep: enough for the later QEC

    # fix the rotation angle
    # phi = 1e-3

    # sweep the physical noise
    p_dep_min = 1e-4
    p_dep_max = 3e-3
    p_dep_list = np.logspace(np.log10(p_dep_min), np.log10(p_dep_max), num=20)


    ### Run the sampling 
    scanner = p_dep_list


    with multiprocessing.Pool(processes = min(ncpus, len(scanner))) as pool:
        processes = []
        param_list = []  # used to track p_dep and ZZtype

        '''
        # for coupling_strength in coupling_strength_scan:
        for id, p_dep in enumerate(scanner):
            # adaptively adjust num_shots
            # num_shots_min = 3e6
            # num_shots_max = 2e8
            num_shots_min = 3e5
            num_shots_max = 4e5
            num_shots = np.floor(num_shots_min*((p_dep_max/p_dep)**(1.5)))
            num_shots = int( np.min([num_shots, num_shots_max])  )

            # first try naive depolarizing ZZrot
            ZZtype = 0
            param_list.append((p_dep, ZZtype))
            processes.append(pool.apply_async(Proj_Count_naive, args=(rounds, rounds_post, rounds_prep,\
            
                                                                    ZZtype, x_distance, z_distance,\
                                                                        p_dep, num_shots, phi) ) )
            
            
            # then try dispersive ZZrot
            ZZtype = 1
            param_list.append((p_dep, ZZtype))
            processes.append(pool.apply_async(Proj_Count_naive, args=(rounds, rounds_post, rounds_prep,\
                                                                    ZZtype, x_distance, z_distance,\
                                                                        p_dep, num_shots, phi) ) )
        '''
        
        # Submit all tasks first
        for id, p_dep in enumerate(scanner):
            # adaptively adjust num_shots
            # num_shots_min = 3e6
            # num_shots_max = 2e8

            num_shots = np.floor(num_shots_min*((p_dep_max/p_dep)**(1.5)))
            num_shots = int( np.min([num_shots, num_shots_max])  )
            
            for ZZtype in [0, 1]:
                param_list.append((p_dep, ZZtype, num_shots))
                processes.append(
                    pool.apply_async(
                        Proj_Count_naive,   # Proj_Count_naive is the worker function to be parallelized
                        args=(
                            rounds, rounds_post, rounds_prep,
                            ZZtype, x_distance, z_distance,
                            p_dep, num_shots, phi
                        )
                    )
                )

        # Collect results after all submissions
        results_dict = {}
        for (p_dep, ZZtype, num_shots), process in zip(param_list, processes):
            try:
                bk_frequencies, bk_fail_shots = process.get()
                
                if p_dep not in results_dict:
                    results_dict[p_dep] = {}
                    
                results_dict[p_dep][ZZtype] = {
                    'frequencies': bk_frequencies,
                    'fail_shots': bk_fail_shots,
                    'num_shots': num_shots,
                    'trace_distance': None,
                    'fail_probability': None
                }
            except Exception as e:
                print(f"Error processing p_dep={p_dep}, ZZtype={ZZtype}: {str(e)}")
                # Handle missing data explicitly
                if p_dep not in results_dict:
                    results_dict[p_dep] = {}
                results_dict[p_dep][ZZtype] = {
                    'frequencies': {},
                    'fail_shots': Counter(),
                    'num_shots': num_shots,
                    'trace_distance': np.nan,
                    'fail_probability': np.nan
                }
        
        # Post-processing
        for p_dep in results_dict:
            for ZZtype in results_dict[p_dep]:
                data = results_dict[p_dep][ZZtype]
                if data['frequencies']:  # Only process valid data
                    Tr_Dist, P_fail = util.Trace_distace_estimate(
                        phi=phi,
                        z_distance=z_distance,
                        Num_shots=data['num_shots'],
                        bk_probabilities=data['frequencies'],
                        bk_fail_shots=data['fail_shots']
                    )
                    data['trace_distance'] = Tr_Dist
                    data['fail_probability'] = P_fail


    '''
    # Collect Proj_Count_naive results
    results_dict = {}
    for (p_dep, ZZtype), process in zip(param_list, processes):
        try:
            # Get the results from Proj_Count_naive
            bk_frequencies, bk_fail_shots = process.get()
            
            # Initialize nested dictionary structure
            if p_dep not in results_dict:
                results_dict[p_dep] = {}
            
            # Store the original results
            results_dict[p_dep][ZZtype] = {
                'frequencies': bk_frequencies,
                'fail_shots': bk_fail_shots,
                'trace_distance': None,  # Placeholder for post-processing
                'fail_probability': None  # Placeholder for post-processing
            }
            
        except Exception as e:
            print(f"Error for p_dep={p_dep}, ZZtype={ZZtype}: {e}")


    # POST-PROCESSING: Add Trace_distace_estimate results
    for p_dep in results_dict:
        for ZZtype in results_dict[p_dep]:
            # Get the original results
            bk_frequencies = results_dict[p_dep][ZZtype]['frequencies']
            bk_fail_shots = results_dict[p_dep][ZZtype]['fail_shots']
            
            # Calculate additional metrics using Trace_distace_estimate
            # (Assuming Trace_distace_estimate can use the outputs from Proj_Count_naive)
            Tr_Dist, P_fail = util.Trace_distace_estimate(phi = phi, z_distance = z_distance, 
                        Num_shots = num_shots, bk_probabilities = bk_frequencies, 
                        bk_fail_shots = bk_fail_shots)
            
            # Add the new attributes to the results dictionary
            results_dict[p_dep][ZZtype]['trace_distance'] = Tr_Dist
            results_dict[p_dep][ZZtype]['fail_probability'] = P_fail
            
    '''

    '''
    """
    Extract 5 lists from results_dict:
    1. p_dep_list: sorted p_dep values
    2. Fail_Prob_Dep: fail_probability for ZZtype=0
    3. Fail_Prob_Disp: fail_probability for ZZtype=1
    4. Tr_Dist_Dep: trace_distance for ZZtype=0
    5. Tr_Dist_Disp: trace_distance for ZZtype=1

    All lists are sorted by p_dep values.
    """
    # Initialize lists
    p_dep_list = []
    Fail_Prob_Dep = []
    Fail_Prob_Disp = []
    Tr_Dist_Dep = []
    Tr_Dist_Disp = []

    # Get sorted p_dep values
    sorted_p_dep = sorted(results_dict.keys())

    # Extract data for each p_dep
    for p_dep in sorted_p_dep:
        zztype_data = results_dict[p_dep]
        
        # Add to p_dep_list
        p_dep_list.append(p_dep)
        
        # Extract data for ZZtype=0 if available
        if 0 in zztype_data:
            data_0 = zztype_data[0]
            Tr_Dist_Dep.append(data_0['trace_distance'])
            Fail_Prob_Dep.append(data_0['fail_probability'])
        else:
            # Handle missing data (you might want to use NaN or None)
            Tr_Dist_Dep.append(np.nan)
            Fail_Prob_Dep.append(np.nan)
        
        # Extract data for ZZtype=1 if available
        if 1 in zztype_data:
            data_1 = zztype_data[1]
            Tr_Dist_Disp.append(data_1['trace_distance'])
            Fail_Prob_Disp.append(data_1['fail_probability'])
        else:
            # Handle missing data
            Tr_Dist_Disp.append(np.nan)
            Fail_Prob_Disp.append(np.nan)
    '''


    # Extract results for plotting (outside pool context)
    p_dep_list = sorted(results_dict.keys())
    Fail_Prob_Dep = []
    Fail_Prob_Disp = []
    Tr_Dist_Dep = []
    Tr_Dist_Disp = []

    for p_dep in p_dep_list:
        data = results_dict[p_dep]
        # Handle ZZtype=0
        if 0 in data:
            Tr_Dist_Dep.append(data[0]['trace_distance'])
            Fail_Prob_Dep.append(data[0]['fail_probability'])
        else:
            Tr_Dist_Dep.append(np.nan)
            Fail_Prob_Dep.append(np.nan)
        
        # Handle ZZtype=1
        if 1 in data:
            Tr_Dist_Disp.append(data[1]['trace_distance'])
            Fail_Prob_Disp.append(data[1]['fail_probability'])
        else:
            Tr_Dist_Disp.append(np.nan)
            Fail_Prob_Disp.append(np.nan)

    # Cleanup shared memory explicitly
    try:
        for (p_dep, ZZtype, _) in param_list:
            shm_name = f"bk_{p_dep}_{ZZtype}"
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
    except:
        pass


    '''
    # former non-parallized program
    # for id, p_dep in enumerate(scanner):   
    #     # adaptively adjust num_shots
    #     num_shots_min = 3e6
    #     num_shots_max = 2e8
    #     num_shots = np.floor(3e6*((p_dep_max/p_dep)**(1.5)))
    #     num_shots = int( np.min([num_shots, num_shots_max])  )

    #     # # adaptively adjust num_shots
    #     # if p_dep > 7e-3:
    #     #         num_shots = int(2e6)  # Fewer shots for larger error rates
    #     # elif 5e-4 < p_dep <= 7e-3:
    #     #     num_shots = int(1e7)  # Intermediate number of shots
    #     # else:
    #     #     num_shots = int(3e7)  # Maximum number of shots for smallest error rates
        

    #     # first try naive depolarizing ZZ
    #     ZZtype = 0
    #     bk_frequencies, bk_fail_shots = Proj_Count_naive(rounds = rounds,  # number of rounds for the syndrome measurement, should >= rounds_post
    #         rounds_post = rounds_post, # number of rounds for the post-selection, should be >= 1
    #         rounds_prep = rounds_prep, # number of rounds for state preparation (also post-selection), should be >= 1
    #         ZZtype = ZZtype, # 0: the naive ZZ rotation gate; 1: the dispersive ZZ rotation gate
    #         # Note that we do not implement the ZZ gate, but only append the corresponding noise channel.
    #         # distance: int = None,
    #         x_distance = x_distance,
    #         z_distance = z_distance,  # for the projection scheme, z_distance should be even, x_distance should be odd
    #         p_dep = p_dep,
    #         Num_shots = num_shots,
    #         phi = phi,
    #         )
    #     Tr_Dist, P_fail = util.Trace_distace_estimate(phi = phi, z_distance = z_distance, 
    #                            Num_shots = num_shots, bk_probabilities = bk_frequencies, 
    #                            bk_fail_shots = bk_fail_shots)
    #     # save the results
    #     Result_Dep_List[id, :] = [Tr_Dist, P_fail]

    #     # then try dispersive ZZ
    #     ZZtype = 1
    #     bk_frequencies, bk_fail_shots = Proj_Count_naive(rounds = rounds,  # number of rounds for the syndrome measurement, should >= rounds_post
    #         rounds_post = rounds_post, # number of rounds for the post-selection, should be >= 1
    #         rounds_prep = rounds_prep, # number of rounds for state preparation (also post-selection), should be >= 1
    #         ZZtype = ZZtype, # 0: the naive ZZ rotation gate; 1: the dispersive ZZ rotation gate
    #         # Note that we do not implement the ZZ gate, but only append the corresponding noise channel.
    #         # distance: int = None,
    #         x_distance = x_distance,
    #         z_distance = z_distance,  # for the projection scheme, z_distance should be even, x_distance should be odd
    #         p_dep = p_dep,
    #         Num_shots = num_shots,
    #         phi = phi,
    #         )
    #     Tr_Dist, P_fail = util.Trace_distace_estimate(phi = phi, z_distance = z_distance, 
    #                            Num_shots = num_shots, bk_probabilities = bk_frequencies, 
    #                            bk_fail_shots = bk_fail_shots)
    #     # save the results
    #     Result_Disp_List[id, :] = [Tr_Dist, P_fail]
    #     print(f"id = {id}, p_dep = {p_dep}")


    # print("Serial timer: " + str(time.time() - start_time))
    '''

    ######################

    ### Plot the results

    '''
    # Result_Dep_List = np.zeros((len(scanner), 2))   # Trace distance, Failure prob. for naive depelorizing ZZ
    # Result_Disp_List = np.zeros((len(scanner), 2))   # Trace distance, Failure prob. for dispersive ZZ

    # collected_data_2D = np.array(Result_Dep_List)
    # Tr_Dist_Dep = collected_data_2D[:, 0]
    # Fail_Prob_Dep = collected_data_2D[:, 1]

    # collected_data_2D = np.array(Result_Disp_List)
    # Tr_Dist_Disp = collected_data_2D[:, 0]
    # Fail_Prob_Disp = collected_data_2D[:, 1]
    '''

    # Define fitting functions
    def linear_func(x, a):
        return a * np.asarray(x)

    def quadratic_func(x, c):
        return c * np.asarray(x)**2

    # Define color schemes
    colors_failure = {
        'former': '#2ca02c',       # Medium green
        'former_light': '#98df8a', # Light green
        'current': '#d62728',      # Red
        'current_light': '#ff9896',# Light pink
        'hw_line': '#555555'       # Gray for hardware line
    }

    colors_trace = {
        'former': '#4C72B0',       # Soft blue
        'former_light': '#7FA6E1', # Lighter blue
        'current': '#C44E52',      # Soft red
        'current_light': '#E7969C',# Lighter pink
        'hw_line': '#555555'       # Gray for hardware line
    }

    ## Plot the failure probability with green color scheme
    NumFit = 10

    # Perform the linear fit
    popt_fail_disp, _ = curve_fit(linear_func, p_dep_list[:NumFit], Fail_Prob_Disp[:NumFit])
    b = popt_fail_disp[0]

    # Plot failure probabilities
    # plt.figure(figsize=(10, 6))
    plt.plot(p_dep_list, Fail_Prob_Dep, 
            label='former scheme', 
            color=colors_failure['former'], 
            marker='o',
            markersize=6,
            linewidth=2)
    plt.plot(p_dep_list, Fail_Prob_Disp, 
            label='current scheme', 
            color=colors_failure['current'], 
            marker='s',
            markersize=6,
            linewidth=2)

    # Add fitting formula if needed
    # formula_label_disp = f"Fit: ${b:.4f} \\cdot p$"
    # plt.plot(p_dep_list, linear_func(p_dep_list, *popt_fail_disp), 
    #          '--', 
    #          color=colors_failure['current_light'], 
    #          label=formula_label_disp,
    #          linewidth=1.5)

    plt.xlabel('Physical Error Rate (p_dep)', fontsize=12)
    plt.ylabel('Failure Probability', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.6)
    plt.axvline(x=1e-3, color=colors_failure['hw_line'], linestyle=':', label='current hardware')
    plt.legend(fontsize=10, framealpha=0.8)

    # Save the plot
    plt.savefig('Projection_Pfail_250822Test.eps', format='eps', dpi=1000, bbox_inches='tight')
    # plt.show()

    ## Plot trace distances with original color scheme
    plt.figure()
    popt_error_I, _ = curve_fit(quadratic_func, p_dep_list[:NumFit], Tr_Dist_Disp[:NumFit])
    b = popt_error_I[0]/phi

    # Create the fitting formula string
    formula_label_disp = f"Fit: ${b:.4f} \\cdot \\varphi p^2$"

    # plt.figure(figsize=(10, 6))
    plt.plot(p_dep_list, Tr_Dist_Dep, 
            label='former scheme', 
            color=colors_trace['former'], 
            marker='o',
            markersize=6,
            linewidth=2)
    plt.plot(p_dep_list, Tr_Dist_Disp, 
            label='current scheme', 
            color=colors_trace['current'], 
            marker='s',
            markersize=6,
            linewidth=2)

    # Plot the quadratic fit
    plt.plot(p_dep_list, quadratic_func(p_dep_list, *popt_error_I), 
            '--', 
            color=colors_trace['current_light'], 
            label=formula_label_disp,
            linewidth=1.5)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Physical Error Rate (p_dep)', fontsize=12)
    plt.ylabel('Trace distance', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.6)
    plt.axvline(x=1e-3, color=colors_trace['hw_line'], linestyle=':', label='current hardware')
    plt.legend(fontsize=10, framealpha=0.8)

    # Save the plot
    plt.savefig('Projection_ErrorRate_250822Test.eps', format='eps', dpi=1000, bbox_inches='tight')
    # plt.show()


    #############################################

    ### Save the results to a .pkl file

    samplenumber = {"alpha":"1.5", "p_dep_max":"3e-3",  "num_shots_min": "3e6", "num_shots_max": str(num_shots_max)} 
    chi_ZZ = 2*np.pi*5e6    # Dispersive coupling strength
    Omega = 2*np.pi*20e6    # g-f driving strength

    fitting_filename_phi_0p1_naive = 'fitting_parameters_phi_0p1_naive.pkl'
    fitting_filename_phi_0p1_ancilla = 'fitting_parameters_phi_0p1_ancilla.pkl'

    # z_distance = 6
    # x_distance = 7
    # rounds_post = 3
    # rounds_prep = 3
    # rounds = 4      # z_distance-rounds_prep: enough for the later QEC

    # # fix the rotation angle
    # phi = 1e-3

    # # sweep the physical noise
    # p_dep_min = 1e-4
    # p_dep_max = 3e-3
    # p_dep_list = np.logspace(np.log10(p_dep_min), np.log10(p_dep_max), num=20)
    # scanner = p_dep_list

    # register for the results
    Result_Dep_List = np.column_stack((Tr_Dist_Dep, Fail_Prob_Dep))  # Trace distance, Failure prob. for naive depelorizing ZZ
    Result_Disp_List = np.column_stack((Tr_Dist_Disp, Fail_Prob_Disp))   # Trace distance, Failure prob. for dispersive ZZ


    # List of variable names to save
    vars_to_save = {
        'p_dep_list': p_dep_list,
        'z_distance': z_distance,
        'x_distance': x_distance,
        'rounds_post': rounds_post,
        'rounds_prep': rounds_prep,
        'rounds': rounds,
        'phi': phi,
        'Result_Dep_List': Result_Dep_List,  # Trace distance, Failure prob. for naive depelorizing ZZ
        'Result_Disp_List': Result_Disp_List, # Trace distance, Failure prob. for dispersive ZZ
        'chi_ZZ': chi_ZZ,
        'Omega': Omega,
        'samplenumber': samplenumber,
        'fitting_filename_phi_0p1_naive' : fitting_filename_phi_0p1_naive,
        'fitting_filename_phi_0p1_ancilla' : fitting_filename_phi_0p1_ancilla
    }

    # Save to .pkl file
    with open(Output_Filename, 'wb') as f:
        pickle.dump(vars_to_save, f)


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows executable support
    main()
