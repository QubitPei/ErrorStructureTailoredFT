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
global ncpus, p_dep, phi, num_shots, k_List, Output_Filename

p_dep = 1e-3
phi = 1e-3
num_shots = 3e8   #!!! sample number for a given k value
k_List = [3,4,5,6]    #!!! the z-distance of the surface code to be considered; k
Output_Filename = 'MainProjScale251016.pkl'   #!!! name of the output data file

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



# Sample the circuit. (used for `count_logical_failure_Projection_parallel`)
def sample_from_circuit(shots: int, circuit: stim.Circuit):
    return circuit.compile_detector_sampler().sample(shots, separate_observables=True)

def count_logical_failure_Projection_parallel(circuit: stim.Circuit, Post_Dec_List: list[int], num_shots: int, ncpus: int) -> int:
    '''
    For a given sample number and circuit, and all the post-select detector locations, calculate the number of failed shots
    
    Use multiprocessing for parallelization.
    '''

    # use `starmap` to do simple parallized sampling, then combine the counting
    with multiprocessing.Pool(ncpus) as pool:
        results = pool.starmap(
            sample_from_circuit,
            [(num_shots // ncpus, circuit)] * ncpus
        )
    
        # Split into detection_events and observable_flips lists
        detections_list, observables_list = zip(*results)  # Unzips into two lists of arrays
    
        # Concatenate each component
        detection_events = np.concatenate(detections_list)    # Shape: (total_shots, num_detectors)
        observable_flips = np.concatenate(observables_list)  # Shape: (total_shots, num_observables)
    
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
        num_bk_fail = count_logical_failure_Projection_parallel(circuit, Post_Dec_List, int(Num_shots), ncpus)

        bk_fail_shots[bk] = num_bk_fail
    
    # return bk_frequencies, bk_fail_shots
    return bk_probabilities, bk_fail_shots



################################################################

### main function: run the sampling

def main():
    # Initialization
    # ncpus = 3

    global ncpus, p_dep, phi, num_shots, k_List
    # k_List = [3,4,5]

    # fix the physical error rate
    # p_dep = 1e-3

    # fix the sample number
    # num_shots = 1e7

    # fix the rotation angle
    # phi = 1e-3

    # register for the results
    # Result_Dep_List = np.zeros((len(k_List), 2))   # Trace distance, Failure prob. for naive depelorizing ZZ
    Result_Disp_List = np.zeros((len(k_List), 2))   # Trace distance, Failure prob. for dispersive ZZ

    start_time = time.time()


    for id, k in enumerate(k_List):
        
        # fix the code parameters
        z_distance = 2*k
        x_distance = 2*k + 1
        rounds_post = 3
        rounds_prep = 1
        rounds = 4      # z_distance-rounds_prep: enough for the later QEC

        # try dispersive ZZ: consider ancilla-free dispersive coupling
        ZZtype = 1
        bk_frequencies, bk_fail_shots = Proj_Count_naive(rounds = rounds,  # number of rounds for the syndrome measurement, should >= rounds_post
            rounds_post = rounds_post, # number of rounds for the post-selection, should be >= 1
            rounds_prep = rounds_prep, # number of rounds for state preparation (also post-selection), should be >= 1
            ZZtype = ZZtype, # 0: the naive ZZ rotation gate; 1: the dispersive ZZ rotation gate
            # Note that we do not implement the ZZ gate, but only append the corresponding noise channel.
            # distance: int = None,
            x_distance = x_distance,
            z_distance = z_distance,  # for the projection scheme, z_distance should be even, x_distance should be odd
            p_dep = p_dep,
            Num_shots = num_shots,
            phi = phi,
            )
        Tr_Dist, P_fail = util.Trace_distace_estimate(phi = phi, z_distance = z_distance, 
                            Num_shots = num_shots, bk_probabilities = bk_frequencies, 
                            bk_fail_shots = bk_fail_shots)
        # save the results
        Result_Disp_List[id, :] = [Tr_Dist, P_fail]
        print(f"id = {id}, k = {k}")

        print("Serial timer: " + str(time.time() - start_time))


    ######################

    ### Plot the results

    # Define vectorized fitting functions
    def linear_func(x, a):
        return a * np.asarray(x)  # Ensure x is treated as array

    def quadratic_func(x, c):
        return c * np.asarray(x)**2  # Ensure x is treated as array

    # Assuming Result_Disp_List is your data
    collected_data_2D = np.array(Result_Disp_List)
    Tr_Dist_Disp = collected_data_2D[:, 0]  # Fixed typo from Tr_Dist_Disp
    Fail_Prob_Disp = collected_data_2D[:, 1]

    # Fit the data - ensure k_List is numpy array
    k_List = np.asarray(k_List)
    popt_error_I, _ = curve_fit(linear_func, k_List, Tr_Dist_Disp)

    # Calculate b - ensure phi and p_dep are scalars
    phi = float(phi)  # Convert to float if not already
    p_dep = float(p_dep)
    b = popt_error_I[0]/(phi*(p_dep**2))
    # b = 16
    # popt_error_I[0] = (phi*(p_dep**2))*b

    # Plotting
    plt.figure()
    plt.plot(k_List, Tr_Dist_Disp, 'o-', color='red', label='Data')
    plt.plot(k_List, linear_func(k_List, *popt_error_I), '--', 
            label=f'Fit: {b:.1f}·k $\phi$ p²')

    # Set x-axis to integer ticks and grids
    plt.xticks(np.arange(min(k_List), max(k_List)+1, 1))  # +1 to include the last integer
    plt.xlabel('k')
    plt.ylabel('Trace Distance')
    plt.grid(True, which='both', axis='both')  # Ensure both major grids are shown
    plt.legend()
    plt.show()

    # Save the plot
    plt.savefig('Projection_ErrorScale_250828test.eps', format='eps', dpi=1000, bbox_inches='tight')


    #############################################

    # Save to a .pkl file !!!

    import pickle

    chi_ZZ = 2*np.pi*5e6    # Dispersive coupling strength
    Omega = 2*np.pi*20e6    # g-f driving strength

    # k_List = [3,4,5]

    # # fix the physical error rate
    # p_dep = 1e-3

    # # fix the sample number in a single trial
    # num_shots = 1e7

    # # repeat number of trials
    # num_trials = 30

    # # fix the rotation angle
    # phi = 1e-3

    # rounds_post = 3
    # rounds_prep = 3
    # rounds = 4      # z_distance-rounds_prep: enough for the later QEC

    # List of variable names to save
    vars_to_save = {
        'num_shots': num_shots,
        'rounds_post': rounds_post,
        'rounds_prep': rounds_prep,
        'rounds': rounds,
        'phi': phi,
        'k_List': k_List,  # k values list: [3,4,5]
        'Result_Disp_List': Result_Disp_List, # Trace distance and failure probability: major results
        'chi_ZZ': chi_ZZ,
        'Omega': Omega,
        'bk_frequencies': bk_frequencies,   # k=5 case data
        'bk_fail_shots': bk_fail_shots,  # k=5 case sampled data
    }

    # Save to .pkl file
    with open(Output_Filename, 'wb') as f:
        pickle.dump(vars_to_save, f)


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows executable support
    main()
