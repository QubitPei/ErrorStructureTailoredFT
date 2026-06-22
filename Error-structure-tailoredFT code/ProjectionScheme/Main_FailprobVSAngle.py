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
global ncpus, z_distance, x_distance, num_shots_max, num_shots_min, Output_Filename, p_dep

z_distance_list = [12,18]
p_dep = 1e-3
# z_distance = 18
# x_distance = 19
# phi = 1e-3
# num_shots_min = 1e7
# num_shots_max = 1.5e8   #!!! maximum allowed sample number for a given noise parameter p_dep and a given error configuration

num_shots_min = 1e6
num_shots_max = 1.5e6   #!!! maximum allowed sample number for a given noise parameter p_dep and a given error configuration

Output_Filename = 'MainProjFailprobVSAngle251113.pkl'   #!!! name of the output data file

# try:
#     ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
# except KeyError:
#     ncpus = multiprocessing.cpu_count()
ncpus = 4     #!!! number of cpus to be used


### Define some useful functions: for parallized Proj_Count_naive

# function to solve theta
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

def expected_sample_binary_strings(k: int, theta: float, N: int) -> Counter[str]:
    """
    Generates the expected combined frequencies of k-bit binary strings based on theta.

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
    expected_frequencies = Counter({binary_strings[i]: int(probabilities[i]*N) for i in range(2**k)})

    # Combine frequencies of binary strings and their complements
    combined_frequencies = Counter()
    seen = set()

    for binary_string, count in expected_frequencies.items():
        complement = ''.join('1' if bit == '0' else '0' for bit in binary_string)
        key = min(binary_string, complement)
        if key not in seen:
            combined_count = expected_frequencies[binary_string] + expected_frequencies.get(complement, 0)
            combined_frequencies[key] = combined_count
            seen.add(key)
            seen.add(complement)

    return combined_frequencies


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



# main worker function to be parallized: for given p_dep, d, theta and shot number, estimate the passing probability, and the proportion of b when passing
# use direct ancilla-free dispersive coupling ZZ rotation gate
# direct sample frequencies
def Proj_Count_naive_frequency(rounds: int,  # number of rounds for the syndrome measurement, should >= rounds_post
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
    theta = solve_theta(phi, k, delta=1e-14)

    # bk_frequencies = sample_binary_strings(k = k, theta = theta, N = Num_shots)  # sample and store the bk frequency histogram

    # run naive `sampling' of bk_frequencies to save time
    bk_frequencies = expected_sample_binary_strings(k = k, theta = theta, N = Num_shots)

    # # record the probability of bk_probabilities
    # bk_probabilities = probability_sample_binary_strings(k = k, theta = theta)

    # Step 2: for each bk, sample the failure shots from the stim simulation and store it
    bk_fail_shots = Counter()  # the counter register for failure shots 
    
    for bk, num_bk in sorted(bk_frequencies.items()):
    # for bk, prob_bk in sorted(bk_probabilities.items()):
        #### Simplify the calculation: ignore the high weight terms
        count0 = bk.count('0')
        count1 = bk.count('1')
        if count0 > 1 and count1 > 1:  # high weight bit strings
            # bk_fail_shots[bk] = int(Num_shots)
            bk_fail_shots[bk] = num_bk
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
        
        num_bk_fail = count_logical_failure_Projection(circuit, Post_Dec_List, num_bk)
        # num_bk_fail = count_logical_failure_Projection(circuit, Post_Dec_List, int(Num_shots))

        bk_fail_shots[bk] = num_bk_fail
    
    return bk_frequencies, bk_fail_shots
    # return bk_probabilities, bk_fail_shots


# main worker function to be parallized: for given p_dep, d, theta and shot number, estimate the passing probability, and the proportion of b when passing
# use direct ancilla-free dispersive coupling ZZ rotation gate
# direct sample frequencies
def Proj_Count_naive_frequency_m3(rounds: int,  # number of rounds for the syndrome measurement, should >= rounds_post
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

    if (z_distance % 3) != 0:
        raise ValueError("z distance should be multiples of 3")
    
    k = int(z_distance/3)
    # theta = phi**(1/k)   # rough value: inaccurate when phi becomes large
    theta = solve_theta(phi, k, delta=1e-14)

    # bk_frequencies = sample_binary_strings(k = k, theta = theta, N = Num_shots)  # sample and store the bk frequency histogram

    # run naive `sampling' of bk_frequencies to save time
    bk_frequencies = expected_sample_binary_strings(k = k, theta = theta, N = Num_shots)

    # # record the probability of bk_probabilities
    # bk_probabilities = probability_sample_binary_strings(k = k, theta = theta)

    # Step 2: for each bk, sample the failure shots from the stim simulation and store it
    bk_fail_shots = Counter()  # the counter register for failure shots 
    
    for bk, num_bk in sorted(bk_frequencies.items()):
    # for bk, prob_bk in sorted(bk_probabilities.items()):
        #### Simplify the calculation: ignore the high weight terms
        count0 = bk.count('0')
        count1 = bk.count('1')
        if count0 > 1 and count1 > 1:  # high weight bit strings
            # bk_fail_shots[bk] = int(Num_shots)
            bk_fail_shots[bk] = num_bk
            continue  # Skip this string
        ####
        
        b = ''.join(bit * 3 for bit in bk)  # generate a weight-d string to indicate the location of Pauli errors
        b_list = [bit == '1' for bit in b]
        ### b_list = [bit == '0' for bit in b]
        # print(b_list)

        # generate the whole circuit
        circuit, Post_Dec_List = asym_surfacecode_circuit.ProjectionCircuit_Check_naive_m3(b_list, 
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
        
        num_bk_fail = count_logical_failure_Projection(circuit, Post_Dec_List, num_bk)
        # num_bk_fail = count_logical_failure_Projection(circuit, Post_Dec_List, int(Num_shots))

        bk_fail_shots[bk] = num_bk_fail
    
    return bk_frequencies, bk_fail_shots
    # return bk_probabilities, bk_fail_shots



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

    # sweep the angle
    phi_min = 1e-4
    phi_max = 2e-1
    phi_list = np.logspace(np.log10(phi_min), np.log10(phi_max), num=30)


    ### Run the sampling: d = 12 
    scanner = phi_list

    z_distance = 12
    x_distance = 13

    with multiprocessing.Pool(processes = min(ncpus, len(scanner))) as pool:
        processes = []
        param_list = []  # used to track p_dep and m
        
        # Submit all tasks first
        for id, phi in enumerate(scanner):
            # adaptively adjust num_shots
            # num_shots_min = 3e6
            # num_shots_max = 2e8

            ZZtype = 1
            num_shots = num_shots_max
            
            for m in [2, 3]:
                param_list.append((phi, m, num_shots))
                if m == 2:
                    processes.append(
                        pool.apply_async(
                            Proj_Count_naive_frequency,   # Proj_Count_naive is the worker function to be parallelized
                            args=(
                                rounds, rounds_post, rounds_prep,
                                ZZtype, x_distance, z_distance,
                                p_dep, num_shots, phi
                            )
                        )
                    )
                else:
                    processes.append(
                        pool.apply_async(
                            Proj_Count_naive_frequency_m3,   # Proj_Count_naive is the worker function to be parallelized
                            args=(
                                rounds, rounds_post, rounds_prep,
                                ZZtype, x_distance, z_distance,
                                p_dep, num_shots, phi
                            )
                        )
                    )
        

        # Collect results after all submissions
        results_dict = {}
        for (phi, m, num_shots), process in zip(param_list, processes):
            try:
                bk_frequencies, bk_fail_shots = process.get()
                
                if phi not in results_dict:
                    results_dict[phi] = {}
                    
                results_dict[phi][m] = {
                    'frequencies': bk_frequencies,
                    'fail_shots': bk_fail_shots,
                    'num_shots': num_shots,
                    'trace_distance': None,
                    'fail_probability': None
                }
            except Exception as e:
                print(f"Error processing phi={phi}, m={m}: {str(e)}")
                # Handle missing data explicitly
                if phi not in results_dict:
                    results_dict[phi] = {}
                results_dict[phi][m] = {
                    'frequencies': {},
                    'fail_shots': Counter(),
                    'num_shots': num_shots,
                    'trace_distance': np.nan,
                    'fail_probability': np.nan
                }
        
        # Post-processing
        for phi in results_dict:
            for m in results_dict[phi]:
                data = results_dict[phi][m]
                if data['frequencies']:  # Only process valid data
                    # Tr_Dist, P_fail = util.Trace_distace_estimate(
                    Tr_Dist, P_fail = util.Trace_distace_estimate_frequency(
                        phi=phi,
                        z_distance=z_distance,
                        Num_shots=data['num_shots'],
                        # bk_probabilities=data['frequencies'],
                        bk_frequencies=data['frequencies'],
                        bk_fail_shots=data['fail_shots']
                    )
                    data['trace_distance'] = Tr_Dist
                    data['fail_probability'] = P_fail


    # Extract results for plotting (outside pool context)
    phi_list = sorted(results_dict.keys())
    Fail_Prob_m2_12 = []
    Fail_Prob_m3_12 = []
    Tr_Dist_m2_12 = []
    Tr_Dist_m3_12 = []

    for phi in phi_list:
        data = results_dict[phi]
        # Handle m=2
        if 2 in data:
            Tr_Dist_m2_12.append(data[2]['trace_distance'])
            Fail_Prob_m2_12.append(data[2]['fail_probability'])
        else:
            Tr_Dist_m2_12.append(np.nan)
            Fail_Prob_m2_12.append(np.nan)
        
        # Handle m=3
        if 3 in data:
            Tr_Dist_m3_12.append(data[3]['trace_distance'])
            Fail_Prob_m3_12.append(data[3]['fail_probability'])
        else:
            Tr_Dist_m3_12.append(np.nan)
            Fail_Prob_m3_12.append(np.nan)

    # Cleanup shared memory explicitly
    try:
        for (phi, m, _) in param_list:
            shm_name = f"bk_{phi}_{m}"
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
    except:
        pass

    
    ### Run the sampling: d = 18 
    # scanner = phi_list

    z_distance = 18
    x_distance = 19

    with multiprocessing.Pool(processes = min(ncpus, len(scanner))) as pool:
        processes = []
        param_list = []  # used to track p_dep and m
        
        # Submit all tasks first
        for id, phi in enumerate(scanner):
            # adaptively adjust num_shots
            # num_shots_min = 3e6
            # num_shots_max = 2e8

            ZZtype = 1
            # num_shots = np.floor(num_shots_min*((p_dep_max/p_dep)**(1.5)))
            num_shots = num_shots_max
            
            for m in [2, 3]:
                param_list.append((phi, m, num_shots))
                if m == 2:
                    processes.append(
                        pool.apply_async(
                            Proj_Count_naive_frequency,   # Proj_Count_naive is the worker function to be parallelized
                            args=(
                                rounds, rounds_post, rounds_prep,
                                ZZtype, x_distance, z_distance,
                                p_dep, num_shots, phi
                            )
                        )
                    )
                else:
                    processes.append(
                        pool.apply_async(
                            Proj_Count_naive_frequency_m3,   # Proj_Count_naive is the worker function to be parallelized
                            args=(
                                rounds, rounds_post, rounds_prep,
                                ZZtype, x_distance, z_distance,
                                p_dep, num_shots, phi
                            )
                        )
                    )
        

        # Collect results after all submissions
        results_dict = {}
        for (phi, m, num_shots), process in zip(param_list, processes):
            try:
                bk_frequencies, bk_fail_shots = process.get()
                
                if phi not in results_dict:
                    results_dict[phi] = {}
                    
                results_dict[phi][m] = {
                    'frequencies': bk_frequencies,
                    'fail_shots': bk_fail_shots,
                    'num_shots': num_shots,
                    'trace_distance': None,
                    'fail_probability': None
                }
            except Exception as e:
                print(f"Error processing phi={phi}, m={m}: {str(e)}")
                # Handle missing data explicitly
                if phi not in results_dict:
                    results_dict[phi] = {}
                results_dict[phi][m] = {
                    'frequencies': {},
                    'fail_shots': Counter(),
                    'num_shots': num_shots,
                    'trace_distance': np.nan,
                    'fail_probability': np.nan
                }
        
        # Post-processing
        for phi in results_dict:
            for m in results_dict[phi]:
                data = results_dict[phi][m]
                if data['frequencies']:  # Only process valid data
                    # Tr_Dist, P_fail = util.Trace_distace_estimate(
                    Tr_Dist, P_fail = util.Trace_distace_estimate_frequency(
                        phi=phi,
                        z_distance=z_distance,
                        Num_shots=data['num_shots'],
                        # bk_probabilities=data['frequencies'],
                        bk_frequencies=data['frequencies'],
                        bk_fail_shots=data['fail_shots']
                    )
                    data['trace_distance'] = Tr_Dist
                    data['fail_probability'] = P_fail


    # Extract results for plotting (outside pool context)
    phi_list = sorted(results_dict.keys())
    Fail_Prob_m2_18 = []
    Fail_Prob_m3_18 = []
    Tr_Dist_m2_18 = []
    Tr_Dist_m3_18 = []

    for phi in phi_list:
        data = results_dict[phi]
        # Handle m=2
        if 2 in data:
            Tr_Dist_m2_18.append(data[2]['trace_distance'])
            Fail_Prob_m2_18.append(data[2]['fail_probability'])
        else:
            Tr_Dist_m2_18.append(np.nan)
            Fail_Prob_m2_18.append(np.nan)
        
        # Handle m=3
        if 3 in data:
            Tr_Dist_m3_18.append(data[3]['trace_distance'])
            Fail_Prob_m3_18.append(data[3]['fail_probability'])
        else:
            Tr_Dist_m3_18.append(np.nan)
            Fail_Prob_m3_18.append(np.nan)

    # Cleanup shared memory explicitly
    try:
        for (phi, m, _) in param_list:
            shm_name = f"bk_{phi}_{m}"
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
    except:
        pass


    ######################

    ### Plot the results

    # Assuming you have your data lists already defined:
    # Fail_Prob_m2_12, Fail_Prob_m3_12, Fail_Prob_m2_18, Fail_Prob_m3_18, p_dep_list

    # Create the plot
    # plt.figure(figsize=(10, 8))

    # Define colors and line styles
    colors = {
        'm2': '#1f77b4',  # Blue for m=2
        'm3': '#d62728',   # Red for m=3
        'hw': '#555555'   # Gray for hardware line  
    }

    line_styles = {
        'd12': '-',       # Solid line for d=12
        'd18': '--'       # Dashed line for d=18
    }

    # Plot all four lines with specified styling
    plt.plot(phi_list, 1- np.array(Fail_Prob_m2_12), 
            color=colors['m2'], linestyle=line_styles['d12'], 
            linewidth=2, marker='o', markersize=5, label='d=12, m=2')

    plt.plot(phi_list, 1- np.array(Fail_Prob_m3_12), 
            color=colors['m3'], linestyle=line_styles['d12'], 
            linewidth=2, marker='s', markersize=5, label='d=12, m=3')

    plt.plot(phi_list, 1- np.array(Fail_Prob_m2_18), 
            color=colors['m2'], linestyle=line_styles['d18'], 
            linewidth=2, marker='^', markersize=5, label='d=18, m=2')

    plt.plot(phi_list, 1- np.array(Fail_Prob_m3_18), 
            color=colors['m3'], linestyle=line_styles['d18'], 
            linewidth=2, marker='D', markersize=5, label='d=18, m=3')

    plt.xscale('log')
    plt.yscale('log')

    plt.axvline(x=1e-3, color=colors['hw'], linestyle=':', label='current hardware', linewidth=2)

    # Customize the plot with larger fonts
    plt.xlabel('Rotation angle phi', fontsize=16)  # Increased from 12 to 16
    plt.ylabel('Successful Probability', fontsize=16)  # Increased from 12 to 16

    # Increase tick label size
    plt.xticks(fontsize=14)  # Larger x-axis tick labels
    plt.yticks(fontsize=14)  # Larger y-axis tick labels

    # Larger legend
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)  # Increased from 10 to 12

    plt.grid(True, which="both", ls="-", alpha=0.6)

    # Adjust layout and show
    plt.tight_layout()

    # Save the plot
    plt.savefig('Proj_PfailVSAngle_251113.eps', format='eps', dpi=1000, bbox_inches='tight')
    plt.show()


    #############################################

    ### Save the results to a .pkl file

    samplenumber = {"num_shots_max": str(num_shots_max)} 
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
    Result_m2_12_List = np.column_stack((Tr_Dist_m2_12, Fail_Prob_m2_12))  # Trace distance, Failure prob. for m=2, d=12
    Result_m3_12_List = np.column_stack((Tr_Dist_m3_12, Fail_Prob_m3_12))   # Trace distance, Failure prob. for m=3, d=12
    Result_m2_18_List = np.column_stack((Tr_Dist_m2_18, Fail_Prob_m2_18))  # Trace distance, Failure prob. for m=2, d=18
    Result_m3_18_List = np.column_stack((Tr_Dist_m3_18, Fail_Prob_m3_18))   # Trace distance, Failure prob. for m=3, d=18


    # List of variable names to save
    vars_to_save = {
        'phi_list': phi_list,
        'z_distance': z_distance,
        'x_distance': x_distance,
        'rounds_post': rounds_post,
        'rounds_prep': rounds_prep,
        'rounds': rounds,
        'p_dep': p_dep,
        'Result_m2_12_List': Result_m2_12_List, 
        'Result_m3_12_List': Result_m3_12_List, 
        'Result_m2_18_List': Result_m2_18_List,  
        'Result_m3_18_List': Result_m3_18_List,
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
