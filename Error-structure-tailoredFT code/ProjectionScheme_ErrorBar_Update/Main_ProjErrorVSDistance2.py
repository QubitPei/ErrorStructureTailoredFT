import numpy as np
import asym_surfacecode_circuit
import util2

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
from tqdm import tqdm


###############################################


### Core variables and file name

# some configuration parameters
global ncpus, p_dep, phi, num_shots, k_List

p_dep = 1e-3
phi = 1e-3
num_shots = 5e8   #!!! sample number for a given k value
k_List = [3,4,5,6,7]    #!!! the z-distance of the surface code to be considered; k

# try:
#     ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
# except KeyError:
#     ncpus = multiprocessing.cpu_count()
ncpus = 26     #!!! number of cpus to be used

fig_name = 'Projection_ErrorScale_260109_m2.eps'

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


# # 新的 Worker 函数：在进程内直接统计失败次数，避免回传大数据
def sample_and_count_failures(shots: int, circuit: stim.Circuit, Post_Dec_List: list[int], batch_size: int = 1000000) -> int:
    """
    在 Worker 内部进行分批采样和统计，极低内存占用。
    """
    sampler = circuit.compile_detector_sampler()
    total_failures = 0
    
    # 将总 shots 分批处理
    num_batches = (shots + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches)):
        # 计算当前批次的大小
        current_batch_shots = min(batch_size, shots - i * batch_size)
        
        # 1. 采样 (只占用这一小批的内存)
        detection_events, _ = sampler.sample(current_batch_shots, separate_observables=True)
        
        # 2. 统计
        selected_columns = detection_events[:, Post_Dec_List]
        abandoned_rows = np.any(selected_columns, axis=1)
        total_failures += int(np.sum(abandoned_rows))
        
        # 3. 循环结束，detection_events 自动释放，内存回收
    return total_failures


def count_logical_failure_Projection_parallel(circuit: stim.Circuit, Post_Dec_List: list[int], num_shots: int, ncpus: int) -> int:
    '''
    For a given sample number and circuit, and all the post-select detector locations, calculate the number of failed shots
    
    Use multiprocessing for parallelization.
    '''
    # 计算每个核心的任务量
    shots_per_cpu = num_shots // ncpus
    # 处理余数，确保总数正确
    remainder = num_shots % ncpus
    shots_args = [shots_per_cpu + 1 if i < remainder else shots_per_cpu for i in range(ncpus)]

    # 使用 starmap 并行运行，注意参数列表现在包含了 Post_Dec_List
    with multiprocessing.Pool(ncpus) as pool:
        results = pool.starmap(
            sample_and_count_failures,
            [(shots, circuit, Post_Dec_List) for shots in shots_args]
        )
    
    # results 现在是一个整数列表 [fail_count_1, fail_count_2, ...]
    # 直接求和即可
    num_fail = sum(results)
    
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
        raise ValueError("z distance should be multiples of 3.")
    
    k = int(z_distance/2)
    # theta = phi**(1/k)   # rough value: inaccurate when phi becomes large
    theta = solve_theta(phi, k, delta=1e-14)

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
        
        num_bk_fail = count_logical_failure_Projection_parallel(circuit, Post_Dec_List, int(Num_shots), ncpus)

        bk_fail_shots[bk] = num_bk_fail
    
    # return bk_frequencies, bk_fail_shots
    return bk_probabilities, bk_fail_shots



################################################################

### main function: run the sampling

def main(name):
    # Initialization
    # ncpus = 3

    global ncpus, p_dep, phi, num_shots, k_List

    # register for the results
    # Result_Dep_List = np.zeros((len(k_List), 3))   # Trace distance, Failure prob. for naive depelorizing ZZ
    Result_Disp_List = np.zeros((len(k_List), 3))   # Trace distance, Failure prob. for dispersive ZZ

    start_time = time.time()


    for id, k in enumerate(k_List):
        
        # fix the code parameters
        z_distance = 2*k
        x_distance = 2*k + 1
        
        # z_distance = 3*k
        # if (z_distance % 2 == 0):
        #     x_distance = 3*k + 1
        # else:
        #     x_distance = 3*k

        
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
            z_distance = z_distance,  # for the projection scheme, z_distance should be the multiples of 3, x_distance should be odd
            p_dep = p_dep,
            Num_shots = num_shots,
            phi = phi,
            )
        Tr_Dist, P_fail = util2.Trace_distace_estimate(phi = phi, z_distance = z_distance, 
                            Num_shots = num_shots, bk_probabilities = bk_frequencies, 
                            bk_fail_shots = bk_fail_shots)
        Std_Tr_Dist = util2.Error_bar_estimate(phi = phi, z_distance = z_distance, 
                            Num_shots = num_shots, bk_probabilities = bk_frequencies, 
                            bk_fail_shots = bk_fail_shots)

        # save the results
        Result_Disp_List[id, :] = [Tr_Dist, P_fail, Std_Tr_Dist]
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
    Tr_Dist_Disp = collected_data_2D[:, 0]
    Fail_Prob_Disp = collected_data_2D[:, 1]
    Std_Tr_Dist_Disp = collected_data_2D[:, 2]

    # Fit the data - ensure k_List is numpy array
    k_List = np.asarray(k_List)
    popt_error_I, _ = curve_fit(linear_func, k_List, Tr_Dist_Disp)

    # Calculate b - ensure phi and p_dep are scalars
    phi = float(phi)  # Convert to float if not already
    p_dep = 1e-3
    Std_ck_Disp = Std_Tr_Dist_Disp/(phi*(p_dep**2))
    b = 3.3

    # Calculate the scaled values for plotting
    Tr_Dist_Disp_scaled = Tr_Dist_Disp/(phi*(p_dep**2))
    # Error bars in the scaled units (2 standard errors)
    error_bars_scaled = 2 * Std_Tr_Dist_Disp/(phi*(p_dep**2))

    # Plotting
    plt.figure()
    # Plot data points with error bars
    plt.errorbar(k_List, Tr_Dist_Disp_scaled, 
                yerr=error_bars_scaled, 
                fmt='o-', 
                color='red', 
                label='Data',
                capsize=5,  # Adds horizontal caps to error bars
                capthick=1.5,  # Thickness of caps
                elinewidth=1.5)  # Thickness of error bar lines

    plt.plot(k_List, b*k_List, '--', 
            label=f'Upper bound {b:.1f}·k')

    # Set x-axis to integer ticks and grids
    plt.xticks(np.arange(min(k_List), max(k_List)+1, 1))
    plt.xlabel('Number of rotation gates k', fontsize=16)
    plt.ylabel(r'$c(k)=D_{Tr}/(p^2\cdot |\varphi|)$', fontsize=16)
    plt.grid(True, which='both', axis='both')
    plt.legend(fontsize=16)

    # Increase tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Optional: Adjust y-axis limits to accommodate error bars
    plt.ylim(bottom=0)  # Ensure plot starts at 0 since values are positive

    # Save the plot
    plt.savefig(fig_name, format='eps', dpi=1000, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(3)  # Show for 3 seconds
    plt.close()   # Then close

    #############################################

    # Save to a .pkl file !!!

    import pickle

    chi_ZZ = 2*np.pi*5e6    # Dispersive coupling strength
    Omega = 2*np.pi*20e6    # g-f driving strength

    # List of variable names to save
    vars_to_save = {
        'num_shots': num_shots,
        'rounds_post': rounds_post,
        'rounds_prep': rounds_prep,
        'rounds': rounds,
        'phi': phi,
        'k_List': k_List,  # k values list: [3,4,5]
        'Result_Disp_List': Result_Disp_List, # Trace distance, failure probability, standard deviation: major results
        'chi_ZZ': chi_ZZ,
        'Omega': Omega,
        'bk_frequencies': bk_frequencies,   # k=5 case data
        'bk_fail_shots': bk_fail_shots,  # k=5 case sampled data
    }

    # Save to .pkl file
    Output_Filename = name + '.pkl'
    with open(Output_Filename, 'wb') as f:
        pickle.dump(vars_to_save, f)


if __name__ == '__main__':
 #   multiprocessing.freeze_support()  # For Windows executable support
    for i in range(10):
        name = './data/MainProjScale260109_' + str(i)   #!!! name of the output data file
        main(name)