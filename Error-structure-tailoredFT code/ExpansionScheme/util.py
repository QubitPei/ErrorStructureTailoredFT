# some useful function during the surface code expansion and decoding

import stim
import numpy as np
import copy 
import pymatching

import asym_surfacecode_circuit

from collections import Counter

import multiprocessing
from multiprocessing.shared_memory import SharedMemory



# full post-selection case: no QEC
def count_logical_postselected_errors(circuit: stim.Circuit, num_shots: int) -> int:
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    num_errors = 0
    num_fail = 0
    for shot in range(num_shots):
        if any(detection_events[shot]):
            num_fail += 1
        else: 
            if observable_flips[shot][0]:
                num_errors += 1

    error_std = binomial_dist_std(num_shots-num_fail, num_errors)
    return num_errors/(num_shots-num_fail), num_fail/num_shots, error_std


#! use the first M-m detectors for post-selection, and all the detectors for decoding
# According to [arXiv:2403.03272], MWPM applies as long as the detector graph do not have hyperedge
# This was shown to be wrong: since the occurance of the gauge operators, the MWPM cannot be directly applied

# split the detectors for post-selection and QEC
def split_detectors_for_decoding(circuit: stim.Circuit, num_ideal_detectors: int):
    # Get the detector graph
    detector_graph = circuit.detector_error_model(approximate_disjoint_errors = True, decompose_errors = True)
    
    # Split detectors into two groups
    num_detectors = len(list(detector_graph.get_detector_coordinates()))
    if num_ideal_detectors > num_detectors:
        raise ValueError(f"num_ideal_detectors ({num_ideal_detectors}) cannot be greater than number of detectors ({num_detectors})")
    
    # Get all detector indices
    all_detectors = list(range(num_detectors))
    
    # Split into post-selection and decoding detectors
    post_selection_detectors = all_detectors[:-num_ideal_detectors]
    # decoding_detectors = all_detectors[-num_ideal_detectors:]
    decoding_detectors = all_detectors
    
    return post_selection_detectors, decoding_detectors


def count_logical_postselected_errors_QEC(circuit: stim.Circuit, num_ideal_detectors: int, num_shots: int) -> tuple[int, int, int]:
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Split the detectors
    post_select_dets, decode_dets = split_detectors_for_decoding(circuit, num_ideal_detectors)

    '''
    # Create matching graph for just the decoding detectors
    matching_graph = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True), # when erasure or Pauli2 error occurs, we need `approximate_disjoint_errors=True`
        # circuit.detector_error_model(decompose_errors=True),  
        weights={},
        faults_matrix=False,
        timelike_weights=False,
    )
    '''

    # Step 1: first do post-selction, filtering the data points to the `kept_rows`
    selected_columns = detection_events[:, post_select_dets]
    abandoned_rows = np.any(selected_columns, axis=1)
    num_fail = np.sum(abandoned_rows)  # Total abandoned rows

    kept_rows = ~abandoned_rows
    filtered_detection_events = detection_events[kept_rows, :]
    filtered_observable_flips = observable_flips[kept_rows, :]

    # Step 2: then perform decoding for the filtered rows
    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)    
    
    predictions = matcher.decode_batch(filtered_detection_events)

    mismatches = (predictions != filtered_observable_flips)
    num_errors = np.sum(np.any(mismatches, axis=1))

    '''
    num_errors = 0
    for shot in range(num_shots):
        # if any(detection_events[shot]):
        if any( detection_events[shot, det] for det in post_select_dets):  # post select the detectors in the set of post_select_dets: first M-m detectors
            num_fail += 1
        else:   # the post-selection pass
            # Run decoding on post-selected data
            predictions_for_shot = matcher.decode_batch(detection_events[shot])
            if observable_flips[shot][0] != predictions_for_shot:
                num_errors += 1
    '''
    
    error_std = binomial_dist_std(num_shots-num_fail, num_errors)
    return num_errors/(num_shots-num_fail), num_fail/num_shots, error_std


#! use only the final m detectors for decoding

# split the detectors for post-selection and QEC
def split_detectors_for_decoding_partial(circuit: stim.Circuit, num_ideal_detectors: int):
    # Get the detector graph
    detector_graph = circuit.detector_error_model(approximate_disjoint_errors = True, decompose_errors = True)
    
    # Split detectors into two groups
    num_detectors = len(list(detector_graph.get_detector_coordinates()))
    if num_ideal_detectors > num_detectors:
        raise ValueError(f"num_ideal_detectors ({num_ideal_detectors}) cannot be greater than number of detectors ({num_detectors})")
    
    # Get all detector indices
    all_detectors = list(range(num_detectors))
    
    # Split into post-selection and decoding detectors
    post_selection_detectors = all_detectors[:-num_ideal_detectors]
    decoding_detectors = all_detectors[-num_ideal_detectors:]
    # decoding_detectors = all_detectors
    
    return post_selection_detectors, decoding_detectors


def count_logical_postselected_errors_QEC_partial(circuit: stim.Circuit, circuit_ref: stim.Circuit, num_ideal_detectors: int, num_shots: int) -> tuple[int, int, int]:
    # Need to input a reference circuit for decoding!
    # The reference circuit captures the decoding_detectors in the normal circuit!

    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Split the detectors
    post_select_dets, decode_dets = split_detectors_for_decoding_partial(circuit, num_ideal_detectors)

    
    # Step 1: first do post-selction, filtering the data points to the `kept_rows`
    selected_columns = detection_events[:, post_select_dets]
    abandoned_rows = np.any(selected_columns, axis=1)
    num_fail = np.sum(abandoned_rows)  # Total abandoned rows

    kept_rows = ~abandoned_rows
    filtered_detection_events = detection_events[kept_rows, :]
    filtered_detection_events = filtered_detection_events[:, decode_dets]
    filtered_observable_flips = observable_flips[kept_rows, :]

    
    # Step 2: then perform decoding for the filtered rows
    # Create matching graph for just the decoding detectors: using the reference circuit!
    # detector_error_model = circuit_ref.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    detector_error_model = circuit_ref.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)    
    
    predictions = matcher.decode_batch(filtered_detection_events)

    mismatches = (predictions != filtered_observable_flips)
    num_errors = np.sum(np.any(mismatches, axis=1))


    error_std = binomial_dist_std(num_shots-num_fail, num_errors)
    return num_errors/(num_shots-num_fail), num_fail/num_shots, error_std


def binomial_dist_std(tot_events, detected_events):
    p = detected_events/tot_events
    return np.sqrt((p * (1-p))/tot_events)

def map_list_of_list(list_2D, callable_func, params):
    return [[callable_func(_item, params) for _item in _list]for _list in list_2D]

def swap_sets(data_1, data_2):
    data_1_intermediate = copy.deepcopy(data_1)
    data_1 = data_2
    data_2 = data_1_intermediate
    return data_1, data_2



#############################

# programs used for the projection scheme

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

    if num_shots < ncpus*100:  # number of shots too small: no need to parallelize
        ncpus = 1
    
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


def count_logical_failure_Projection_async(circuit: stim.Circuit, Post_Dec_List: list[int], num_shots: int, ncpus: int) -> int:
    '''
    For a given sample number and circuit, and all the post-select detector locations, calculate the number of failed shots
    
    Use multiprocessing for parallelization; using `apply_async` method.
    '''

    # use `apply_async` to do parallized sampling
    with multiprocessing.Pool(ncpus) as pool:
        processes = []
        for idx in range(ncpus):
            processes.append( pool.apply_async( sample_from_circuit, args = (num_shots // ncpus, circuit) ) )
        results = [process.get() for process in processes]
    
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



# subroutine: for *THE IDEAL CIRCUIT with b=00...0*, sample the number of failed shots
def count_logical_failure_QEC_Projection(circuit: stim.Circuit, Post_Dec_List: list[int], num_shots: int) -> tuple[int, int]:
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

    kept_rows = ~abandoned_rows
    filtered_detection_events = detection_events[kept_rows, :]
    filtered_observable_flips = observable_flips[kept_rows, :]

    # Step 2: then perform decoding for the filtered rows
    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
            # we need the assumption of approximate_disjoint_error, because the generic two-qubit Pauli channels and erasure error occur
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)    
    
    predictions = matcher.decode_batch(filtered_detection_events)

    mismatches = (predictions != filtered_observable_flips)
    num_errors = np.sum(np.any(mismatches, axis=1))

    # error_std = binomial_dist_std(num_shots-num_fail, num_errors)
    return num_fail, num_errors


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


# the real bk sampling function based on multinomial sampling
def sample_binary_strings(k: int, theta: float, N: int) -> Counter[str]:
    """
    Samples a k-bit binary string N times using multinomial sampling based on probabilities
    derived from theta, and combines the frequencies of each string with its complement.

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

    # Perform multinomial sampling
    samples = np.random.multinomial(N, probabilities)

    # Create a Counter for the frequencies
    frequencies = Counter({binary_strings[i]: samples[i] for i in range(2**k)})

    # Combine frequencies of binary strings and their complements
    combined_frequencies: dict[str, float]
    seen = set()

    for binary_string, count in frequencies.items():
        complement = ''.join('1' if bit == '0' else '0' for bit in binary_string)
        key = min(binary_string, complement)
        if key not in seen:
            combined_count = frequencies[binary_string] + frequencies.get(complement, 0)
            combined_frequencies[key] = combined_count
            seen.add(key)
            seen.add(complement)

    return combined_frequencies
    # the Counter is a Dict, so it is not ordered. We can sort it by the following way:
    # for binary_string, count in sorted(combined_frequencies.items())

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


# main program: for given p_dep, d, theta and shot number, estimate the passing probability, and the proportion of b when passing
def Proj_Count(rounds: int,  # number of rounds for the syndrome measurement, should >= rounds_post
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
        b = ''.join(bit * 2 for bit in bk)  # generate a weight-d string to indicate the location of Pauli errors
        b_list = [bit == '1' for bit in b]
        ### b_list = [bit == '0' for bit in b]
        # print(b_list)

        # generate the whole circuit
        circuit, Post_Dec_List = asym_surfacecode_circuit.ProjectionCircuit_Check(b_list, 
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

# main program: for given p_dep, d, theta and shot number, estimate the passing probability, and the proportion of b when passing
# use direct dispersive coupling ZZ rotation gate
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

# main program: for given p_dep, d, theta and shot number, estimate the passing probability, and the proportion of b when passing
# use direct dispersive coupling ZZ rotation gate
# use the parallized version of sampling
def Proj_Count_naive_parallel(rounds: int,  # number of rounds for the syndrome measurement, should >= rounds_post
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
        ncpus: int = 1,
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
        # num_bk_fail = count_logical_failure_Projection_async(circuit, Post_Dec_List, int(Num_shots), ncpus)
        
        bk_fail_shots[bk] = num_bk_fail
    
    # return bk_frequencies, bk_fail_shots
    return bk_probabilities, bk_fail_shots



# main program: for given p_dep, d, theta and shot number, estimate the passing probability, and the proportion of b when passing
# use direct dispersive coupling ZZ rotation gate
# use the parallized version of sampling
# perform real sampling based on the frequency value
def Proj_Count_naive_parallel_frequency(rounds: int,  # number of rounds for the syndrome measurement, should >= rounds_post
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
        ncpus: int = 1,
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
            bk_fail_shots[bk] = int(num_bk)
            print('bk=',bk)
            # bk_fail_shots[bk] = int(Num_shots)
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
        num_bk_fail = count_logical_failure_Projection_parallel(circuit, Post_Dec_List, num_bk, ncpus)
        
        # num_bk_fail = count_logical_failure_Projection_async(circuit, Post_Dec_List, int(Num_shots), ncpus)
        
        bk_fail_shots[bk] = num_bk_fail
    
    return bk_frequencies, bk_fail_shots
    # return bk_probabilities, bk_fail_shots



# main program: for given p_dep, d, theta and shot number, estimate the passing probability, and the proportion of b when passing
# use direct dispersive coupling ZZ rotation gate
# use the parallized version of sampling
# perform real sampling based on the frequency value
# EXTENDED ZZ rotation: m=3!
def Proj_Count_naive_parallel_frequency_m3(rounds: int,  # number of rounds for the syndrome measurement, should >= rounds_post
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
        ncpus: int = 1,
        ) -> Counter[str]:

    # Step 1: sample bk, determine the histogram for the later stim circuit simulation.  
    if (z_distance % 3) != 0:
        raise ValueError("z distance should be multples of 3")
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
            bk_fail_shots[bk] = int(num_bk)
            # bk_fail_shots[bk] = int(Num_shots)
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
        
        # num_bk_fail = count_logical_failure_Projection(circuit, Post_Dec_List, num_bk)
        num_bk_fail = count_logical_failure_Projection_parallel(circuit, Post_Dec_List, num_bk, ncpus)
        
        # num_bk_fail = count_logical_failure_Projection_async(circuit, Post_Dec_List, int(Num_shots), ncpus)
        
        bk_fail_shots[bk] = num_bk_fail
    
    return bk_frequencies, bk_fail_shots
    # return bk_probabilities, bk_fail_shots




# based on the counted value, estiamte the trace distance
def Trace_distace_estimate_frequency(phi:float= 0, z_distance: int= None, 
                           Num_shots: int= None, bk_frequencies: Counter[str] = None, 
                           bk_fail_shots: Counter[str] = None) -> tuple[float, float]:
    # count the passed instances
    bk_pass_shots = Counter()
    num_pass = 0
    num_undetected_pass = 0
    for bk, num_bk in sorted(bk_frequencies.items()):
        num_bk_pass = num_bk - bk_fail_shots[bk]
        bk_pass_shots[bk] = num_bk_pass
        num_pass += num_bk_pass
        if any(bit == '1' for bit in bk):  # non-zero string
            num_undetected_pass += num_bk_pass

    # Tr_Dist based on the Bayesian probability: P_{ud|pass}
    k = int(z_distance/2)
    # theta = phi**(1/k)
    theta = solve_theta(phi, k, delta=1e-14)
    # phi_prime = - theta**(k-2)
    p_error = (np.sin(theta)**2) * (np.cos(theta)**2) * ( np.sin(theta)**(2*k-4) + np.cos(theta)**(2*k-4) )
    phi_prime = - np.arcsin( (1/np.sqrt(p_error)) * np.sin(theta)**(k-1) * np.cos(theta) )
    
    Tr_Dist = np.sin(phi-phi_prime)*(num_undetected_pass/num_pass)
    P_fail = 1 - num_pass/Num_shots
    if P_fail > 1.001:
        raise ValueError(f"P_fail cannot be larger than 1!")

    return (Tr_Dist, P_fail)



def Trace_distace_estimate(phi:float= 0, z_distance: int= None, 
                           Num_shots: int= None, bk_probabilities: dict[str,float] = None, 
                           bk_fail_shots: Counter[str] = None) -> tuple[float, float]:
    
    # # count the passed instances
    # bk_pass_shots = Counter()
    # num_pass = 0
    # num_undetected_pass = 0
    # for bk, num_bk in sorted(bk_frequencies.items()):
    #     num_bk_pass = num_bk - bk_fail_shots[bk]
    #     bk_pass_shots[bk] = num_bk_pass
    #     num_pass += num_bk_pass
    #     if any(bit == '1' for bit in bk):  # non-zero string
    #         num_undetected_pass += num_bk_pass

    # Tr_Dist = np.abs(phi)*num_undetected_pass/num_pass
    # P_fail = 1 - num_pass/Num_shots

    # count the collective probability: P_{bk,pass}
    P_bk_pass: dict[str, float] = {}
    Ppass = 0
    P_ud_pass = 0
    for bk, num_bk in sorted(bk_probabilities.items()):
        P_bk_pass_value = bk_probabilities[bk] * ( 1 - (bk_fail_shots[bk]/Num_shots) )
        P_bk_pass[bk] = P_bk_pass_value
        Ppass += P_bk_pass_value
        if any(bit == '1' for bit in bk):  # non-zero string
            P_ud_pass += P_bk_pass_value

    # Tr_Dist based on the Bayesian probability: P_{ud|pass}
    k = int(z_distance/2)
    # theta = phi**(1/k)
    theta = solve_theta(phi, k, delta=1e-14)
    # phi_prime = - theta**(k-2)
    p_error = (np.sin(theta)**2) * (np.cos(theta)**2) * ( np.sin(theta)**(2*k-4) + np.cos(theta)**(2*k-4) )
    phi_prime = - np.arcsin( (1/np.sqrt(p_error)) * np.sin(theta)**(k-1) * np.cos(theta) )

    Tr_Dist = np.sin(phi-phi_prime)*P_ud_pass/Ppass
    P_fail = 1 - Ppass

    return (Tr_Dist, P_fail)

def Trace_distace_estimate_m3(phi:float= 0, z_distance: int= None, 
                           Num_shots: int= None, bk_probabilities: dict[str,float] = None, 
                           bk_fail_shots: Counter[str] = None) -> tuple[float, float]:
    
    # # count the passed instances
    # bk_pass_shots = Counter()
    # num_pass = 0
    # num_undetected_pass = 0
    # for bk, num_bk in sorted(bk_frequencies.items()):
    #     num_bk_pass = num_bk - bk_fail_shots[bk]
    #     bk_pass_shots[bk] = num_bk_pass
    #     num_pass += num_bk_pass
    #     if any(bit == '1' for bit in bk):  # non-zero string
    #         num_undetected_pass += num_bk_pass

    # Tr_Dist = np.abs(phi)*num_undetected_pass/num_pass
    # P_fail = 1 - num_pass/Num_shots

    # count the collective probability: P_{bk,pass}
    P_bk_pass: dict[str, float] = {}
    Ppass = 0
    P_ud_pass = 0
    for bk, num_bk in sorted(bk_probabilities.items()):
        P_bk_pass_value = bk_probabilities[bk] * ( 1 - (bk_fail_shots[bk]/Num_shots) )
        P_bk_pass[bk] = P_bk_pass_value
        Ppass += P_bk_pass_value
        if any(bit == '1' for bit in bk):  # non-zero string
            P_ud_pass += P_bk_pass_value

    # Tr_Dist based on the Bayesian probability: P_{ud|pass}
    k = int(z_distance/3)
    # theta = phi**(1/k)
    theta = solve_theta(phi, k, delta=1e-14)
    # phi_prime = - theta**(k-2)
    p_error = (np.sin(theta)**2) * (np.cos(theta)**2) * ( np.sin(theta)**(2*k-4) + np.cos(theta)**(2*k-4) )
    phi_prime = - np.arcsin( (1/np.sqrt(p_error)) * np.sin(theta)**(k-1) * np.cos(theta) )

    Tr_Dist = np.sin(phi-phi_prime)*P_ud_pass/Ppass
    P_fail = 1 - Ppass

    return (Tr_Dist, P_fail)