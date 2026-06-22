import stim
import numpy as np
import scipy as sp
import util2
# import json
from typing import Callable, Set, List, Dict, Tuple, NamedTuple, Optional
from dataclasses import dataclass
import math 

import pickle   # for the data loading

"""
This program generates the necessary functions used for the projection schemes, 
including the asymmetric rotated surface code syndrome measurment, plus-state preparation, rotation error injection and logical error counting. 
"""


def append_anti_basis_error(circuit: stim.Circuit, targets: List[int], p: float, basis: str) -> None:
    if p > 0:
        if basis == "X":
            circuit.append_operation("Z_ERROR", targets, p)
        else:
            circuit.append_operation("X_ERROR", targets, p)

@dataclass
class CircuitGenParameters:   # the class is used to record the circuit & noise parameters and apply noisy gates
    rounds: int
    distance: int = None
    x_distance: int = None
    z_distance: int = None    # in our case, z_distance = x_distance - 1
    after_clifford_depolarization: float = 0
    before_round_data_depolarization: float = 0
    before_measure_flip_probability: float = 0
    after_reset_flip_probability: float = 0
    exclude_other_basis_detectors: bool = False
    qubit_initialization_pattern: str = ''
    top_left_2_body_meas_type: str = 'X'  # keep track of the direction of the rotated surface code

    # the following methods are used to apply the noisy gates to an input circuit with the noise parameters in CircuitGenPatameters
    def append_begin_round_tick(
            self,
            circuit: stim.Circuit,
            data_qubits: List[int],
            if_noiseless: bool = False,
    ) -> None:
        circuit.append_operation("TICK", [])
        if not if_noiseless and self.before_round_data_depolarization > 0:
            circuit.append_operation("DEPOLARIZE1", data_qubits, self.before_round_data_depolarization)

    def append_unitary_1(
            self,
            circuit: stim.Circuit,
            name: str,
            targets: List[int],
            if_noiseless: bool = False,
    ) -> None:
        circuit.append_operation(name, targets)
        if not if_noiseless and self.after_clifford_depolarization > 0:
            circuit.append_operation("DEPOLARIZE1", targets, self.after_clifford_depolarization)
            # circuit.append_operation("DEPOLARIZE1", targets, 0)     # for test usage

    def append_unitary_2(
            self,
            circuit: stim.Circuit,
            name: str,
            targets: List[int],
            if_noiseless: bool = False,
    ) -> None:
        circuit.append_operation(name, targets)
        if not if_noiseless and self.after_clifford_depolarization > 0:
            circuit.append_operation("DEPOLARIZE2", targets, self.after_clifford_depolarization)

    def append_reset(
            self,
            circuit: stim.Circuit,
            targets: List[int],
            basis: str = "Z",
            if_noiseless: bool = False,
    ) -> None:
        circuit.append_operation("R" + basis, targets)
        if not if_noiseless and self.after_reset_flip_probability > 0:
            append_anti_basis_error(circuit, targets, self.after_reset_flip_probability, basis)

    def append_depolarize_error(
            self,
            circuit: stim.Circuit,
            targets: List[int],
            error: float = 0.0,
    ) -> None:
        
        if error > 0:
            circuit.append_operation("DEPOLARIZE1", targets, error)

    def append_measure(self, circuit: stim.Circuit, targets: List[int], basis: str = "Z", if_noiseless: bool = False) -> None:
        if not if_noiseless and self.before_measure_flip_probability > 0:
            append_anti_basis_error(circuit, targets, self.before_measure_flip_probability, basis)
        circuit.append_operation("M" + basis, targets)

    def append_measure_reset(
            self,
            circuit: stim.Circuit,
            targets: List[int],
            basis: str = "Z",
            if_noiseless: bool = False,
    ) -> None:
        if not if_noiseless and self.before_measure_flip_probability > 0:
            append_anti_basis_error(circuit, targets, self.before_measure_flip_probability, basis)
        circuit.append_operation("MR" + basis, targets)
        if not if_noiseless and self.after_reset_flip_probability > 0:
            append_anti_basis_error(circuit, targets, self.after_reset_flip_probability, basis)

    def append_ideal_swap(
            self,
            circuit: stim.Circuit,
            target_1: int,
            target_2: int,
    ) -> None:
        circuit.append_operation("SWAP", [target_1, target_2])
        # circuit.append_operation("CNOT", [target_1, target_2])
        # circuit.append_operation("CNOT", [target_2, target_1])
        # circuit.append_operation("CNOT", [target_1, target_2])




##########################################
## Step I: define (1) surface-code syndrome measurement circuit and (2) the surface code + state preparation circuit

class surface_code(NamedTuple):   # the named tuple here is used to store the basic information (coordinates) of the surface code
    # distance
    distance: int
    x_distance: int
    z_distance: int

    # data qubits
    data_coords: Set[complex]
    x_observable: List[complex]
    z_observable: List[complex]
    
    # measurement qubits
    x_measure_coords: Set[complex] = set()
    z_measure_coords: Set[complex] = set()
    x_measure_coords_post: Set[complex] = set()
    z_measure_coords_post: Set[complex] = set()

    # zx measurement order
    z_order: List[complex] = []
    x_order: List[complex] = []

    # direction of the code
    top_left_2_body_meas_type: str = 'X'
    
    # define two dictionaries for fast coordinate <-> index conversion
    coord2ind: Dict[complex, int] = None
    ind2coord: Dict[int, complex] = None


# create the named tuple which store all the coordinate info related to the surface code
def surface_code_initialization(params: CircuitGenParameters) -> surface_code:
    # set the distance for the rotated surface code
    if params.distance is not None:
        distance = params.distance
    else:
        if params.x_distance is not None:
            x_distance = params.x_distance
        if params.z_distance is not None:
            z_distance = params.z_distance
        distance = min(x_distance,z_distance)
    
    # check whether distance is legal
    if params.distance is not None and params.distance < 2:
        raise ValueError("Need a distance >= 2")
    if params.x_distance is not None and (params.x_distance < 2 or
                                          params.z_distance < 2):
        raise ValueError("Need a distance >= 2")

    # Place data qubits: specified by the complex-valued coordinates q
    # default: assume the top_left_2_body_meas_type: str = 'X'
    data_coords: Set[complex] = set()
    x_observable: List[complex] = []
    z_observable: List[complex] = []
    
    for x in [i + 0.5 for i in range(z_distance)]:
        for y in [i + 0.5 for i in range(x_distance)]:
            q = x * 2 + y * 2j
            data_coords.add(q)
            if y == 0.5:
                z_observable.append(q)
            if x == 0.5:
                x_observable.append(q)

    # Place measurement qubits: specified by the complex-valued coordinates q
    # x_measure_coords labels the X stabilizer measurements, not the logical measurements
    # we also specify the measurment qubits that will be used for projection post-selection
    x_measure_coords: Set[complex] = set()
    z_measure_coords: Set[complex] = set()
    x_measure_coords_post: Set[complex] = set()
    z_measure_coords_post: Set[complex] = set()

    for x in range(z_distance + 1):
        for y in range(x_distance + 1):
            q = x * 2 + y * 2j

            on_boundary_1 = x == 0 or x == z_distance  # vertical boundary check
            on_boundary_2 = y == 0 or y == x_distance  # horizontal boundary check
            parity = (x % 2) != (y % 2) # check the x,y parity difference: if they are different, then parity is TRUE!!!
            # missing checks on the boundaries
            if on_boundary_1 and parity:
                continue
            if on_boundary_2 and not parity:
                continue

            if parity:   # if different parity: X; else Z
                x_measure_coords.add(q)
                if q.imag < 7:   # for the first four columns: the measurement qubit belongs to the projection post-selection detectors
                    x_measure_coords_post.add(q)
            else:
                z_measure_coords.add(q)
                if q.imag < 7:   # for the first four columns: the measurement qubit belongs to the projection post-selection detectors
                    z_measure_coords_post.add(q)

    # Define interaction orders so that hook errors run against the error grain instead of with it.
    z_order: List[complex] = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    x_order: List[complex] = [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]

    # convert the coordinates to indicies
    # we remark that, many redundant q is introduced to keep the coord->index mapping simple
    def coord_to_index(coord: complex, params: dict=None) -> int:
        coord = coord - math.fmod(coord.real, 2) * 1j
        # padded_qubit_nbr = params.prep_4112_params["total_qubit_nbr"]
        padded_qubit_nbr = 0
        return int(coord.real + coord.imag * (z_distance + 0.5)) + padded_qubit_nbr

    # based on the requirement on the top left 2-body check, determine whether to flip the surface code
    # data qubit coordinates keep invariant: change the x,z parity check & observable definition
    if params.top_left_2_body_meas_type == 'Z':
        x_observable, z_observable = util2.swap_sets(x_observable, z_observable)
        x_measure_coords, z_measure_coords = util2.swap_sets(x_measure_coords, z_measure_coords)
        x_measure_coords_post, z_measure_coords_post = util2.swap_sets(x_measure_coords_post, z_measure_coords_post)
        x_order, z_order = util2.swap_sets(x_order, z_order)


    # create the index dictionary for the fast query

    # Index the measurement qubits and data qubits: store them to dictionaries.
    ###  coord: coordinate, complex ;  ind: index, int   
    coord2ind: Dict[complex, int] = {}
    for coord in data_coords:
        coord2ind[coord] = coord_to_index(coord, params)

    for coord in x_measure_coords:
        coord2ind[coord] = coord_to_index(coord, params)

    for coord in z_measure_coords:
        coord2ind[coord] = coord_to_index(coord, params)
    
    ind2coord: Dict[int, complex] = {v: k for k, v in coord2ind.items()}   # define a dictionary from index to coord

    data = {'distance': distance, 'x_distance': x_distance, 'z_distance': z_distance, \
             'data_coords': data_coords, 'x_observable': x_observable, 'z_observable': z_observable, \
            'x_measure_coords': x_measure_coords, 'z_measure_coords': z_measure_coords, \
            'x_measure_coords_post': x_measure_coords_post, 'z_measure_coords_post': z_measure_coords_post,\
            'z_order': z_order, 'x_order': x_order, 'top_left_2_body_meas_type': params.top_left_2_body_meas_type, \
            'coord2ind': coord2ind, 'ind2coord': ind2coord}
    
    
    return surface_code(**data)

### complex coordinate and the true coordinate correspondence example: 1 + 3j -> (1, 3) : the first row, the third column

## detailed surface code SE circuit
def surface_code_SE_circuit(
        params: CircuitGenParameters,
        sc: surface_code, 
        *, # the argument after * must be queried as keyword arguments
        exclude_other_basis_detectors: bool = False,
        wraparound_length: Optional[int] = None  # periodic boundary condition?
) -> tuple[stim.Circuit, int]:
    
    # Store the indicies to dictionaries for the future fast query
    data_qubits = [sc.coord2ind[p] for p in sc.data_coords]
    x_logical_qubits = [sc.coord2ind[p] for p in sc.x_observable]
    z_logical_qubits = [sc.coord2ind[p] for p in sc.z_observable]
    measurement_qubits = [sc.coord2ind[p] for p in sc.x_measure_coords]
    measurement_qubits += [sc.coord2ind[p] for p in sc.z_measure_coords]
    x_measurement_qubits = [sc.coord2ind[p] for p in sc.x_measure_coords]
    x_measurement_qubits_post = [sc.coord2ind[p] for p in sc.x_measure_coords_post]
    z_measurement_qubits_post = [sc.coord2ind[p] for p in sc.z_measure_coords_post]

    all_qubits: List[int] = []
    all_qubits += data_qubits + measurement_qubits

    all_qubits.sort()
    data_qubits.sort()
    measurement_qubits.sort()
    x_measurement_qubits.sort()

    # Reverse index the measurement order used for defining detectors
    data_coord_to_order: Dict[complex, int] = {}
    measure_coord_to_order: Dict[complex, int] = {}
    for q in data_qubits:
        data_coord_to_order[sc.ind2coord[q]] = len(data_coord_to_order)   # this dictionary will provide plain-ordered sequence of all the data qubits: start from 1
    for q in measurement_qubits:
        measure_coord_to_order[sc.ind2coord[q]] = len(measure_coord_to_order)  # this dictionary will provide plain-ordered sequence of all the measurements

    # List out CNOT gate targets using given interaction orders.
    # CNOT gates are specified by the complex-valued coordinates
    cnot_targets: List[List[int]] = [[], [], [], []]  # list it by 4 time bins
    for k in range(4):
        for measure in sorted(sc.x_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + sc.x_order[k]
            if data in sc.coord2ind:
                cnot_targets[k].append(sc.coord2ind[measure])  # measure control data
                cnot_targets[k].append(sc.coord2ind[data])
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                cnot_targets[k].append(sc.coord2ind[measure])
                cnot_targets[k].append(sc.coord2ind[data_wrapped])

        for measure in sorted(sc.z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + sc.z_order[k]
            if data in sc.coord2ind:
                cnot_targets[k].append(sc.coord2ind[data])     # data control measure
                cnot_targets[k].append(sc.coord2ind[measure])
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                cnot_targets[k].append(sc.coord2ind[data_wrapped])
                cnot_targets[k].append(sc.coord2ind[measure])

    # Build the syndrome extraction circuits that make up the surface code cycle
    cycle_actions = stim.Circuit()
    params.append_reset(cycle_actions, measurement_qubits)
    params.append_begin_round_tick(cycle_actions, data_qubits)
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits)
    for targets in cnot_targets:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets)
        # print(targets)
    cycle_actions.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits)
    cycle_actions.append_operation("TICK", [])
    # params.append_measure_reset(cycle_actions, measurement_qubits)
    params.append_measure(cycle_actions, measurement_qubits)

    return cycle_actions



###################################################
## Step II: add projection to the circuit

# In Stim, a detector does not explicitly need to be set to "1" or any specific value. 
# Detectors are defined by asserting that the parity of measurements is always the same, 
# regardless of whether it's even or odd. If a pair of measurements is always 1-0 or 0-1, 
# this is a valid use case for a detector. The detector will identify when the parity 
# deviates from the expected consistent result. 


# function: for a fixed sampled bitstring b, measure logical X, decode and get the outcome
    #! we need to descriminate the detectors for post-selection!
# function: function to sample the bitstirng b and record the failure probability and conditional probability

def ProjectionCircuit(b: list[bool], # a length-(z_distance) boolean vector
        rounds: int,  # number of rounds for the syndrome measurement, should >= rounds_post
        rounds_post: int, # number of rounds for the post-selection in projection, should be >= 1
        rounds_prep: int, # number of rounds for state preparation (also post-selection), should be >= 1
        ZZtype: int, # 0: the naive ZZ rotation gate; 1: the dispersive ZZ rotation gate
        # Note that we do not implement the ZZ gate, but only append the corresponding noise channel.
        # distance: int = None,
        x_distance: int = None,
        z_distance: int = None,  # for the projection scheme, z_distance should be even, x_distance should be odd
        after_clifford_depolarization: float = 0.0,
        before_round_data_depolarization: float = 0.0,
        before_measure_flip_probability: float = 0.0,
        after_reset_flip_probability: float = 0.0,
        # exclude_other_basis_detectors: bool = False,
        top_left_2_body_meas_type: str = 'X',   
        ) -> tuple[stim.Circuit, list[int]] :
    """
        Generates the circuit: for a fixed sampled bitstring input b, 
        (b: list[bool] should be a length-z_distance boolean vector indicating the occurace of operationaol Z errors),
        
        Check whether the X outcome of + state preparation is correct.
        
        Meanwhile, output the detector indicies used for post-selection.

        The generated circuits can include configurable noise.

        The generated circuits include DETECTOR and OBSERVABLE_INCLUDE annotations so
        that their detection events and logical observables can be sampled.

        The generated circuits include TICK annotations to mark the progression of time.
        (E.g. so that converting them using `stimcirq.stim_circuit_to_cirq_circuit` will
        produce a `cirq.Circuit` with the intended moment structure.)

        Note that the toric_code circuits currently only include one of the two logical observables.

        Args:
            code_task: A string identifying the type of circuit to generate. Available
                code tasks are:
                    - "surface_code:rotated_memory_x"
                    - "surface_code:rotated_memory_z"
                    - "surface_code:unrotated_memory_x"
                    - "surface_code:unrotated_memory_z"
                    - "toric_code:unrotated_memory_x"
                    - "toric_code:unrotated_memory_z"
            distance: Defaults to None. The desired code distance of the generated
                circuit. The code distance is the minimum number of physical
                errors needed to cause a logical error. This parameter indirectly determines
                how many qubits the generated circuit uses.
            x_distance: Defaults to None. The desired code distance of the X
            logical
                operator in the generated circuit: the minimum number of X physical
                errors needed to cause a X logical error.
            z_distance: Defaults to None. The desired code distance of the Z
            logical
                operator in the generated circuit: the minimum number of Z physical
                errors needed to cause a Z logical error.
            rounds: How many times the measurement qubits in the generated circuit will
                be measured. Indirectly determines the duration of the generated
                circuit.
            after_clifford_depolarization: Defaults to 0. The probability (p) of
                `DEPOLARIZE1(p)` operations to add after every single-qubit Clifford
                operation and `DEPOLARIZE2(p)` operations to add after every two-qubit
                Clifford operation. The after-Clifford depolarizing operations are only
                included if this probability is not 0.
            before_round_data_depolarization: Defaults to 0. The probability (p) of
                `DEPOLARIZE1(p)` operations to apply to every data qubit at the start of
                a round of stabilizer measurements. The start-of-round depolarizing
                operations are only included if this probability is not 0.
            before_measure_flip_probability: Defaults to 0. The probability (p) of
                `X_ERROR(p)` operations applied to qubits before each measurement (X
                basis measurements use `Z_ERROR(p)` instead). The before-measurement
                flips are only included if this probability is not 0.
            after_reset_flip_probability: Defaults to 0. The probability (p) of
                `X_ERROR(p)` operations applied to qubits after each reset (X basis
                resets use `Z_ERROR(p)` instead). The after-reset flips are only
                included if this probability is not 0.
            exclude_other_basis_detectors: Defaults to False. If True, do not add
                detectors to measurement qubits that are measured in the opposite
                basis to the chosen basis of the logical observable.

        Returns:
            The generated circuit.
            The location of the detectors used for post-selection.
        """

    # initialization
    # parameter class generation
    params = CircuitGenParameters(
            rounds=rounds,
            # distance=distance,
            x_distance=x_distance,
            z_distance=z_distance,
            after_clifford_depolarization=after_clifford_depolarization,
            before_round_data_depolarization=before_round_data_depolarization,
            before_measure_flip_probability=before_measure_flip_probability,
            after_reset_flip_probability=after_reset_flip_probability,
            # exclude_other_basis_detectors=exclude_other_basis_detectors,
            # qubit_initialization_pattern = qubit_initialization_pattern,
            top_left_2_body_meas_type = top_left_2_body_meas_type
        )  # this is a tuple of all the required parameters

    # surface code namedtuple generation
    sc = surface_code_initialization(params)

    # store the indicies to lists for the future fast query
    data_qubits = [sc.coord2ind[p] for p in sc.data_coords]
    x_logical_qubits = [sc.coord2ind[p] for p in sc.x_observable]
    z_logical_qubits = [sc.coord2ind[p] for p in sc.z_observable]
    measurement_qubits = [sc.coord2ind[p] for p in sc.x_measure_coords]
    measurement_qubits += [sc.coord2ind[p] for p in sc.z_measure_coords]
    x_measurement_qubits = [sc.coord2ind[p] for p in sc.x_measure_coords]
    x_measurement_qubits_post = [sc.coord2ind[p] for p in sc.x_measure_coords_post]
    z_measurement_qubits_post = [sc.coord2ind[p] for p in sc.z_measure_coords_post]

    # for p in x_measurement_qubits_post:
    #     print(p)
    # for p in z_measurement_qubits_post:
    #     print(p)

    all_qubits: List[int] = []
    all_qubits += data_qubits + measurement_qubits

    all_qubits.sort()
    data_qubits.sort()
    measurement_qubits.sort()
    x_measurement_qubits.sort()

    # Reverse index the measurement order used for defining detectors
    # count the Z and X type measurement together: mainly for the 
    data_coord_to_order: Dict[complex, int] = {}
    measure_coord_to_order: Dict[complex, int] = {}
    for q in data_qubits:
        data_coord_to_order[sc.ind2coord[q]] = len(data_coord_to_order)   # this dictionary will provide plain-ordered sequence of all the data qubits: start from 1
    for q in measurement_qubits:
        measure_coord_to_order[sc.ind2coord[q]] = len(measure_coord_to_order)  # this dictionary will provide plain-ordered sequence of all the measurements

    # List out CNOT gate targets using given interaction orders.
    # CNOT gates are specified by the complex-valued coordinates
    cnot_targets: List[List[int]] = [[], [], [], []]  # list it by 4 time bins
    for k in range(4):
        for measure in sorted(sc.x_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + sc.x_order[k]
            if data in sc.coord2ind:
                cnot_targets[k].append(sc.coord2ind[measure])  # measure control data
                cnot_targets[k].append(sc.coord2ind[data])

        for measure in sorted(sc.z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + sc.z_order[k]
            if data in sc.coord2ind:
                cnot_targets[k].append(sc.coord2ind[data])     # data control measure
                cnot_targets[k].append(sc.coord2ind[measure])

    # define the syndrome extraction circuits that make up the surface code cycle
    cycle_actions = stim.Circuit()
    params.append_reset(cycle_actions, measurement_qubits)
    params.append_begin_round_tick(cycle_actions, data_qubits)
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits)
    for targets in cnot_targets:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets)
        # print(targets)
    cycle_actions.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits)
    cycle_actions.append_operation("TICK", [])
    # params.append_measure_reset(cycle_actions, measurement_qubits)
    params.append_measure(cycle_actions, measurement_qubits)

    # define the noiseless SE circuit
    cycle_actions_ideal = stim.Circuit()
    params.append_reset(cycle_actions_ideal, measurement_qubits, if_noiseless = True)
    params.append_begin_round_tick(cycle_actions_ideal, data_qubits, if_noiseless = True)
    params.append_unitary_1(cycle_actions_ideal, "H", x_measurement_qubits, if_noiseless = True)
    for targets in cnot_targets:
        cycle_actions_ideal.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions_ideal, "CNOT", targets, if_noiseless = True)
        # print(targets)
    cycle_actions_ideal.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions_ideal, "H", x_measurement_qubits, if_noiseless = True)
    cycle_actions_ideal.append_operation("TICK", [])
    params.append_measure(cycle_actions_ideal, measurement_qubits, if_noiseless = True)


    ####################

    # 1. Plus state preparation by d rounds of syndrome measurement : head

    head = stim.Circuit()
    for k, v in sorted(sc.ind2coord.items()):
        head.append_operation("QUBIT_COORDS", [k], [v.real, v.imag])

    # Ideal initialization of the surface code: all + state for the data qubits
    params.append_reset(head, data_qubits, "X", if_noiseless = True)
    # first round of surface code syndrome check for code expansion
    head += cycle_actions
    # head += cycle_actions_ideal
    # for all the x-type syndrome check, append a detector
    for m_coord in sc.x_measure_coords:
        # all_touched_data_qubit_coord = [x_order_add + m_coord for x_order_add in sc.x_order  if x_order_add + m_coord in sc.data_coords] # first list all x_order + m_coord results, then select the ones in data_coords
        head.append_operation(
            "DETECTOR",
            [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[m_coord])],
            [m_coord.real, m_coord.imag, 0.0] # this provide the coordinate to the detector
            )
            # print("X detectors at: ", m_coord)

    # list of the detectors used for post-selection
    # Record the reversed location: the last detector in the whole circuit is -1 
    Post_Dec_List = []

    # Build the repeated body of the circuit, including the detectors comparing to previous cycles.
    # rounds_prep = 1
    for t in range(rounds_prep):
        SEcirc = cycle_actions.copy()
        # SEcirc = cycle_actions_ideal.copy()
        meas_num = len(measurement_qubits)
        SEcirc.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
        # append detectors on all the measurement qubits
        for m_index in measurement_qubits:
            m_coord = sc.ind2coord[m_index]
            k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
            SEcirc.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
                [m_coord.real, m_coord.imag, 0.0]
            )
            Post_Dec_List = [x - 1 for x in Post_Dec_List]
            if (m_index in x_measurement_qubits_post) or (m_index in z_measurement_qubits_post): # measurement qubit belong to the post-selection set 
                Post_Dec_List.append(-1)

        head += SEcirc    # used for the circuit checking
        head.append("TICK")

    ####################

    # 2. Apply 'sampled' ZZ operation and noise on the logical Z edge: body

    body = stim.Circuit()

    # check whether the length of b string is the same as the length of logical Z operator
    # z_logical_qubits = [sc.coord2ind[p] for p in sc.z_observable]
    if len(b) != len(z_logical_qubits):
        raise ValueError("Need a b string with the length of d")

    for idLz, Lz_index in enumerate(sorted(z_logical_qubits)):
        if b[idLz] == True:
            # Apply a Pauli Z operator on the corresponding data qubit
            # body.append_operation("Z", Lz_index)
            body.append_operation("Z_ERROR", Lz_index, 1)  # add a deterministic Z error on the qubit
    
    # append the gate noise after each "ZZ(\theta)" gate
    if len(z_logical_qubits) % 2 != 0:
        raise ValueError("z logical should be even")
    if ZZtype == 0:     # the gate is done naively, append depolarizing noise
        for Lz_index_1, Lz_index_2 in zip(z_logical_qubits[::2], z_logical_qubits[1::2]):  # enumerate the pairs in z_logical_qubits
            body.append_operation("DEPOLARIZE2", [Lz_index_1, Lz_index_2], after_clifford_depolarization)
    else:  # the gate is done by dispersive coupling
        # load the fitting parameters for the ZZ rotation gate
        with open("fitting_parameters_phi_0p1.pkl", "rb") as f:
            loaded_parameters = pickle.load(f)

        # Define the Pauli operators for labeling
        pauli_operators = [
            "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"
        ]   # no identity term

        pauli_error_vector = np.zeros(15) # register for Pauli error rates

        p_dep = after_clifford_depolarization

        for id_pauli, pauli in enumerate(pauli_operators):
            Pauli2_parameter = loaded_parameters[pauli]
            shape = Pauli2_parameter["shape"]
            coefficients = Pauli2_parameter["coefficients"]
            if shape == "linear":
                pauli_error_vector[id_pauli] = coefficients[0]*p_dep
            elif shape == "quadratic":
                # pauli_error_vector[id_pauli] = coefficients[0]*(p_dep**2)
                pauli_error_vector[id_pauli] = 0
            elif shape == "hybrid":
                pauli_error_vector[id_pauli] = coefficients[0] * p_dep + coefficients[1] * p_dep**2
            else:
                print("Error!")
                exit()

        # erasure error
        Pauli2_parameter = loaded_parameters["Pfail"]
        shape = Pauli2_parameter["shape"]
        coefficients = Pauli2_parameter["coefficients"]
        if shape == "linear":
            pfail = coefficients[0]*p_dep
        elif shape == "quadratic":
            pfail = coefficients[0]*(p_dep**2)
        elif shape == "hybrid":
            pauli_error_vector[id_pauli] = coefficients[0] * p_dep + coefficients[1] * p_dep**2
        else:
            print("Error!")
            exit()

        for Lz_index_1, Lz_index_2 in zip(z_logical_qubits[::2], z_logical_qubits[1::2]):  # enumerate the pairs
            Lz_coord_1 = sc.ind2coord[Lz_index_1]
            body.append_operation("PAULI_CHANNEL_2", [Lz_index_1, Lz_index_2], pauli_error_vector)  # D0 and D2 rotation gate: Pauli errors
            body.append("HERALDED_ERASE", Lz_index_1, pfail)
            body.append("DETECTOR", stim.target_rec(-1), [Lz_coord_1.real, Lz_coord_1.imag, 0.0])
        
    body.append("TICK")

    ####################

    # 3. Post-selection and extra SE operations: tail

    tail = stim.Circuit()

    # first round: check the detector corrspondence more carefully: identify the post-selection detector
        # introduce a register for all the post-selection detectors
    # 2~(rounds_post) round: identify the post-selection detector
    # rounds_post+1 ~ rounds: normal SE circuit


    ## first round
    tail_1stSE = cycle_actions.copy()
    # tail_1stSE = cycle_actions_ideal.copy()   # for test usage
    meas_num = len(measurement_qubits)
    tail_1stSE.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    
    # counter for the erasure detection from the ZZ rotation gate
    Num_Erasure = 0
    

    # Update erausre detector to the list of post-selction detectors
    if ZZtype != 0:  # the dispersive ZZ rotation
        Num_Erasure = int(len(z_logical_qubits)/2)

        Post_Dec_List = [x - Num_Erasure for x in Post_Dec_List]
        Post_Dec_List.extend(range(-Num_Erasure, 0))
    
    # append detectors on all the measurement qubits
    for m_index in measurement_qubits:
        m_coord = sc.ind2coord[m_index]
        k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
        tail_1stSE.append_operation(
            "DETECTOR",
            [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num - Num_Erasure)],
            [m_coord.real, m_coord.imag, 0.0]
        )
        Post_Dec_List = [x - 1 for x in Post_Dec_List]
        if (m_index in x_measurement_qubits_post) or (m_index in z_measurement_qubits_post): # measurement qubit belong to the post-selection set 
            Post_Dec_List.append(-1)
        
    
    tail += tail_1stSE
    tail.append_operation('TICK')
    
    ## second to rounds_post round
    meas_num = len(measurement_qubits)
    for t in range(2, rounds_post + 1):
        tail_2ndSE = cycle_actions.copy()
        # tail_2ndSE = cycle_actions_ideal.copy()   # for test usage
        tail_2ndSE.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        # append detectors on all the measurement qubits
        for m_index in measurement_qubits:
            m_coord = sc.ind2coord[m_index]
            k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
            tail_2ndSE.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
                [m_coord.real, m_coord.imag, 0.0]
            )
            Post_Dec_List = [x - 1 for x in Post_Dec_List]
            if (m_index in x_measurement_qubits_post) or (m_index in z_measurement_qubits_post): # measurement qubit belong to the post-selection set 
                Post_Dec_List.append(-1)

        tail += tail_2ndSE
        tail.append_operation('TICK')

    ## rounds_post+1 to rounds round
    meas_num = len(measurement_qubits)
    for t in range(rounds_post + 1, rounds + 1):
        tail_3rdSE = cycle_actions.copy()
        # tail_3rdSE = cycle_actions_ideal.copy()   # for test usage
        tail_3rdSE.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        # append detectors on all the measurement qubits
        for m_index in measurement_qubits:
            m_coord = sc.ind2coord[m_index]
            k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
            tail_3rdSE.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
                [m_coord.real, m_coord.imag, 0.0]
            )
        
        Post_Dec_List = [x - meas_num for x in Post_Dec_List]

        tail += tail_3rdSE
        tail.append_operation('TICK')


    ## add ideal QEC and then logical X measurement

    # Build the ideal QEC, add detectors
    ideal_QEC = cycle_actions_ideal.copy()
    meas_num = len(measurement_qubits)

    ideal_QEC.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    for m_index in measurement_qubits:
        m_coord = sc.ind2coord[m_index]
        k = meas_num - measure_coord_to_order[m_coord] - 1
        ideal_QEC.append_operation(
            "DETECTOR",
            [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
            [m_coord.real, m_coord.imag, 0.0]
        )
    
    Post_Dec_List = [x - meas_num for x in Post_Dec_List]

    tail += ideal_QEC
    tail.append_operation('TICK')

    # Append logical X measurement at the end
    params.append_measure(tail, x_logical_qubits, "X", if_noiseless=True) 
    # add X observable
    tail.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-len(sc.x_observable),0)], 0.0)

        

    # the overall circuit:
    Proj_Cir = stim.Circuit()
    Proj_Cir = head + body + tail

    return (Proj_Cir, Post_Dec_List)


def ProjectionCircuit_Check(b: list[bool], # a length-(z_distance) boolean vector
        rounds: int,  # number of rounds for the syndrome measurement, should >= rounds_post
        rounds_post: int, # number of rounds for the post-selection in projection, should be >= 1
        rounds_prep: int, # number of rounds for state preparation (also post-selection), should be >= 1
        ZZtype: int, # 0: the naive ZZ rotation gate; 1: the dispersive ZZ rotation gate
        # Note that we do not implement the ZZ gate, but only append the corresponding noise channel.
        # distance: int = None,
        x_distance: int = None,
        z_distance: int = None,  # for the projection scheme, z_distance should be even, x_distance should be odd
        after_clifford_depolarization: float = 0.0,
        before_round_data_depolarization: float = 0.0,
        before_measure_flip_probability: float = 0.0,
        after_reset_flip_probability: float = 0.0,
        # exclude_other_basis_detectors: bool = False,
        top_left_2_body_meas_type: str = 'X',   
        ) -> tuple[stim.Circuit, list[int]] :
    """
        Generates the circuit: for a fixed sampled bitstring input b, 
        (b: list[bool] should be a length-z_distance boolean vector indicating the occurace of operationaol Z errors),
        
        Check whether the X outcome of + state preparation is correct.
        
        Meanwhile, output the detector indicies used for post-selection.

        The generated circuits can include configurable noise.

        The generated circuits include DETECTOR and OBSERVABLE_INCLUDE annotations so
        that their detection events and logical observables can be sampled.

        The generated circuits include TICK annotations to mark the progression of time.
        (E.g. so that converting them using `stimcirq.stim_circuit_to_cirq_circuit` will
        produce a `cirq.Circuit` with the intended moment structure.)

        Note that the toric_code circuits currently only include one of the two logical observables.

        Args:
            code_task: A string identifying the type of circuit to generate. Available
                code tasks are:
                    - "surface_code:rotated_memory_x"
                    - "surface_code:rotated_memory_z"
                    - "surface_code:unrotated_memory_x"
                    - "surface_code:unrotated_memory_z"
                    - "toric_code:unrotated_memory_x"
                    - "toric_code:unrotated_memory_z"
            distance: Defaults to None. The desired code distance of the generated
                circuit. The code distance is the minimum number of physical
                errors needed to cause a logical error. This parameter indirectly determines
                how many qubits the generated circuit uses.
            x_distance: Defaults to None. The desired code distance of the X
            logical
                operator in the generated circuit: the minimum number of X physical
                errors needed to cause a X logical error.
            z_distance: Defaults to None. The desired code distance of the Z
            logical
                operator in the generated circuit: the minimum number of Z physical
                errors needed to cause a Z logical error.
            rounds: How many times the measurement qubits in the generated circuit will
                be measured. Indirectly determines the duration of the generated
                circuit.
            after_clifford_depolarization: Defaults to 0. The probability (p) of
                `DEPOLARIZE1(p)` operations to add after every single-qubit Clifford
                operation and `DEPOLARIZE2(p)` operations to add after every two-qubit
                Clifford operation. The after-Clifford depolarizing operations are only
                included if this probability is not 0.
            before_round_data_depolarization: Defaults to 0. The probability (p) of
                `DEPOLARIZE1(p)` operations to apply to every data qubit at the start of
                a round of stabilizer measurements. The start-of-round depolarizing
                operations are only included if this probability is not 0.
            before_measure_flip_probability: Defaults to 0. The probability (p) of
                `X_ERROR(p)` operations applied to qubits before each measurement (X
                basis measurements use `Z_ERROR(p)` instead). The before-measurement
                flips are only included if this probability is not 0.
            after_reset_flip_probability: Defaults to 0. The probability (p) of
                `X_ERROR(p)` operations applied to qubits after each reset (X basis
                resets use `Z_ERROR(p)` instead). The after-reset flips are only
                included if this probability is not 0.
            exclude_other_basis_detectors: Defaults to False. If True, do not add
                detectors to measurement qubits that are measured in the opposite
                basis to the chosen basis of the logical observable.

        Returns:
            The generated circuit.
            The location of the detectors used for post-selection.
        """

    # initialization
    # parameter class generation
    params = CircuitGenParameters(
            rounds=rounds,
            # distance=distance,
            x_distance=x_distance,
            z_distance=z_distance,
            after_clifford_depolarization=after_clifford_depolarization,
            before_round_data_depolarization=before_round_data_depolarization,
            before_measure_flip_probability=before_measure_flip_probability,
            after_reset_flip_probability=after_reset_flip_probability,
            # exclude_other_basis_detectors=exclude_other_basis_detectors,
            # qubit_initialization_pattern = qubit_initialization_pattern,
            top_left_2_body_meas_type = top_left_2_body_meas_type
        )  # this is a tuple of all the required parameters

    # surface code namedtuple generation
    sc = surface_code_initialization(params)

    # store the indicies to lists for the future fast query
    data_qubits = [sc.coord2ind[p] for p in sc.data_coords]
    x_logical_qubits = [sc.coord2ind[p] for p in sc.x_observable]
    z_logical_qubits = [sc.coord2ind[p] for p in sc.z_observable]
    measurement_qubits = [sc.coord2ind[p] for p in sc.x_measure_coords]
    measurement_qubits += [sc.coord2ind[p] for p in sc.z_measure_coords]
    x_measurement_qubits = [sc.coord2ind[p] for p in sc.x_measure_coords]
    x_measurement_qubits_post = [sc.coord2ind[p] for p in sc.x_measure_coords_post]
    z_measurement_qubits_post = [sc.coord2ind[p] for p in sc.z_measure_coords_post]

    # for p in x_measurement_qubits_post:
    #     print(p)
    # for p in z_measurement_qubits_post:
    #     print(p)

    all_qubits: List[int] = []
    all_qubits += data_qubits + measurement_qubits

    all_qubits.sort()
    data_qubits.sort()
    measurement_qubits.sort()
    x_measurement_qubits.sort()

    # Reverse index the measurement order used for defining detectors
    # count the Z and X type measurement together: mainly for the 
    data_coord_to_order: Dict[complex, int] = {}
    measure_coord_to_order: Dict[complex, int] = {}
    for q in data_qubits:
        data_coord_to_order[sc.ind2coord[q]] = len(data_coord_to_order)   # this dictionary will provide plain-ordered sequence of all the data qubits: start from 1
    for q in measurement_qubits:
        measure_coord_to_order[sc.ind2coord[q]] = len(measure_coord_to_order)  # this dictionary will provide plain-ordered sequence of all the measurements

    # List out CNOT gate targets using given interaction orders.
    # CNOT gates are specified by the complex-valued coordinates
    cnot_targets: List[List[int]] = [[], [], [], []]  # list it by 4 time bins
    for k in range(4):
        for measure in sorted(sc.x_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + sc.x_order[k]
            if data in sc.coord2ind:
                cnot_targets[k].append(sc.coord2ind[measure])  # measure control data
                cnot_targets[k].append(sc.coord2ind[data])

        for measure in sorted(sc.z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + sc.z_order[k]
            if data in sc.coord2ind:
                cnot_targets[k].append(sc.coord2ind[data])     # data control measure
                cnot_targets[k].append(sc.coord2ind[measure])

    # define the syndrome extraction circuits that make up the surface code cycle
    cycle_actions = stim.Circuit()
    params.append_reset(cycle_actions, measurement_qubits)
    params.append_begin_round_tick(cycle_actions, data_qubits)
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits)
    for targets in cnot_targets:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets)
        # print(targets)
    cycle_actions.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits)
    cycle_actions.append_operation("TICK", [])
    # params.append_measure_reset(cycle_actions, measurement_qubits)
    params.append_measure(cycle_actions, measurement_qubits)


    '''
    def coord_to_index(coord: complex, params: dict=None) -> int:
        coord = coord - math.fmod(coord.real, 2) * 1j
        # padded_qubit_nbr = params.prep_4112_params["total_qubit_nbr"]
        padded_qubit_nbr = 0
        return int(coord.real + coord.imag * (z_distance + 0.5)) + padded_qubit_nbr
    
    cycle_actions = stim.Circuit()
    params.append_reset(cycle_actions, measurement_qubits, if_noiseless=True)
    params.append_begin_round_tick(cycle_actions, data_qubits, if_noiseless=True)
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits, if_noiseless=True)
    for targets in [cnot_targets[0]]:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets, if_noiseless=True)
        # print(targets)
    for targets in [cnot_targets[1]]:
        cycle_actions.append_operation("TICK", [])
        pairs = list(zip(targets[::2], targets[1::2]))
        for target_pair in pairs:
            # if coord_to_index(0+4j, params) in target_pair or coord_to_index(2+4j, params) in target_pair \
            #     or coord_to_index(4+4j, params) in target_pair or coord_to_index(6+4j, params) in target_pair:
            # if coord_to_index(4+4j, params) in target_pair \
            #     or coord_to_index(4+4j, params) in target_pair:
            #     params.append_unitary_2(cycle_actions, "CNOT", list(target_pair), if_noiseless=False)
            # else:
            #     params.append_unitary_2(cycle_actions, "CNOT", list(target_pair), if_noiseless=True)
            
            params.append_unitary_2(cycle_actions, "CNOT", list(target_pair), if_noiseless=False)
        # print(targets)
    for targets in [cnot_targets[2]]:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets, if_noiseless=True)
        # print(targets)
    for targets in [cnot_targets[3]]:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets, if_noiseless=True)
        # print(targets)
    cycle_actions.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits, if_noiseless=True)
    cycle_actions.append_operation("TICK", [])
    # params.append_measure_reset(cycle_actions, measurement_qubits)
    params.append_measure(cycle_actions, measurement_qubits, if_noiseless=True)
    '''

    # define the noiseless SE circuit
    cycle_actions_ideal = stim.Circuit()
    params.append_reset(cycle_actions_ideal, measurement_qubits, if_noiseless = True)
    params.append_begin_round_tick(cycle_actions_ideal, data_qubits, if_noiseless = True)
    params.append_unitary_1(cycle_actions_ideal, "H", x_measurement_qubits, if_noiseless = True)
    for targets in cnot_targets[:4]:
        cycle_actions_ideal.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions_ideal, "CNOT", targets, if_noiseless = True)
        # print(targets)
    cycle_actions_ideal.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions_ideal, "H", x_measurement_qubits, if_noiseless = True)
    cycle_actions_ideal.append_operation("TICK", [])
    params.append_measure(cycle_actions_ideal, measurement_qubits, if_noiseless = True)


    ####################

    # 1. Plus state preparation by d rounds of syndrome measurement : head

    head = stim.Circuit()
    for k, v in sorted(sc.ind2coord.items()):
        head.append_operation("QUBIT_COORDS", [k], [v.real, v.imag])

    # list of the detectors used for post-selection
    # Record the reversed location: the last detector in the whole circuit is -1 
    Post_Dec_List = []

    # Ideal initialization of the surface code: all + state for the data qubits
    params.append_reset(head, data_qubits, "X", if_noiseless = True)
    # first round of surface code syndrome check for code expansion
    head += cycle_actions
    # head += cycle_actions_ideal
    # for all the x-type syndrome check, append a detector
    for m_index in x_measurement_qubits:
        m_coord = sc.ind2coord[m_index]
        # all_touched_data_qubit_coord = [x_order_add + m_coord for x_order_add in sc.x_order  if x_order_add + m_coord in sc.data_coords] # first list all x_order + m_coord results, then select the ones in data_coords
        head.append_operation(
            "DETECTOR",
            [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[m_coord])],
            [m_coord.real, m_coord.imag, 0.0] # this provide the coordinate to the detector
            )
            # print("X detectors at: ", m_coord)
        Post_Dec_List = [x - 1 for x in Post_Dec_List]        
        if (m_index in x_measurement_qubits_post): # measurement qubit belong to the post-selection set 
            Post_Dec_List.append(-1)

    

    # Build the repeated body of the circuit, including the detectors comparing to previous cycles.
    # rounds_prep = 1
    for t in range(rounds_prep):
        SEcirc = cycle_actions.copy()
        # SEcirc = cycle_actions_ideal.copy()
        meas_num = len(measurement_qubits)
        SEcirc.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
        # append detectors on all the measurement qubits
        for m_index in measurement_qubits:
            m_coord = sc.ind2coord[m_index]
            k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
            SEcirc.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
                [m_coord.real, m_coord.imag, 0.0]
            )
            Post_Dec_List = [x - 1 for x in Post_Dec_List]
            if (m_index in x_measurement_qubits_post) or (m_index in z_measurement_qubits_post): # measurement qubit belong to the post-selection set 
                Post_Dec_List.append(-1)

        head += SEcirc    # used for the circuit checking
        head.append('TICK')


    ####################

    # 2. Apply 'sampled' ZZ operation and noise on the logical Z edge: body

    body = stim.Circuit()

    # check whether the length of b string is the same as the length of logical Z operator
    # z_logical_qubits = [sc.coord2ind[p] for p in sc.z_observable]
    if len(b) != len(z_logical_qubits):
        raise ValueError("Need a b string with the length of d")

    for idLz, Lz_index in enumerate(sorted(z_logical_qubits)):
        if b[idLz] == True:
            # Apply a Pauli Z operator on the corresponding data qubit
            # body.append_operation("Z", Lz_index)
            body.append_operation("Z_ERROR", Lz_index, 1)  # add a deterministic Z error on the qubit
    
    # append the gate noise after each "ZZ(\theta)" gate
    if len(z_logical_qubits) % 2 != 0:
        raise ValueError("z logical should be even")
    if ZZtype == 0:     # the gate is done naively, append depolarizing noise
        # aaa = 1
        for Lz_index_1, Lz_index_2 in zip(z_logical_qubits[::2], z_logical_qubits[1::2]):  # enumerate the pairs in z_logical_qubits
            body.append_operation("DEPOLARIZE2", [Lz_index_1, Lz_index_2], after_clifford_depolarization)
    else:  # the gate is done by dispersive coupling
        # load the fitting parameters for the ZZ rotation gate
        with open("fitting_parameters_phi_0p1.pkl", "rb") as f:
            loaded_parameters = pickle.load(f)

        # Define the Pauli operators for labeling
        pauli_operators = [
            "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"
        ]   # no identity term

        pauli_error_vector = np.zeros(15) # register for Pauli error rates

        p_dep = after_clifford_depolarization

        for id_pauli, pauli in enumerate(pauli_operators):
            Pauli2_parameter = loaded_parameters[pauli]
            shape = Pauli2_parameter["shape"]
            coefficients = Pauli2_parameter["coefficients"]
            if shape == "linear":
                pauli_error_vector[id_pauli] = coefficients[0]*p_dep
            elif shape == "quadratic":
                pauli_error_vector[id_pauli] = coefficients[0]*(p_dep**2)
            elif shape == "hybrid":
                pauli_error_vector[id_pauli] = coefficients[0] * p_dep + coefficients[1] * p_dep**2
            else:
                print("Error!")
                exit()

        # erasure error
        Pauli2_parameter = loaded_parameters["Pfail"]
        shape = Pauli2_parameter["shape"]
        coefficients = Pauli2_parameter["coefficients"]
        if shape == "linear":
            pfail = coefficients[0]*p_dep
        elif shape == "quadratic":
            pfail = coefficients[0]*(p_dep**2)
        elif shape == "hybrid":
            pauli_error_vector[id_pauli] = coefficients[0] * p_dep + coefficients[1] * p_dep**2
        else:
            print("Error!")
            exit()

        for Lz_index_1, Lz_index_2 in zip(z_logical_qubits[::2], z_logical_qubits[1::2]):  # enumerate the pairs
            Lz_coord_1 = sc.ind2coord[Lz_index_1]
            body.append_operation("PAULI_CHANNEL_2", [Lz_index_1, Lz_index_2], pauli_error_vector)  # D0 and D2 rotation gate: Pauli errors
            body.append("HERALDED_ERASE", Lz_index_1, pfail)
            body.append("DETECTOR", stim.target_rec(-1), [Lz_coord_1.real, Lz_coord_1.imag, 0.0])
        
    body.append("TICK")

    ####################

    # 3. Post-selection and extra SE operations: tail

    tail = stim.Circuit()

    # first round: check the detector corrspondence more carefully: identify the post-selection detector
        # introduce a register for all the post-selection detectors
    # 2~(rounds_post) round: identify the post-selection detector
    # rounds_post+1 ~ rounds: normal SE circuit


    ## first round
    tail_1stSE = cycle_actions.copy()
    # tail_1stSE = cycle_actions_ideal.copy()   # for test usage
    meas_num = len(measurement_qubits)
    tail_1stSE.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    
    # counter for the erasure detection from the ZZ rotation gate
    Num_Erasure = 0
    

    # Update erausre detector to the list of post-selction detectors
    if ZZtype != 0:  # the dispersive ZZ rotation
        Num_Erasure = int(len(z_logical_qubits)/2)

        Post_Dec_List = [x - Num_Erasure for x in Post_Dec_List]
        Post_Dec_List.extend(range(-Num_Erasure, 0))
    
    # append detectors on all the measurement qubits
    for m_index in measurement_qubits:
        m_coord = sc.ind2coord[m_index]
        k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
        tail_1stSE.append_operation(
            "DETECTOR",
            [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num - Num_Erasure)],
            [m_coord.real, m_coord.imag, 0.0]
        )
        Post_Dec_List = [x - 1 for x in Post_Dec_List]
        if (m_index in x_measurement_qubits_post) or (m_index in z_measurement_qubits_post): # measurement qubit belong to the post-selection set 
            Post_Dec_List.append(-1)
        
    
    tail += tail_1stSE
    tail.append_operation('TICK')
    
    ## second to rounds_post round
    meas_num = len(measurement_qubits)
    for t in range(2, rounds_post + 1):
        tail_2ndSE = cycle_actions.copy()
        # tail_2ndSE = cycle_actions_ideal.copy()   # for test usage
        tail_2ndSE.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        # append detectors on all the measurement qubits
        for m_index in measurement_qubits:
            m_coord = sc.ind2coord[m_index]
            k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
            tail_2ndSE.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
                [m_coord.real, m_coord.imag, 0.0]
            )
            Post_Dec_List = [x - 1 for x in Post_Dec_List]
            if (m_index in x_measurement_qubits_post) or (m_index in z_measurement_qubits_post): # measurement qubit belong to the post-selection set 
                Post_Dec_List.append(-1)

        tail += tail_2ndSE
        tail.append_operation('TICK')

    ## rounds_post+1 to rounds round
    meas_num = len(measurement_qubits)
    for t in range(rounds_post + 1, rounds + 1):
        tail_3rdSE = cycle_actions.copy()
        # tail_3rdSE = cycle_actions_ideal.copy()   # for test usage
        tail_3rdSE.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        # append detectors on all the measurement qubits
        for m_index in measurement_qubits:
            m_coord = sc.ind2coord[m_index]
            k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
            tail_3rdSE.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
                [m_coord.real, m_coord.imag, 0.0]
            )
        
        Post_Dec_List = [x - meas_num for x in Post_Dec_List]

        tail += tail_3rdSE
        tail.append_operation('TICK')


    ## add ideal QEC and then logical X measurement

    # Build the ideal QEC, add detectors
    ideal_QEC = cycle_actions_ideal.copy()
    meas_num = len(measurement_qubits)

    ideal_QEC.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    for m_index in measurement_qubits:
        m_coord = sc.ind2coord[m_index]
        k = meas_num - measure_coord_to_order[m_coord] - 1
        ideal_QEC.append_operation(
            "DETECTOR",
            [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
            [m_coord.real, m_coord.imag, 0.0]
        )
    
    Post_Dec_List = [x - meas_num for x in Post_Dec_List]

    tail += ideal_QEC
    tail.append_operation('TICK')

    # Append logical X measurement at the end
    params.append_measure(tail, x_logical_qubits, "X", if_noiseless=True) 
    # add X observable
    tail.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-len(sc.x_observable),0)], 0.0)

        

    # the overall circuit:
    Proj_Cir = stim.Circuit()
    Proj_Cir = head + body + tail

    return (Proj_Cir, Post_Dec_List)

def ProjectionCircuit_Check_naive(b: list[bool], # a length-(z_distance) boolean vector
        rounds: int,  # number of rounds for the syndrome measurement, should >= rounds_post
        rounds_post: int, # number of rounds for the post-selection in projection, should be >= 1
        rounds_prep: int, # number of rounds for state preparation (also post-selection), should be >= 1
        ZZtype: int, # 0: the naive ZZ rotation gate; 1: the dispersive ZZ rotation gate
        # Note that we do not implement the ZZ gate, but only append the corresponding noise channel.
        # distance: int = None,
        x_distance: int = None,
        z_distance: int = None,  # for the projection scheme, z_distance should be even, x_distance should be odd
        after_clifford_depolarization: float = 0.0,
        before_round_data_depolarization: float = 0.0,
        before_measure_flip_probability: float = 0.0,
        after_reset_flip_probability: float = 0.0,
        # exclude_other_basis_detectors: bool = False,
        top_left_2_body_meas_type: str = 'X',   
        ) -> tuple[stim.Circuit, list[int]] :
    """
        Generates the circuit: for a fixed sampled bitstring input b, 
        (b: list[bool] should be a length-z_distance boolean vector indicating the occurace of operationaol Z errors),
        
        Check whether the X outcome of + state preparation is correct.
        
        Meanwhile, output the detector indicies used for post-selection.

        The generated circuits can include configurable noise.

        The generated circuits include DETECTOR and OBSERVABLE_INCLUDE annotations so
        that their detection events and logical observables can be sampled.

        The generated circuits include TICK annotations to mark the progression of time.
        (E.g. so that converting them using `stimcirq.stim_circuit_to_cirq_circuit` will
        produce a `cirq.Circuit` with the intended moment structure.)

        Note that the toric_code circuits currently only include one of the two logical observables.

        Args:
            code_task: A string identifying the type of circuit to generate. Available
                code tasks are:
                    - "surface_code:rotated_memory_x"
                    - "surface_code:rotated_memory_z"
                    - "surface_code:unrotated_memory_x"
                    - "surface_code:unrotated_memory_z"
                    - "toric_code:unrotated_memory_x"
                    - "toric_code:unrotated_memory_z"
            distance: Defaults to None. The desired code distance of the generated
                circuit. The code distance is the minimum number of physical
                errors needed to cause a logical error. This parameter indirectly determines
                how many qubits the generated circuit uses.
            x_distance: Defaults to None. The desired code distance of the X
            logical
                operator in the generated circuit: the minimum number of X physical
                errors needed to cause a X logical error.
            z_distance: Defaults to None. The desired code distance of the Z
            logical
                operator in the generated circuit: the minimum number of Z physical
                errors needed to cause a Z logical error.
            rounds: How many times the measurement qubits in the generated circuit will
                be measured. Indirectly determines the duration of the generated
                circuit.
            after_clifford_depolarization: Defaults to 0. The probability (p) of
                `DEPOLARIZE1(p)` operations to add after every single-qubit Clifford
                operation and `DEPOLARIZE2(p)` operations to add after every two-qubit
                Clifford operation. The after-Clifford depolarizing operations are only
                included if this probability is not 0.
            before_round_data_depolarization: Defaults to 0. The probability (p) of
                `DEPOLARIZE1(p)` operations to apply to every data qubit at the start of
                a round of stabilizer measurements. The start-of-round depolarizing
                operations are only included if this probability is not 0.
            before_measure_flip_probability: Defaults to 0. The probability (p) of
                `X_ERROR(p)` operations applied to qubits before each measurement (X
                basis measurements use `Z_ERROR(p)` instead). The before-measurement
                flips are only included if this probability is not 0.
            after_reset_flip_probability: Defaults to 0. The probability (p) of
                `X_ERROR(p)` operations applied to qubits after each reset (X basis
                resets use `Z_ERROR(p)` instead). The after-reset flips are only
                included if this probability is not 0.
            exclude_other_basis_detectors: Defaults to False. If True, do not add
                detectors to measurement qubits that are measured in the opposite
                basis to the chosen basis of the logical observable.

        Returns:
            The generated circuit.
            The location of the detectors used for post-selection.

        The difference between: `Projection_Check` and `Projection_Check_naive` is that, 
            when ZZtype = 1, we use different design for the ZZ rotation gate.
            for `Projection_Check`, we use the ancillary-based ZZ rotation gate
            for `Projection_Check_naive`, we use the ancillary-free ZZ rotation gate.
        """

    # initialization
    # parameter class generation
    params = CircuitGenParameters(
            rounds=rounds,
            # distance=distance,
            x_distance=x_distance,
            z_distance=z_distance,
            after_clifford_depolarization=after_clifford_depolarization,
            before_round_data_depolarization=before_round_data_depolarization,
            before_measure_flip_probability=before_measure_flip_probability,
            after_reset_flip_probability=after_reset_flip_probability,
            # exclude_other_basis_detectors=exclude_other_basis_detectors,
            # qubit_initialization_pattern = qubit_initialization_pattern,
            top_left_2_body_meas_type = top_left_2_body_meas_type
        )  # this is a tuple of all the required parameters

    # surface code namedtuple generation
    sc = surface_code_initialization(params)

    # store the indicies to lists for the future fast query
    data_qubits = [sc.coord2ind[p] for p in sc.data_coords]
    x_logical_qubits = [sc.coord2ind[p] for p in sc.x_observable]
    z_logical_qubits = [sc.coord2ind[p] for p in sc.z_observable]
    measurement_qubits = [sc.coord2ind[p] for p in sc.x_measure_coords]
    measurement_qubits += [sc.coord2ind[p] for p in sc.z_measure_coords]
    x_measurement_qubits = [sc.coord2ind[p] for p in sc.x_measure_coords]
    x_measurement_qubits_post = [sc.coord2ind[p] for p in sc.x_measure_coords_post]
    z_measurement_qubits_post = [sc.coord2ind[p] for p in sc.z_measure_coords_post]

    # for p in x_measurement_qubits_post:
    #     print(p)
    # for p in z_measurement_qubits_post:
    #     print(p)

    all_qubits: List[int] = []
    all_qubits += data_qubits + measurement_qubits

    all_qubits.sort()
    data_qubits.sort()
    measurement_qubits.sort()
    x_measurement_qubits.sort()

    # Reverse index the measurement order used for defining detectors
    # count the Z and X type measurement together: mainly for the 
    data_coord_to_order: Dict[complex, int] = {}
    measure_coord_to_order: Dict[complex, int] = {}
    for q in data_qubits:
        data_coord_to_order[sc.ind2coord[q]] = len(data_coord_to_order)   # this dictionary will provide plain-ordered sequence of all the data qubits: start from 1
    for q in measurement_qubits:
        measure_coord_to_order[sc.ind2coord[q]] = len(measure_coord_to_order)  # this dictionary will provide plain-ordered sequence of all the measurements

    # List out CNOT gate targets using given interaction orders.
    # CNOT gates are specified by the complex-valued coordinates
    cnot_targets: List[List[int]] = [[], [], [], []]  # list it by 4 time bins
    for k in range(4):
        for measure in sorted(sc.x_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + sc.x_order[k]
            if data in sc.coord2ind:
                cnot_targets[k].append(sc.coord2ind[measure])  # measure control data
                cnot_targets[k].append(sc.coord2ind[data])

        for measure in sorted(sc.z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + sc.z_order[k]
            if data in sc.coord2ind:
                cnot_targets[k].append(sc.coord2ind[data])     # data control measure
                cnot_targets[k].append(sc.coord2ind[measure])

    # define the syndrome extraction circuits that make up the surface code cycle
    cycle_actions = stim.Circuit()
    params.append_reset(cycle_actions, measurement_qubits)
    params.append_begin_round_tick(cycle_actions, data_qubits)
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits)
    for targets in cnot_targets:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets)
        # print(targets)
    cycle_actions.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits)
    cycle_actions.append_operation("TICK", [])
    # params.append_measure_reset(cycle_actions, measurement_qubits)
    params.append_measure(cycle_actions, measurement_qubits)


    '''
    def coord_to_index(coord: complex, params: dict=None) -> int:
        coord = coord - math.fmod(coord.real, 2) * 1j
        # padded_qubit_nbr = params.prep_4112_params["total_qubit_nbr"]
        padded_qubit_nbr = 0
        return int(coord.real + coord.imag * (z_distance + 0.5)) + padded_qubit_nbr
    
    cycle_actions = stim.Circuit()
    params.append_reset(cycle_actions, measurement_qubits, if_noiseless=True)
    params.append_begin_round_tick(cycle_actions, data_qubits, if_noiseless=True)
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits, if_noiseless=True)
    for targets in [cnot_targets[0]]:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets, if_noiseless=True)
        # print(targets)
    for targets in [cnot_targets[1]]:
        cycle_actions.append_operation("TICK", [])
        pairs = list(zip(targets[::2], targets[1::2]))
        for target_pair in pairs:
            # if coord_to_index(0+4j, params) in target_pair or coord_to_index(2+4j, params) in target_pair \
            #     or coord_to_index(4+4j, params) in target_pair or coord_to_index(6+4j, params) in target_pair:
            # if coord_to_index(4+4j, params) in target_pair \
            #     or coord_to_index(4+4j, params) in target_pair:
            #     params.append_unitary_2(cycle_actions, "CNOT", list(target_pair), if_noiseless=False)
            # else:
            #     params.append_unitary_2(cycle_actions, "CNOT", list(target_pair), if_noiseless=True)
            
            params.append_unitary_2(cycle_actions, "CNOT", list(target_pair), if_noiseless=False)
        # print(targets)
    for targets in [cnot_targets[2]]:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets, if_noiseless=True)
        # print(targets)
    for targets in [cnot_targets[3]]:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets, if_noiseless=True)
        # print(targets)
    cycle_actions.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits, if_noiseless=True)
    cycle_actions.append_operation("TICK", [])
    # params.append_measure_reset(cycle_actions, measurement_qubits)
    params.append_measure(cycle_actions, measurement_qubits, if_noiseless=True)
    '''

    # define the noiseless SE circuit
    cycle_actions_ideal = stim.Circuit()
    params.append_reset(cycle_actions_ideal, measurement_qubits, if_noiseless = True)
    params.append_begin_round_tick(cycle_actions_ideal, data_qubits, if_noiseless = True)
    params.append_unitary_1(cycle_actions_ideal, "H", x_measurement_qubits, if_noiseless = True)
    for targets in cnot_targets[:4]:
        cycle_actions_ideal.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions_ideal, "CNOT", targets, if_noiseless = True)
        # print(targets)
    cycle_actions_ideal.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions_ideal, "H", x_measurement_qubits, if_noiseless = True)
    cycle_actions_ideal.append_operation("TICK", [])
    params.append_measure(cycle_actions_ideal, measurement_qubits, if_noiseless = True)


    ####################

    # 1. Plus state preparation by d rounds of syndrome measurement : head

    head = stim.Circuit()
    for k, v in sorted(sc.ind2coord.items()):
        head.append_operation("QUBIT_COORDS", [k], [v.real, v.imag])

    # list of the detectors used for post-selection
    # Record the reversed location: the last detector in the whole circuit is -1 
    Post_Dec_List = []

    # Ideal initialization of the surface code: all + state for the data qubits
    params.append_reset(head, data_qubits, "X", if_noiseless = True)
    # first round of surface code syndrome check for code expansion
    head += cycle_actions
    # head += cycle_actions_ideal
    # for all the x-type syndrome check, append a detector
    for m_index in x_measurement_qubits:
        m_coord = sc.ind2coord[m_index]
        # all_touched_data_qubit_coord = [x_order_add + m_coord for x_order_add in sc.x_order  if x_order_add + m_coord in sc.data_coords] # first list all x_order + m_coord results, then select the ones in data_coords
        head.append_operation(
            "DETECTOR",
            [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[m_coord])],
            [m_coord.real, m_coord.imag, 0.0] # this provide the coordinate to the detector
            )
            # print("X detectors at: ", m_coord)
        Post_Dec_List = [x - 1 for x in Post_Dec_List]        
        if (m_index in x_measurement_qubits_post): # measurement qubit belong to the post-selection set 
            Post_Dec_List.append(-1)

    

    # Build the repeated body of the circuit, including the detectors comparing to previous cycles.
    # rounds_prep = 1
    for t in range(rounds_prep):
        SEcirc = cycle_actions.copy()
        # SEcirc = cycle_actions_ideal.copy()
        meas_num = len(measurement_qubits)
        SEcirc.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
        # append detectors on all the measurement qubits
        for m_index in measurement_qubits:
            m_coord = sc.ind2coord[m_index]
            k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
            SEcirc.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
                [m_coord.real, m_coord.imag, 0.0]
            )
            Post_Dec_List = [x - 1 for x in Post_Dec_List]
            if (m_index in x_measurement_qubits_post) or (m_index in z_measurement_qubits_post): # measurement qubit belong to the post-selection set 
                Post_Dec_List.append(-1)

        head += SEcirc    # used for the circuit checking
        head.append('TICK')


    ####################

    # 2. Apply 'sampled' ZZ operation and noise on the logical Z edge: body

    body = stim.Circuit()

    # check whether the length of b string is the same as the length of logical Z operator
    # z_logical_qubits = [sc.coord2ind[p] for p in sc.z_observable]
    if len(b) != len(z_logical_qubits):
        raise ValueError("Need a b string with the length of d")

    for idLz, Lz_index in enumerate(sorted(z_logical_qubits)):
        if b[idLz] == True:
            # Apply a Pauli Z operator on the corresponding data qubit
            # body.append_operation("Z", Lz_index)
            body.append_operation("Z_ERROR", Lz_index, 1)  # add a deterministic Z error on the qubit
    
    # append the gate noise after each "ZZ(\theta)" gate
    if len(z_logical_qubits) % 2 != 0:
        raise ValueError("z logical should be even")
    if ZZtype == 0:     # the gate is done naively, append depolarizing noise
        for Lz_index_1, Lz_index_2 in zip(z_logical_qubits[::2], z_logical_qubits[1::2]):  # enumerate the pairs in z_logical_qubits
            body.append_operation("DEPOLARIZE2", [Lz_index_1, Lz_index_2], after_clifford_depolarization)
    else:  # the gate is done by dispersive coupling
        # load the fitting parameters for the ZZ rotation gate
        with open("fitting_parameters_phi_0p1_naive.pkl", "rb") as f:
            loaded_parameters = pickle.load(f)

        # Define the Pauli operators for labeling
        pauli_operators = [
            "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"
        ]   # no identity term

        pauli_error_vector = np.zeros(15) # register for Pauli error rates

        p_dep = after_clifford_depolarization

        for id_pauli, pauli in enumerate(pauli_operators):
            Pauli2_parameter = loaded_parameters[pauli]
            shape = Pauli2_parameter["shape"]
            coefficients = Pauli2_parameter["coefficients"]
            if shape == "linear":
                pauli_error_vector[id_pauli] = coefficients[0]*p_dep
            elif shape == "quadratic":
                pauli_error_vector[id_pauli] = coefficients[0]*(p_dep**2)
            elif shape == "hybrid":
                pauli_error_vector[id_pauli] = coefficients[0] * p_dep + coefficients[1] * p_dep**2
            else:
                print("Error!")
                exit()
        for Lz_index_1, Lz_index_2 in zip(z_logical_qubits[::2], z_logical_qubits[1::2]):  # enumerate the pairs in z_logical_qubits
            body.append_operation("PAULI_CHANNEL_2", [Lz_index_1, Lz_index_2], pauli_error_vector)  # D0 and D2 rotation gate: Pauli errors

        # # erasure error
        # Pauli2_parameter = loaded_parameters["Pfail"]
        # shape = Pauli2_parameter["shape"]
        # coefficients = Pauli2_parameter["coefficients"]
        # if shape == "linear":
        #     pfail = coefficients[0]*p_dep
        # elif shape == "quadratic":
        #     pfail = coefficients[0]*(p_dep**2)
        # elif shape == "hybrid":
        #     pauli_error_vector[id_pauli] = coefficients[0] * p_dep + coefficients[1] * p_dep**2
        # else:
        #     print("Error!")
        #     exit()

        # for Lz_index_1, Lz_index_2 in zip(z_logical_qubits[::2], z_logical_qubits[1::2]):  # enumerate the pairs
        #     Lz_coord_1 = sc.ind2coord[Lz_index_1]
        #     body.append_operation("PAULI_CHANNEL_2", [Lz_index_1, Lz_index_2], pauli_error_vector)  # D0 and D2 rotation gate: Pauli errors
        #     body.append("HERALDED_ERASE", Lz_index_1, pfail)
        #     body.append("DETECTOR", stim.target_rec(-1), [Lz_coord_1.real, Lz_coord_1.imag, 0.0])
        
    body.append("TICK")

    ####################

    # 3. Post-selection and extra SE operations: tail

    tail = stim.Circuit()

    # first round: check the detector corrspondence more carefully: identify the post-selection detector
        # introduce a register for all the post-selection detectors
    # 2~(rounds_post) round: identify the post-selection detector
    # rounds_post+1 ~ rounds: normal SE circuit


    ## first round
    tail_1stSE = cycle_actions.copy()
    # tail_1stSE = cycle_actions_ideal.copy()   # for test usage
    meas_num = len(measurement_qubits)
    tail_1stSE.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    
    # counter for the erasure detection from the ZZ rotation gate
    Num_Erasure = 0

    # # Update erausre detector to the list of post-selction detectors
    # if ZZtype != 0:  # the dispersive ZZ rotation
    #     Num_Erasure = int(len(z_logical_qubits)/2)

    #     Post_Dec_List = [x - Num_Erasure for x in Post_Dec_List]
    #     Post_Dec_List.extend(range(-Num_Erasure, 0))
    
    # append detectors on all the measurement qubits
    for m_index in measurement_qubits:
        m_coord = sc.ind2coord[m_index]
        k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
        tail_1stSE.append_operation(
            "DETECTOR",
            [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num - Num_Erasure)],
            [m_coord.real, m_coord.imag, 0.0]
        )
        Post_Dec_List = [x - 1 for x in Post_Dec_List]
        if (m_index in x_measurement_qubits_post) or (m_index in z_measurement_qubits_post): # measurement qubit belong to the post-selection set 
            Post_Dec_List.append(-1)
        
    
    tail += tail_1stSE
    tail.append_operation('TICK')
    
    ## second to rounds_post round
    meas_num = len(measurement_qubits)
    for t in range(2, rounds_post + 1):
        tail_2ndSE = cycle_actions.copy()
        # tail_2ndSE = cycle_actions_ideal.copy()   # for test usage
        tail_2ndSE.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        # append detectors on all the measurement qubits
        for m_index in measurement_qubits:
            m_coord = sc.ind2coord[m_index]
            k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
            tail_2ndSE.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
                [m_coord.real, m_coord.imag, 0.0]
            )
            Post_Dec_List = [x - 1 for x in Post_Dec_List]
            if (m_index in x_measurement_qubits_post) or (m_index in z_measurement_qubits_post): # measurement qubit belong to the post-selection set 
                Post_Dec_List.append(-1)

        tail += tail_2ndSE
        tail.append_operation('TICK')

    ## rounds_post+1 to rounds round
    meas_num = len(measurement_qubits)
    for t in range(rounds_post + 1, rounds + 1):
        tail_3rdSE = cycle_actions.copy()
        # tail_3rdSE = cycle_actions_ideal.copy()   # for test usage
        tail_3rdSE.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        # append detectors on all the measurement qubits
        for m_index in measurement_qubits:
            m_coord = sc.ind2coord[m_index]
            k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
            tail_3rdSE.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
                [m_coord.real, m_coord.imag, 0.0]
            )
        
        Post_Dec_List = [x - meas_num for x in Post_Dec_List]

        tail += tail_3rdSE
        tail.append_operation('TICK')


    ## add ideal QEC and then logical X measurement

    # Build the ideal QEC, add detectors
    ideal_QEC = cycle_actions_ideal.copy()
    meas_num = len(measurement_qubits)

    ideal_QEC.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    for m_index in measurement_qubits:
        m_coord = sc.ind2coord[m_index]
        k = meas_num - measure_coord_to_order[m_coord] - 1
        ideal_QEC.append_operation(
            "DETECTOR",
            [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
            [m_coord.real, m_coord.imag, 0.0]
        )
    
    Post_Dec_List = [x - meas_num for x in Post_Dec_List]

    tail += ideal_QEC
    tail.append_operation('TICK')

    # Append logical X measurement at the end
    params.append_measure(tail, x_logical_qubits, "X", if_noiseless=True) 
    # add X observable
    tail.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-len(sc.x_observable),0)], 0.0)

        

    # the overall circuit:
    Proj_Cir = stim.Circuit()
    Proj_Cir = head + body + tail

    return (Proj_Cir, Post_Dec_List)



def ProjectionCircuit_Check_naive_m3(b: list[bool], # a length-(z_distance) boolean vector with the shape like 000111000
        rounds: int,  # number of rounds for the syndrome measurement, should >= rounds_post
        rounds_post: int, # number of rounds for the post-selection in projection, should be >= 1
        rounds_prep: int, # number of rounds for state preparation (also post-selection), should be >= 1
        ZZtype: int, # 0: the naive ZZ rotation gate; 1: the dispersive ZZ rotation gate
        # Note that we do not implement the ZZ gate, but only append the corresponding noise channel.
        # distance: int = None,
        x_distance: int = None,
        z_distance: int = None,  # for the projection scheme, z_distance should be even, x_distance should be odd
        after_clifford_depolarization: float = 0.0,
        before_round_data_depolarization: float = 0.0,
        before_measure_flip_probability: float = 0.0,
        after_reset_flip_probability: float = 0.0,
        # exclude_other_basis_detectors: bool = False,
        top_left_2_body_meas_type: str = 'X',   
        ) -> tuple[stim.Circuit, list[int]] :
    """
        Generates the circuit: for a fixed sampled bitstring input b, 
        (b: list[bool] should be a length-z_distance boolean vector indicating the occurace of operationaol Z errors),
        
        Check whether the X outcome of + state preparation is correct.
        
        Meanwhile, output the detector indicies used for post-selection.

        The generated circuits can include configurable noise.

        The generated circuits include DETECTOR and OBSERVABLE_INCLUDE annotations so
        that their detection events and logical observables can be sampled.

        The generated circuits include TICK annotations to mark the progression of time.
        (E.g. so that converting them using `stimcirq.stim_circuit_to_cirq_circuit` will
        produce a `cirq.Circuit` with the intended moment structure.)

        Note that the toric_code circuits currently only include one of the two logical observables.

        Args:
            code_task: A string identifying the type of circuit to generate. Available
                code tasks are:
                    - "surface_code:rotated_memory_x"
                    - "surface_code:rotated_memory_z"
                    - "surface_code:unrotated_memory_x"
                    - "surface_code:unrotated_memory_z"
                    - "toric_code:unrotated_memory_x"
                    - "toric_code:unrotated_memory_z"
            distance: Defaults to None. The desired code distance of the generated
                circuit. The code distance is the minimum number of physical
                errors needed to cause a logical error. This parameter indirectly determines
                how many qubits the generated circuit uses.
            x_distance: Defaults to None. The desired code distance of the X
            logical
                operator in the generated circuit: the minimum number of X physical
                errors needed to cause a X logical error.
            z_distance: Defaults to None. The desired code distance of the Z
            logical
                operator in the generated circuit: the minimum number of Z physical
                errors needed to cause a Z logical error.
            rounds: How many times the measurement qubits in the generated circuit will
                be measured. Indirectly determines the duration of the generated
                circuit.
            after_clifford_depolarization: Defaults to 0. The probability (p) of
                `DEPOLARIZE1(p)` operations to add after every single-qubit Clifford
                operation and `DEPOLARIZE2(p)` operations to add after every two-qubit
                Clifford operation. The after-Clifford depolarizing operations are only
                included if this probability is not 0.
            before_round_data_depolarization: Defaults to 0. The probability (p) of
                `DEPOLARIZE1(p)` operations to apply to every data qubit at the start of
                a round of stabilizer measurements. The start-of-round depolarizing
                operations are only included if this probability is not 0.
            before_measure_flip_probability: Defaults to 0. The probability (p) of
                `X_ERROR(p)` operations applied to qubits before each measurement (X
                basis measurements use `Z_ERROR(p)` instead). The before-measurement
                flips are only included if this probability is not 0.
            after_reset_flip_probability: Defaults to 0. The probability (p) of
                `X_ERROR(p)` operations applied to qubits after each reset (X basis
                resets use `Z_ERROR(p)` instead). The after-reset flips are only
                included if this probability is not 0.
            exclude_other_basis_detectors: Defaults to False. If True, do not add
                detectors to measurement qubits that are measured in the opposite
                basis to the chosen basis of the logical observable.

        Returns:
            The generated circuit.
            The location of the detectors used for post-selection.

        The difference between: `Projection_Check` and `Projection_Check_naive` is that, 
            when ZZtype = 1, we use different design for the ZZ rotation gate.
            for `Projection_Check`, we use the ancillary-based ZZ rotation gate
            for `Projection_Check_naive`, we use the ancillary-free ZZ rotation gate.
        """

    # initialization
    # parameter class generation
    params = CircuitGenParameters(
            rounds=rounds,
            # distance=distance,
            x_distance=x_distance,
            z_distance=z_distance,
            after_clifford_depolarization=after_clifford_depolarization,
            before_round_data_depolarization=before_round_data_depolarization,
            before_measure_flip_probability=before_measure_flip_probability,
            after_reset_flip_probability=after_reset_flip_probability,
            # exclude_other_basis_detectors=exclude_other_basis_detectors,
            # qubit_initialization_pattern = qubit_initialization_pattern,
            top_left_2_body_meas_type = top_left_2_body_meas_type
        )  # this is a tuple of all the required parameters

    # surface code namedtuple generation
    sc = surface_code_initialization(params)

    # store the indicies to lists for the future fast query
    data_qubits = [sc.coord2ind[p] for p in sc.data_coords]
    x_logical_qubits = [sc.coord2ind[p] for p in sc.x_observable]
    z_logical_qubits = [sc.coord2ind[p] for p in sc.z_observable]
    measurement_qubits = [sc.coord2ind[p] for p in sc.x_measure_coords]
    measurement_qubits += [sc.coord2ind[p] for p in sc.z_measure_coords]
    x_measurement_qubits = [sc.coord2ind[p] for p in sc.x_measure_coords]
    x_measurement_qubits_post = [sc.coord2ind[p] for p in sc.x_measure_coords_post]
    z_measurement_qubits_post = [sc.coord2ind[p] for p in sc.z_measure_coords_post]

    # for p in x_measurement_qubits_post:
    #     print(p)
    # for p in z_measurement_qubits_post:
    #     print(p)

    all_qubits: List[int] = []
    all_qubits += data_qubits + measurement_qubits

    all_qubits.sort()
    data_qubits.sort()
    measurement_qubits.sort()
    x_measurement_qubits.sort()

    # Reverse index the measurement order used for defining detectors
    # count the Z and X type measurement together: mainly for the 
    data_coord_to_order: Dict[complex, int] = {}
    measure_coord_to_order: Dict[complex, int] = {}
    for q in data_qubits:
        data_coord_to_order[sc.ind2coord[q]] = len(data_coord_to_order)   # this dictionary will provide plain-ordered sequence of all the data qubits: start from 1
    for q in measurement_qubits:
        measure_coord_to_order[sc.ind2coord[q]] = len(measure_coord_to_order)  # this dictionary will provide plain-ordered sequence of all the measurements

    # List out CNOT gate targets using given interaction orders.
    # CNOT gates are specified by the complex-valued coordinates
    cnot_targets: List[List[int]] = [[], [], [], []]  # list it by 4 time bins
    for k in range(4):
        for measure in sorted(sc.x_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + sc.x_order[k]
            if data in sc.coord2ind:
                cnot_targets[k].append(sc.coord2ind[measure])  # measure control data
                cnot_targets[k].append(sc.coord2ind[data])

        for measure in sorted(sc.z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + sc.z_order[k]
            if data in sc.coord2ind:
                cnot_targets[k].append(sc.coord2ind[data])     # data control measure
                cnot_targets[k].append(sc.coord2ind[measure])

    # define the syndrome extraction circuits that make up the surface code cycle
    cycle_actions = stim.Circuit()
    params.append_reset(cycle_actions, measurement_qubits)
    params.append_begin_round_tick(cycle_actions, data_qubits)
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits)
    for targets in cnot_targets:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets)
        # print(targets)
    cycle_actions.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits)
    cycle_actions.append_operation("TICK", [])
    # params.append_measure_reset(cycle_actions, measurement_qubits)
    params.append_measure(cycle_actions, measurement_qubits)


    '''
    def coord_to_index(coord: complex, params: dict=None) -> int:
        coord = coord - math.fmod(coord.real, 2) * 1j
        # padded_qubit_nbr = params.prep_4112_params["total_qubit_nbr"]
        padded_qubit_nbr = 0
        return int(coord.real + coord.imag * (z_distance + 0.5)) + padded_qubit_nbr
    
    cycle_actions = stim.Circuit()
    params.append_reset(cycle_actions, measurement_qubits, if_noiseless=True)
    params.append_begin_round_tick(cycle_actions, data_qubits, if_noiseless=True)
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits, if_noiseless=True)
    for targets in [cnot_targets[0]]:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets, if_noiseless=True)
        # print(targets)
    for targets in [cnot_targets[1]]:
        cycle_actions.append_operation("TICK", [])
        pairs = list(zip(targets[::2], targets[1::2]))
        for target_pair in pairs:
            # if coord_to_index(0+4j, params) in target_pair or coord_to_index(2+4j, params) in target_pair \
            #     or coord_to_index(4+4j, params) in target_pair or coord_to_index(6+4j, params) in target_pair:
            # if coord_to_index(4+4j, params) in target_pair \
            #     or coord_to_index(4+4j, params) in target_pair:
            #     params.append_unitary_2(cycle_actions, "CNOT", list(target_pair), if_noiseless=False)
            # else:
            #     params.append_unitary_2(cycle_actions, "CNOT", list(target_pair), if_noiseless=True)
            
            params.append_unitary_2(cycle_actions, "CNOT", list(target_pair), if_noiseless=False)
        # print(targets)
    for targets in [cnot_targets[2]]:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets, if_noiseless=True)
        # print(targets)
    for targets in [cnot_targets[3]]:
        cycle_actions.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions, "CNOT", targets, if_noiseless=True)
        # print(targets)
    cycle_actions.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions, "H", x_measurement_qubits, if_noiseless=True)
    cycle_actions.append_operation("TICK", [])
    # params.append_measure_reset(cycle_actions, measurement_qubits)
    params.append_measure(cycle_actions, measurement_qubits, if_noiseless=True)
    '''

    # define the noiseless SE circuit
    cycle_actions_ideal = stim.Circuit()
    params.append_reset(cycle_actions_ideal, measurement_qubits, if_noiseless = True)
    params.append_begin_round_tick(cycle_actions_ideal, data_qubits, if_noiseless = True)
    params.append_unitary_1(cycle_actions_ideal, "H", x_measurement_qubits, if_noiseless = True)
    for targets in cnot_targets[:4]:
        cycle_actions_ideal.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions_ideal, "CNOT", targets, if_noiseless = True)
        # print(targets)
    cycle_actions_ideal.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions_ideal, "H", x_measurement_qubits, if_noiseless = True)
    cycle_actions_ideal.append_operation("TICK", [])
    params.append_measure(cycle_actions_ideal, measurement_qubits, if_noiseless = True)


    ####################

    # 1. Plus state preparation by d rounds of syndrome measurement : head

    head = stim.Circuit()
    for k, v in sorted(sc.ind2coord.items()):
        head.append_operation("QUBIT_COORDS", [k], [v.real, v.imag])

    # list of the detectors used for post-selection
    # Record the reversed location: the last detector in the whole circuit is -1 
    Post_Dec_List = []

    # Ideal initialization of the surface code: all + state for the data qubits
    params.append_reset(head, data_qubits, "X", if_noiseless = True)
    # first round of surface code syndrome check for code expansion
    head += cycle_actions
    # head += cycle_actions_ideal
    # for all the x-type syndrome check, append a detector
    for m_index in x_measurement_qubits:
        m_coord = sc.ind2coord[m_index]
        # all_touched_data_qubit_coord = [x_order_add + m_coord for x_order_add in sc.x_order  if x_order_add + m_coord in sc.data_coords] # first list all x_order + m_coord results, then select the ones in data_coords
        head.append_operation(
            "DETECTOR",
            [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[m_coord])],
            [m_coord.real, m_coord.imag, 0.0] # this provide the coordinate to the detector
            )
            # print("X detectors at: ", m_coord)
        Post_Dec_List = [x - 1 for x in Post_Dec_List]        
        if (m_index in x_measurement_qubits_post): # measurement qubit belong to the post-selection set 
            Post_Dec_List.append(-1)

    

    # Build the repeated body of the circuit, including the detectors comparing to previous cycles.
    # rounds_prep = 1
    for t in range(rounds_prep):
        SEcirc = cycle_actions.copy()
        # SEcirc = cycle_actions_ideal.copy()
        meas_num = len(measurement_qubits)
        SEcirc.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
        # append detectors on all the measurement qubits
        for m_index in measurement_qubits:
            m_coord = sc.ind2coord[m_index]
            k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
            SEcirc.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
                [m_coord.real, m_coord.imag, 0.0]
            )
            Post_Dec_List = [x - 1 for x in Post_Dec_List]
            if (m_index in x_measurement_qubits_post) or (m_index in z_measurement_qubits_post): # measurement qubit belong to the post-selection set 
                Post_Dec_List.append(-1)

        head += SEcirc    # used for the circuit checking
        head.append('TICK')


    ####################

    # 2. Apply 'sampled' ZZZ operation and noise on the logical Z edge: body

    body = stim.Circuit()

    # check whether the length of b string is the same as the length of logical Z operator
    # z_logical_qubits = [sc.coord2ind[p] for p in sc.z_observable]
    if len(b) != len(z_logical_qubits):
        raise ValueError("Need a b string with the length of d")

    for idLz, Lz_index in enumerate(sorted(z_logical_qubits)):
        if b[idLz] == True:
            # Apply a Pauli Z operator on the corresponding data qubit
            # body.append_operation("Z", Lz_index)
            body.append_operation("Z_ERROR", Lz_index, 1)  # add a deterministic Z error on the qubit
    
    # append the gate noise after each "ZZ(\theta)" gate; CNOT extension and its noise also added
    if len(z_logical_qubits) % 3 != 0:
        raise ValueError("z logical should be multiples of 3")
    # append CNOT and the corresponding depolarization noises
    for Lz_index_2, Lz_index_3 in zip(z_logical_qubits[1::3], z_logical_qubits[2::3]):  # enumerate the 2rd and 3th qubit in each m=3 group
        body.append_operation("CNOT", [Lz_index_2, Lz_index_3])
        body.append_operation("DEPOLARIZE2", [Lz_index_2, Lz_index_3], after_clifford_depolarization)

    if ZZtype == 0:     # the gate is done naively, append depolarizing noise
        for Lz_index_1, Lz_index_2 in zip(z_logical_qubits[::3], z_logical_qubits[1::3]):  # enumerate the 1st and 2nd qubit in each m=3 group
            body.append_operation("DEPOLARIZE2", [Lz_index_1, Lz_index_2], after_clifford_depolarization)
    else:  # the gate is done by dispersive coupling
        # load the fitting parameters for the ZZ rotation gate
        with open("fitting_parameters_phi_0p1_naive.pkl", "rb") as f:
            loaded_parameters = pickle.load(f)

        # Define the Pauli operators for labeling
        pauli_operators = [
            "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"
        ]   # no identity term

        pauli_error_vector = np.zeros(15) # register for Pauli error rates

        p_dep = after_clifford_depolarization

        for id_pauli, pauli in enumerate(pauli_operators):
            Pauli2_parameter = loaded_parameters[pauli]
            shape = Pauli2_parameter["shape"]
            coefficients = Pauli2_parameter["coefficients"]
            if shape == "linear":
                pauli_error_vector[id_pauli] = coefficients[0]*p_dep
            elif shape == "quadratic":
                pauli_error_vector[id_pauli] = coefficients[0]*(p_dep**2)
            elif shape == "hybrid":
                pauli_error_vector[id_pauli] = coefficients[0] * p_dep + coefficients[1] * p_dep**2
            else:
                print("Error!")
                exit()
        for Lz_index_1, Lz_index_2 in zip(z_logical_qubits[::3], z_logical_qubits[1::3]):  # enumerate the 1st and 2nd qubit in each m=3 group
            body.append_operation("PAULI_CHANNEL_2", [Lz_index_1, Lz_index_2], pauli_error_vector)  # D0 and D2 rotation gate: Pauli errors

        # # erasure error
        # Pauli2_parameter = loaded_parameters["Pfail"]
        # shape = Pauli2_parameter["shape"]
        # coefficients = Pauli2_parameter["coefficients"]
        # if shape == "linear":
        #     pfail = coefficients[0]*p_dep
        # elif shape == "quadratic":
        #     pfail = coefficients[0]*(p_dep**2)
        # elif shape == "hybrid":
        #     pauli_error_vector[id_pauli] = coefficients[0] * p_dep + coefficients[1] * p_dep**2
        # else:
        #     print("Error!")
        #     exit()

        # for Lz_index_1, Lz_index_2 in zip(z_logical_qubits[::2], z_logical_qubits[1::2]):  # enumerate the pairs
        #     Lz_coord_1 = sc.ind2coord[Lz_index_1]
        #     body.append_operation("PAULI_CHANNEL_2", [Lz_index_1, Lz_index_2], pauli_error_vector)  # D0 and D2 rotation gate: Pauli errors
        #     body.append("HERALDED_ERASE", Lz_index_1, pfail)
        #     body.append("DETECTOR", stim.target_rec(-1), [Lz_coord_1.real, Lz_coord_1.imag, 0.0])
    
    # append CNOT and the corresponding depolarization noises
    for Lz_index_2, Lz_index_3 in zip(z_logical_qubits[1::3], z_logical_qubits[2::3]):  # enumerate the 2rd and 3th qubit in each m=3 group
        body.append_operation("CNOT", [Lz_index_2, Lz_index_3])
        body.append_operation("DEPOLARIZE2", [Lz_index_2, Lz_index_3], after_clifford_depolarization)

    body.append("TICK")

    ####################

    # 3. Post-selection and extra SE operations: tail

    tail = stim.Circuit()

    # first round: check the detector corrspondence more carefully: identify the post-selection detector
        # introduce a register for all the post-selection detectors
    # 2~(rounds_post) round: identify the post-selection detector
    # rounds_post+1 ~ rounds: normal SE circuit


    ## first round
    tail_1stSE = cycle_actions.copy()
    # tail_1stSE = cycle_actions_ideal.copy()   # for test usage
    meas_num = len(measurement_qubits)
    tail_1stSE.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    
    # counter for the erasure detection from the ZZ rotation gate
    Num_Erasure = 0

    # # Update erausre detector to the list of post-selction detectors
    # if ZZtype != 0:  # the dispersive ZZ rotation
    #     Num_Erasure = int(len(z_logical_qubits)/2)

    #     Post_Dec_List = [x - Num_Erasure for x in Post_Dec_List]
    #     Post_Dec_List.extend(range(-Num_Erasure, 0))
    
    # append detectors on all the measurement qubits
    for m_index in measurement_qubits:
        m_coord = sc.ind2coord[m_index]
        k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
        tail_1stSE.append_operation(
            "DETECTOR",
            [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num - Num_Erasure)],
            [m_coord.real, m_coord.imag, 0.0]
        )
        Post_Dec_List = [x - 1 for x in Post_Dec_List]
        if (m_index in x_measurement_qubits_post) or (m_index in z_measurement_qubits_post): # measurement qubit belong to the post-selection set 
            Post_Dec_List.append(-1)
        
    
    tail += tail_1stSE
    tail.append_operation('TICK')
    
    ## second to rounds_post round
    meas_num = len(measurement_qubits)
    for t in range(2, rounds_post + 1):
        tail_2ndSE = cycle_actions.copy()
        # tail_2ndSE = cycle_actions_ideal.copy()   # for test usage
        tail_2ndSE.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        # append detectors on all the measurement qubits
        for m_index in measurement_qubits:
            m_coord = sc.ind2coord[m_index]
            k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
            tail_2ndSE.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
                [m_coord.real, m_coord.imag, 0.0]
            )
            Post_Dec_List = [x - 1 for x in Post_Dec_List]
            if (m_index in x_measurement_qubits_post) or (m_index in z_measurement_qubits_post): # measurement qubit belong to the post-selection set 
                Post_Dec_List.append(-1)

        tail += tail_2ndSE
        tail.append_operation('TICK')

    ## rounds_post+1 to rounds round
    meas_num = len(measurement_qubits)
    for t in range(rounds_post + 1, rounds + 1):
        tail_3rdSE = cycle_actions.copy()
        # tail_3rdSE = cycle_actions_ideal.copy()   # for test usage
        tail_3rdSE.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        # append detectors on all the measurement qubits
        for m_index in measurement_qubits:
            m_coord = sc.ind2coord[m_index]
            k = meas_num - measure_coord_to_order[m_coord] - 1   # measure_coord_to_order convert coordinate to the rank of the measurement
            tail_3rdSE.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
                [m_coord.real, m_coord.imag, 0.0]
            )
        
        Post_Dec_List = [x - meas_num for x in Post_Dec_List]

        tail += tail_3rdSE
        tail.append_operation('TICK')


    ## add ideal QEC and then logical X measurement

    # Build the ideal QEC, add detectors
    ideal_QEC = cycle_actions_ideal.copy()
    meas_num = len(measurement_qubits)

    ideal_QEC.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    for m_index in measurement_qubits:
        m_coord = sc.ind2coord[m_index]
        k = meas_num - measure_coord_to_order[m_coord] - 1
        ideal_QEC.append_operation(
            "DETECTOR",
            [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - meas_num)],
            [m_coord.real, m_coord.imag, 0.0]
        )
    
    Post_Dec_List = [x - meas_num for x in Post_Dec_List]

    tail += ideal_QEC
    tail.append_operation('TICK')

    # Append logical X measurement at the end
    params.append_measure(tail, x_logical_qubits, "X", if_noiseless=True) 
    # add X observable
    tail.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-len(sc.x_observable),0)], 0.0)

        

    # the overall circuit:
    Proj_Cir = stim.Circuit()
    Proj_Cir = head + body + tail

    return (Proj_Cir, Post_Dec_List)