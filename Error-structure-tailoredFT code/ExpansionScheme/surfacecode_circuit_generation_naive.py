import stim
import numpy as np
import scipy as sp
import util
# import json
from typing import Callable, Set, List, Dict, Tuple, Optional
from dataclasses import dataclass
import math 

import pickle   # for the data loading

def append_anti_basis_error(circuit: stim.Circuit, targets: List[int], p: float, basis: str) -> None:
    if p > 0:
        if basis == "X":
            circuit.append_operation("Z_ERROR", targets, p)
        else:
            circuit.append_operation("X_ERROR", targets, p)


@dataclass
class CircuitGenParameters:   # the class is used to record the circuit parameters
    rounds: int
    prep_4112_params: dict
    distance: int = None
    x_distance: int = None
    z_distance: int = None
    after_clifford_depolarization: float = 0
    before_round_data_depolarization: float = 0
    before_measure_flip_probability: float = 0
    after_reset_flip_probability: float = 0
    exclude_other_basis_detectors: bool = False
    qubit_initialization_pattern: str = ''
    top_left_2_body_meas_type: str = 'X'

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
## Step I: first define [4,1,1,2] rotation state preparation circuit
# def 4112prep(p_dep:float, p_decay:float, p_dephase:float) -> stim.Circuit:
def StatePrep4112(p_dep:float, basis:int, angle:int) -> tuple[stim.Circuit, dict]:
    '''
    Circuit to prepare the 4112 rotation state
    2-level transmon: depolarizing noise; 3-level transmon: decay + dephasing
    Return the circuit and 4112 parameter dictionary.

    angle: 0: phi = 0; 1: phi = np.pi/4
    '''

    # circuit initialization
    # N4112 = 8   # number of qubits in the 4112 circuit
    N = 8     # number of qubits in the 4112 circuit

    # sub-circuit offset
    Ind0_4112 = 0
    qubits = np.array(range(N))
    data = qubits[2:6]   # 4 qubits in the middle: data
    ancilla = qubits[np.r_[0:2,6:8]]   # 4 qubits on both sides: ancilla

    # Load the fitting parameters
    if angle == 0:   # phi = 0
        with open("fitting_parameters_phi_0_naive.pkl", "rb") as f:
            loaded_parameters = pickle.load(f)
    else:   # phi = np.pi/4
        with open("fitting_parameters_phi_piover4_naive.pkl", "rb") as f:
            loaded_parameters = pickle.load(f)


    ############################
    # subcircuit definition
    def ZX_QED_4112(circuit:stim.Circuit, Ind0_4112:int, rec_g:int, p_dep: float) -> tuple[stim.Circuit, int, int]:
        '''
        ZX_QED circuit as a subroutine.
        The gauge detector location rec_g < 0 is used and updated. 
        If rec_g = 0, then the Z gauge value is known to be a fixed value 0
        '''

        # counter for the number of measurements in the subroutine
        numM = 0

        ## 4112 1-FT ZX-QED procedure: detector 1,2,3
        circuit.append("R", [x+Ind0_4112 for x in [0,1]])
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [0,1]], p_dep) 
        circuit.append("TICK")

        circuit.append("CNOT", [x+Ind0_4112 for x in [2,0,4,1]])
        circuit.append("DEPOLARIZE2", [x+Ind0_4112 for x in [2,0,4,1]], p_dep) 
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [3,5]], p_dep)
        circuit.append("R", [x+Ind0_4112 for x in [6,7]])
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [6,7]], p_dep)
        circuit.append("TICK")

        circuit.append("CNOT", [x+Ind0_4112 for x in [3,0,5,1]])
        circuit.append("DEPOLARIZE2", [x+Ind0_4112 for x in [3,0,5,1]], p_dep) 
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [2,4]], p_dep)
        circuit.append("H", [x+Ind0_4112 for x in [6,7]])
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [6,7]], p_dep)
        circuit.append("TICK")

        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [0,1]], p_dep)  # measurement error
        if rec_g == 0:
            circuit.append("M", [x+Ind0_4112 for x in [0,1]])
            circuit.append("DETECTOR", [stim.target_rec(x) for x in [-2]])  # gauge Z operator detector: should be 0
        else:
            circuit.append("M", [x+Ind0_4112 for x in [0,1]])
            rec_g = rec_g - 2
            circuit.append("DETECTOR", [stim.target_rec(x) for x in [-2, rec_g]])  # gauge Z operator detector: should be 0
        numM = numM + 2
        circuit.append("DETECTOR",[stim.target_rec(x) for x in [-1,-2]])  # SZ stabilizer: should be 0
        circuit.append("CNOT", [x+Ind0_4112 for x in [6,2,7,3]])
        circuit.append("DEPOLARIZE2", [x+Ind0_4112 for x in [6,2,7,3]], p_dep) 
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [4,5]], p_dep)
        circuit.append("TICK")

        circuit.append("CNOT", [x+Ind0_4112 for x in [6,4,7,5]])
        circuit.append("DEPOLARIZE2", [x+Ind0_4112 for x in [6,4,7,5]], p_dep) 
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [2,3]], p_dep)
        circuit.append("TICK")

        circuit.append("H", [x+Ind0_4112 for x in [6,7]])
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [6,7]], p_dep) # H gate error
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in data], p_dep) # idling error
        circuit.append("TICK")

        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [6,7]], p_dep) # measurement error
        circuit.append("M", [x+Ind0_4112 for x in [6,7]])
        rec_g = -2  # reset the location of gauge detector
        numM = numM +2
        circuit.append("DETECTOR",[stim.target_rec(x) for x in [-1,-2]])  # SX stabilizer: should be 0
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in data], p_dep) # idling error
        circuit.append("TICK")

        return circuit, rec_g, numM


    def XZ_QED_4112(circuit:stim.Circuit, Ind0_4112:int, rec_g:int, p_dep: float) -> tuple[stim.Circuit, int, int]:
        '''
        XZ_QED circuit as a subroutine.
        The gauge detector location rec_g < 0 is used and updated. 
        If rec_g = 0, then the Z gauge value is known to be a fixed value 0
        '''
        
        numM = 0

        circuit.append("R", [x+Ind0_4112 for x in [6,7]])
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [6,7]], p_dep)
        circuit.append("TICK")

        circuit.append("H", [x+Ind0_4112 for x in [6,7]])
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [6,7]], p_dep)
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in data], p_dep) # idling error
        circuit.append("TICK")

        circuit.append("CNOT", [x+Ind0_4112 for x in [6,2,7,3]])
        circuit.append("DEPOLARIZE2", [x+Ind0_4112 for x in [6,2,7,3]], p_dep) 
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [4,5]], p_dep)
        circuit.append("TICK")

        circuit.append("CNOT", [x+Ind0_4112 for x in [6,4,7,5]])
        circuit.append("DEPOLARIZE2", [x+Ind0_4112 for x in [6,4,7,5]], p_dep) 
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [2,3]], p_dep)
        circuit.append("R", [x+Ind0_4112 for x in [0,1]])
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [0,1]], p_dep)
        circuit.append("TICK")

        circuit.append("CNOT", [x+Ind0_4112 for x in [2,0,4,1]])
        circuit.append("DEPOLARIZE2", [x+Ind0_4112 for x in [2,0,4,1]], p_dep) 
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [3,5]], p_dep)
        circuit.append("H", [x+Ind0_4112 for x in [6,7]])
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [6,7]], p_dep)
        circuit.append("TICK")

        circuit.append("CNOT", [x+Ind0_4112 for x in [3,0,5,1]])
        circuit.append("DEPOLARIZE2", [x+Ind0_4112 for x in [3,0,5,1]], p_dep) 
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [2,4]], p_dep)
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [6,7]], p_dep)  # measurement error
        circuit.append("M", [x+Ind0_4112 for x in [6,7]])
        if rec_g == 0:   # the gauge X operator is fixed at the beginning
            circuit.append("DETECTOR",[stim.target_rec(x) for x in [-2]])
        else:   # the gauge X is determined by former measurements
            rec_g = rec_g - 2
            # rec_MZZlast = rec_MZZlast - 2
            circuit.append("DETECTOR",[stim.target_rec(x) for x in [-2, rec_g]])  # gauge X operator detector: should be the same as first measurement
        numM = numM + 2
        circuit.append("DETECTOR",[stim.target_rec(x) for x in [-1,-2]])  # SX stabilizer: should be 0
        circuit.append("TICK")

        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in data], p_dep) # idling error
        circuit.append("DEPOLARIZE1", [x+Ind0_4112 for x in [0,1]], p_dep)  # measurement error
        circuit.append("M", [x+Ind0_4112 for x in [0,1]])
        # rec_MZZlast = rec_MZZlast - 2
        rec_g = -2
        numM = numM + 2
        circuit.append("DETECTOR",[stim.target_rec(x) for x in [-1,-2]])  # SZ stabilizer: should be 0
        circuit.append("TICK")
        
        return circuit, rec_g, numM

    #####################
    # the 4112 rot-state prep circuit

    circ0 = stim.Circuit()


    # state prep on the 4112 code
    match basis:
        case 0: # X-basis
            circ0.append("R", [x+Ind0_4112 for x in data])
            circ0.append("DEPOLARIZE1", [x+Ind0_4112 for x in data], p_dep) # data state preparation error
            circ0.append("TICK")   # for synchronizing the circuit, no real use
            # prepare two Bell pairs: should be 1-FT; the gauge qubit is prepared to + state
            circ0.append("H", [x+Ind0_4112 for x in [data[0], data[2]]] )  # D0 and D2 add Hadamard
            circ0.append("DEPOLARIZE1", [x+Ind0_4112 for x in data], p_dep) # both H and idling experience error
            circ0.append("CNOT", [x+Ind0_4112 for x in data])
            circ0.append("DEPOLARIZE2", [x+Ind0_4112 for x in data], p_dep) 
            circ0.append("TICK")
        case 1: # Z-basis
            circ0.append("R", [x+Ind0_4112 for x in data])
            circ0.append("DEPOLARIZE1", [x+Ind0_4112 for x in data], p_dep) # data state preparation error
            circ0.append("TICK")   # for synchronizing the circuit, no real use
            # prepare two Bell pairs: should be 1-FT; the gauge qubit is prepared to 0 state
            circ0.append("H", [x+Ind0_4112 for x in [data[0], data[1]]] )  # D0 and D2 add Hadamard
            circ0.append("DEPOLARIZE1", [x+Ind0_4112 for x in data], p_dep) # both H and idling experience error
            circ0.append("CNOT", [x+Ind0_4112 for x in [data[0], data[2], data[1], data[3]] ])
            circ0.append("DEPOLARIZE2", [x+Ind0_4112 for x in [data[0], data[2], data[1], data[3]]], p_dep) 
            circ0.append("TICK")
        case 2: # Y-basis: after the X-basis preparation: add an extra noiseless S gate   (Problematic? Maybe enough for the benchmarking purpose)
            # 1-FT state preparation without measurement
            circ0 = stim.Circuit()
            # reset the data qubits0
            circ0.append("R", [x+Ind0_4112 for x in data])
            circ0.append("DEPOLARIZE1", [x+Ind0_4112 for x in data], p_dep) # data state preparation error
            circ0.append("TICK")   # for synchronizing the circuit, no real use

            # prepare two Bell pairs: should be 1-FT; the gauge qubit is prepared to 0 state
            circ0.append("H", [x+Ind0_4112 for x in [data[0], data[2]]] )  # D0 and D2 add Hadamard
            circ0.append("DEPOLARIZE1", [x+Ind0_4112 for x in data], p_dep) # both H and idling experience error
            circ0.append("CNOT", [x+Ind0_4112 for x in data])
            circ0.append("DEPOLARIZE2", [x+Ind0_4112 for x in data], p_dep) 
            circ0.append("TICK")

            #! extra noiseless logical S gate
            circ0.append("CNOT", [x+Ind0_4112 for x in [data[0], data[2]]] )
            circ0.append("S", data[2]+Ind0_4112 )
            circ0.append("CNOT", [x+Ind0_4112 for x in [data[0], data[2]]] )  


    # perform the ZZ-rot gate with the noise structure appended
    match angle:
        case 0: # phi == 0
            # Define the Pauli operators for labeling
            pauli_operators = [
                "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"
            ]   # no identity term

            pauli_error_vector = np.zeros(15) # register for Pauli error rates

            for id_pauli, pauli in enumerate(pauli_operators):
                params = loaded_parameters[pauli]
                shape = params["shape"]
                coefficients = params["coefficients"]
                if shape == "linear":
                    pauli_error_vector[id_pauli] = coefficients[0]*p_dep
                elif shape == "quadratic":
                    pauli_error_vector[id_pauli] = coefficients[0]*(p_dep**2)
                else:
                    print("Error!")
                    exit()

            circ0.append("PAULI_CHANNEL_2", [x+Ind0_4112 for x in [data[0], data[2]]], pauli_error_vector)  # D0 and D2 rotation gate: Pauli errors

            # #! we first ignore the erasure failure here! This will affect the detector error model.
            # # erasure error
            # params = loaded_parameters["Pfail"]
            # shape = params["shape"]
            # coefficients = params["coefficients"]
            # if shape == "linear":
            #     pfail = coefficients[0]*p_dep
            # elif shape == "quadratic":
            #     pfail = coefficients[0]*(p_dep**2)
            # else:
            #     print("Error!")
            #     exit()
            
            # circ0.append("HERALDED_ERASE", data[0]+Ind0_4112, pfail)
            # circ0.append("DETECTOR", stim.target_rec(-1))
            
            circ0.append("TICK")


        case 1: # phi == np.pi/4
            # first implement the "noiseless" S gate
            # extra noiseless logical S gate
            circ0.append("CNOT", [x+Ind0_4112 for x in [data[0], data[2]]] )
            circ0.append("S", data[2]+Ind0_4112 )
            circ0.append("CNOT", [x+Ind0_4112 for x in [data[0], data[2]]] )

            # then append the noise channel
            # Define the Pauli operators for labeling
            pauli_operators = [
                "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"
            ]   # no identity term

            pauli_error_vector = np.zeros(15) # register for Pauli error rates

            for id_pauli, pauli in enumerate(pauli_operators):
                params = loaded_parameters[pauli]
                shape = params["shape"]
                coefficients = params["coefficients"]
                if shape == "linear":
                    pauli_error_vector[id_pauli] = coefficients[0]*p_dep
                elif shape == "quadratic":
                    pauli_error_vector[id_pauli] = coefficients[0]*(p_dep**2)
                else:
                    print("Error!")
                    exit()

            circ0.append("PAULI_CHANNEL_2", [x+Ind0_4112 for x in [data[0], data[2]]], pauli_error_vector)  # D0 and D2 rotation gate: Pauli errors

            # # erasure error
            # params = loaded_parameters["Pfail"]
            # shape = params["shape"]
            # coefficients = params["coefficients"]
            # if shape == "linear":
            #     pfail = coefficients[0]*p_dep
            # elif shape == "quadratic":
            #     pfail = coefficients[0]*(p_dep**2)
            # else:
            #     print("Error!")
            #     exit()

            # circ0.append("HERALDED_ERASE", data[0]+Ind0_4112, pfail)
            # circ0.append("DETECTOR", stim.target_rec(-1))
            circ0.append("TICK")

    # perform 4112-QED
    match basis:
        case 0 | 2: # X-basis (Y basis also follow the same procedure) 
            # ZX-QED: rec_g = 0
            rec_g = 0
            circ0, rec_g, numM = ZX_QED_4112(circ0, Ind0_4112, rec_g, p_dep)
            gauge_basis = 'X'
        case 1: # Z-basis
            # XZ-QED: rec_g = 0
            rec_g = 0
            circ0, rec_g, numM = XZ_QED_4112(circ0, Ind0_4112, rec_g, p_dep)
            gauge_basis = 'Z'

    def basis_conversion(basis):  # logical basis register
        if basis == 0:
            return 'X'
        elif basis == 1:
            return 'Z'
        elif basis == 2:
            return 'Y'

    #! assign the label correspondence from 4112 index (2,3,4,5) to the data qubit (0,1,2,3)
    # Here we use the convention that for 4112 code: LX = X0X1, LZ = Z0Z2, GX = X0X2, GZ = Z0Z1
    # For rotated surface code, the data qubits on the upper left corner has coordinates (1, 1), (3, 1), (1, 3), (3, 3) with a two-body stabilizer measuring (1,1) and (3,1)
    # This step is quite manual: the values of 0,1,2,3 correspond to the index of the data qubits
    index_to_label_4112 = {0: 2, 1: 3, 2: 4, 3: 5}

    prep_4112_params = {"total_qubit_nbr": N,
                        "prepare_logical_basis": basis_conversion(basis),
                        "gauge_basis": gauge_basis,
                        "gauge_rec_index_reverse": rec_g, 
                        "index_to_label_4112": index_to_label_4112, 
                        "p_dep": p_dep,
                        "angle": angle
                        }  # parameter dictionary
    return circ0, prep_4112_params



#####################################
## Step II: define (1) the 4112->surface expansion circuit and (2) complete the surface code circuit

### complex coordinate and the true coordinate correspondence example: 1 + 3j -> (1, 3) : the first row, the third column

## function to connect [4,1,1,2] circuit with the later surface code expansion circuit
def generate_4112_to_surface_expansion_circuit_from_params(
        params: CircuitGenParameters,
) -> stim.Circuit:
    
    if params.prep_4112_params['prepare_logical_basis'] == 'X':
        is_memory_x = True
    elif params.prep_4112_params['prepare_logical_basis'] == 'Z':
        is_memory_x = False
    else:  # params.prep_4112_params['prepare_logical_basis'] == 'Y':
        is_memory_x = True   # the gauge setting and surface code expansion method of Y basis is the same as X basis

    x_distance = params.distance
    z_distance = params.distance

    # Place data qubits: specified by the complex-valued coordinates q
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
    x_measure_coords: Set[complex] = set()
    z_measure_coords: Set[complex] = set()
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
            else:
                z_measure_coords.add(q)

    # Define interaction orders so that hook errors run against the error grain instead of with it.
    z_order: List[complex] = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    x_order: List[complex] = [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]

    # convert the coordinates to indicies
    # we remark that, many redundant q is introduced to keep the coord->index mapping simple
    def coord_to_index(q: complex, params: dict=None) -> int:
        q = q - math.fmod(q.real, 2) * 1j
        padded_qubit_nbr = params.prep_4112_params["total_qubit_nbr"]
        return int(q.real + q.imag * (z_distance + 0.5)) + padded_qubit_nbr

    # based on the requirement on the top left 2-body check, determine whether to flip the surface code
    # data qubit coordinates keep invariant: change the x,z parity check & observable definition
    if params.top_left_2_body_meas_type == 'Z':
        x_observable, z_observable = util.swap_sets(x_observable, z_observable)
        x_measure_coords, z_measure_coords = util.swap_sets(x_measure_coords, z_measure_coords)
        x_order, z_order = util.swap_sets(x_order, z_order)
    
    # Here, we return the circuit and number of ideal detectors generated from the function `finish_surface_code_circuit` below
    return finish_surface_code_circuit(
        coord_to_index,
        data_coords,
        x_measure_coords,
        z_measure_coords,
        params,
        x_order,
        z_order,
        x_observable,
        z_observable,
        is_memory_x
    )


## detailed surface code circuit
# We first perform 3 rounds of noisy QED, then add an extra round of ideal QEC, then determine the x/y measurement based on the phi value
def finish_surface_code_circuit(
        coord_to_index: Callable[[complex, dict], int],
        data_coords: Set[complex],
        x_measure_coords: Set[complex],
        z_measure_coords: Set[complex],
        params: CircuitGenParameters,
        x_order: List[complex],
        z_order: List[complex],
        x_observable: List[complex],
        z_observable: List[complex],
        is_memory_x: bool,   # determine it is either x or z memory
        *,
        exclude_other_basis_detectors: bool = False,
        wraparound_length: Optional[int] = None
) -> tuple[stim.Circuit, int]:
    if params.rounds < 1:
        raise ValueError("Need rounds >= 1")
    if params.distance is not None and params.distance < 2:
        raise ValueError("Need a distance >= 2")
    if params.x_distance is not None and (params.x_distance < 2 or
                                          params.z_distance < 2):
        raise ValueError("Need a distance >= 2")

    chosen_basis_observable = x_observable if is_memory_x else z_observable   ### need to be revised
    chosen_basis_measure_coords = x_measure_coords if is_memory_x else z_measure_coords

    # Index the measurement qubits and data qubits: store them to dictionaries.
    ###  p: coordinate ;  q: index   
    p2q: Dict[complex, int] = {}
    for q in data_coords:
        p2q[q] = coord_to_index(q, params)

    for q in x_measure_coords:
        p2q[q] = coord_to_index(q, params)

    for q in z_measure_coords:
        p2q[q] = coord_to_index(q, params)

    q2p: Dict[int, complex] = {v: k for k, v in p2q.items()}   # define a dictionary from index to coord


    data_qubits = [p2q[p] for p in data_coords]
    x_logical_qubits = [p2q[p] for p in x_observable]
    z_logical_qubits = [p2q[p] for p in z_observable]
    measurement_qubits = [p2q[p] for p in x_measure_coords]
    measurement_qubits += [p2q[p] for p in z_measure_coords]
    x_measurement_qubits = [p2q[p] for p in x_measure_coords]

    # set the surface code initialization
    if params.qubit_initialization_pattern == 'sym':
        initial_data_z_coord = []
        initial_data_x_coord = []
        for q in data_coords:
            if params.top_left_2_body_meas_type == 'X':
                if q.real<=q.imag:                # arguable: try asymmetric construction: check which one is better
                    initial_data_x_coord.append(q)
                else:
                    initial_data_z_coord.append(q)
            else:   # params.top_left_2_body_meas_type == 'Z':
                if q.real<=q.imag:                
                    initial_data_z_coord.append(q)
                else:
                    initial_data_x_coord.append(q)
    else:  # params.qubit_initialization_pattern == 'asym':
        initial_data_z_coord = []
        initial_data_x_coord = []
        for q in data_coords:
            if params.top_left_2_body_meas_type == 'X':
                if q.real <= 3:  # for the first two rows: initialize based on the gauge stabilizer type                
                    initial_data_x_coord.append(q)
                else:
                    initial_data_z_coord.append(q)
            else: # params.top_left_2_body_meas_type == 'Z':
                if q.real <= 3:  # for the first two rows: initialize based on the gauge stabilizer type                
                    initial_data_z_coord.append(q)
                else:
                    initial_data_x_coord.append(q)

    
    # List to record the location of 4112 data qubits: (order is not important, this is only for the detector assignment usage)
    data_4112_qubits_coord = [1 + 1j, 1 + 3j, 3 + 1j, 3 + 3j]

    [initial_data_x, initial_data_z] = util.map_list_of_list([initial_data_x_coord, initial_data_z_coord], coord_to_index, params)

    # if params.expand_4112_type != '':
    #     data_4112_qubits_coord = [1 + 1j, 1 + 3j, 3 + 1j, 3 + 3j]
    #     meas_4112_qubits_x_coord = [2 + 0j, 2 + 4j]
    #     meas_4112_qubits_z_coord = [2 + 2j]

    #     [data_4112_qubits, meas_4112_qubits_x, meas_4112_qubits_z] = util.map_list_of_list([data_4112_qubits_coord, meas_4112_qubits_x_coord, meas_4112_qubits_z_coord], coord_to_index, params)
    #     # print('4112 data qubits', data_4112_qubits)
    #     if params.expand_4112_type == 'rect':
    #         initial_data_z_coord = [q for q in data_coords if (q.real not in (1, 3))]
    #         initial_data_x_coord = [q for q in data_coords if not (q.real not in (1, 3))]
    #         # print("X data coord: ", initial_data_x_coord)
    #         [initial_data_x, initial_data_z] = util.map_list_of_list([initial_data_x_coord, initial_data_z_coord], coord_to_index, params)
        
        
    #     elif params.expand_4112_type == 'sym':
    #         initial_data_z_coord = []
    #         initial_data_x_coord = []
    #         for q in data_coords:
    #             if q == 3 + 1j:
    #                 initial_data_x_coord.append(q)
    #             elif q.real<=q.imag:
    #                 initial_data_x_coord.append(q)
    #             else:
    #                 initial_data_z_coord.append(q)

    #         # print("X data coord: ", initial_data_x_coord)
    #         [initial_data_x, initial_data_z] = util.map_list_of_list([initial_data_x_coord, initial_data_z_coord], coord_to_index, params)

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
        data_coord_to_order[q2p[q]] = len(data_coord_to_order)   # this dictionary will provide plain-ordered sequence of all the data qubits: start from 1
    for q in measurement_qubits:
        measure_coord_to_order[q2p[q]] = len(measure_coord_to_order)  # this dictionary will provide plain-ordered sequence of all the measurements

    # List out CNOT gate targets using given interaction orders.
    # CNOT gates are specified by the complex-valued coordinates
    cnot_targets: List[List[int]] = [[], [], [], []]  # list it by 4 time bins
    for k in range(4):
        for measure in sorted(x_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + x_order[k]
            if data in p2q:
                cnot_targets[k].append(p2q[measure])
                cnot_targets[k].append(p2q[data])
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                cnot_targets[k].append(p2q[measure])
                cnot_targets[k].append(p2q[data_wrapped])

        for measure in sorted(z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + z_order[k]
            if data in p2q:
                cnot_targets[k].append(p2q[data])
                cnot_targets[k].append(p2q[measure])
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                cnot_targets[k].append(p2q[data_wrapped])
                cnot_targets[k].append(p2q[measure])

    # Build the repeated actions that make up the surface code cycle
    # split the measurement and reset error!
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


    # Build the start of the circuit, getting a state that's ready to cycle
    # In particular, the first cycle has different detectors and so has to be handled special.
    
    # assign coordinates on the surface code qubits
    # real part is the first axis, imaginary part is the second axis
    head = stim.Circuit()
    for k, v in sorted(q2p.items()):
        head.append_operation("QUBIT_COORDS", [k], [v.real, v.imag])

    # Ideal initialization of the surface code (4112 qubits not included)
    params.append_reset(head, initial_data_x, "X")
    params.append_reset(head, initial_data_z, "Z")
    # params.append_reset(head, measurement_qubits)

    # based on the index_to_label_4112 values (2,3,4,5) and the corrsponding coodinates, swap the qubits
    for _index, (key, value) in enumerate(params.prep_4112_params["from_4112_to_coord"].items()):
        params.append_ideal_swap(head, key, coord_to_index(value, params))

    # # Since ideal, order does not matter here
    # for i in range(4):

    #     # print([data_4112_qubits[i], meas_4112_qubits_z[0]])
    #     params.append_unitary_2(head, "CNOT", [data_4112_qubits[i], meas_4112_qubits_z[0]], if_noiseless=True)
    # params.append_measure_reset(head, meas_4112_qubits_z, if_noiseless=True)

    # # Add state prep error
    # params.append_depolarize_error(head, data_qubits + measurement_qubits, error = params.after_reset_flip_probability)

    # first round of surface code syndrome check for code expansion
    head += cycle_actions

    ###! Setup detectors given by initializations (check!)
    # if all the touched data qubits of a given syndrome check is initialized in x or z, then append a detector
    for m_coord in x_measure_coords:
        all_touched_data_qubit_coord = [x_order_add + m_coord for x_order_add in x_order  if x_order_add + m_coord in data_coords] # first list all x_order + m_coord results, then select the ones in data_coords
        if all([(_coord in initial_data_x_coord and _coord not in data_4112_qubits_coord) for _coord in all_touched_data_qubit_coord]):  # if all the touch qubits are initialized in x and not belong to 4112 data qubits
            head.append_operation(
                "DETECTOR",
                [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[m_coord])],
                [m_coord.real, m_coord.imag, 0.0] # this provide the coordinate to the detector
                )
            # print("X detectors at: ", m_coord)

    for m_coord in z_measure_coords:
        all_touched_data_qubit_coord = [z_order_add + m_coord for z_order_add in z_order  if z_order_add + m_coord in data_coords]
        if all([(_coord in initial_data_z_coord and _coord not in data_4112_qubits_coord) for _coord in all_touched_data_qubit_coord]):
            head.append_operation(
                "DETECTOR",
                [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[m_coord])],
                [m_coord.real, m_coord.imag, 0.0]
                )
    
    #! Setup one detectors for the 4-qubit stabilizer and two detectors for the stabilizers from gauge operators
    head.append_operation(
        "DETECTOR",
        [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[2 + 2j])],
        [2, 2, 0.0]
        )
    
    head.append_operation(
        "DETECTOR",
        [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[2 + 0j]), stim.target_rec(-len(measurement_qubits) + params.prep_4112_params["gauge_rec_index_reverse"])],
        [2, 0, 0.0]
        )
    head.append_operation(
        "DETECTOR",
        [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[2 + 4j]), stim.target_rec(-len(measurement_qubits) + params.prep_4112_params["gauge_rec_index_reverse"])],
        [2, 4, 0.0]
        )
    

    # Build the repeated body of the circuit, including the detectors comparing to previous cycles.
    body = cycle_actions.copy()
    m = len(measurement_qubits)
    body.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    for m_index in measurement_qubits:
        m_coord = q2p[m_index]
        k = len(measurement_qubits) - measure_coord_to_order[m_coord] - 1
        if not exclude_other_basis_detectors or m_coord in chosen_basis_measure_coords:
            body.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                [m_coord.real, m_coord.imag, 0.0]
            )

    #! Build the end of the circuit, getting out of the cycle state and terminating.
    # first perform one round of ideal QEC (not QED!); then measurement X,Y,Z in a non-transversal way (logical observable)
    # the detector here are all used for QEC! need a counter for the usage of later decoding: descriminate QED and QEC

    # In particular, the data measurements create detectors that have to be handled special.
    # Also, the tail is responsible for identifying the logical observable.
    
    # first build noiseless surface code cycle
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

    # Build the ideal QEC, add detectors
    ideal_QEC = cycle_actions_ideal.copy()
    m = len(measurement_qubits)
    num_detectors_ideal = 0
    ideal_QEC.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    for m_index in measurement_qubits:
        m_coord = q2p[m_index]
        k = len(measurement_qubits) - measure_coord_to_order[m_coord] - 1
        # recall that measure_coord_to_order dictionary provides the plain order!
        if not exclude_other_basis_detectors or m_coord in chosen_basis_measure_coords:
            ideal_QEC.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                [m_coord.real, m_coord.imag, 0.0]
            )
            # check the parity of this detector with the former detector at the same location
            num_detectors_ideal = num_detectors_ideal + 1
        
    # Build the logical measurement and observables
    tail_logical_measure = stim.Circuit()
    if params.prep_4112_params['prepare_logical_basis'] == 'X':  # the initial state is prepared to logical +
        if params.prep_4112_params['angle'] == 0:   # phi = 0 
            # in this case, we measure logical X on the edge of logical X qubits
            # recall that 'z_observable' and 'x_observable' store the complex location of the logical data qubits!
            # 'z_logical_qubits' and 'x_logical_qubits' store the index of the qubits
            params.append_measure(tail_logical_measure, x_logical_qubits, "X", if_noiseless=True) 

            # add X observable
            tail_logical_measure.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-len(x_observable),0)], 0.0)
        else: # phi = np.pi/4
            # in this case, we measure logical Y on the edge of logical X and Z qubits
            # Here, we assume that the overlapped data qubit for logical X and Z is the one with coordinate 1+1j
            params.append_measure(tail_logical_measure, x_logical_qubits[1:], "X", if_noiseless=True)
            params.append_measure(tail_logical_measure, z_logical_qubits[1:], "Z", if_noiseless=True)
            params.append_measure(tail_logical_measure, p2q[1+1j], "Y", if_noiseless=True)

            # add logical Y observable
            tail_logical_measure.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-(2*(len(x_observable)-1)+1),0)], 0.0)
    elif params.prep_4112_params['prepare_logical_basis'] == 'Z':  # the initial state is prepared to logical 0
        # regardless of the phi angle, here we measure logical Z
        params.append_measure(tail_logical_measure, z_logical_qubits, "Z", if_noiseless=True)
        # add X observable
        tail_logical_measure.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-len(x_observable),0)], 0.0)
    else:  # params.prep_4112_params['prepare_logical_basis'] == 'Y':  # the initial state is prepared to logical +i
        if params.prep_4112_params['angle'] == 1:   # phi = np.pi/4 
            # in this case, we measure logical X on the edge of logical X qubits
            params.append_measure(tail_logical_measure, x_logical_qubits, "X", if_noiseless=True) 

            # add X observable
            tail_logical_measure.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-len(x_observable),0)], 1.0)
        else: # phi = 0
            # in this case, we measure logical Y on the edge of logical X and Z qubits
            # Here, we assume that the overlapped data qubit for logical X and Z is the one with coordinate 1+1j
            params.append_measure(tail_logical_measure, x_logical_qubits[1:], "X", if_noiseless=True)
            params.append_measure(tail_logical_measure, z_logical_qubits[1:], "Z", if_noiseless=True)
            params.append_measure(tail_logical_measure, p2q[1+1j], "Y", if_noiseless=True)

            # add logical Y observable
            tail_logical_measure.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-(2*(len(x_observable)-1)+1),0)], 0.0)

    """
    # Here, we assume ideal measurements to isolate out state preparation errors.
    tail = stim.Circuit()
    params.append_measure(tail, data_qubits, "ZX"[is_memory_x], if_noiseless=True)  ### THIS BECOMES INVALID WHEN phi=np.pi/4!
    # Detectors
    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        detectors: List[int] = []
        for delta in z_order:
            data = measure + delta
            if data in p2q:
                detectors.append(-len(data_qubits) + data_coord_to_order[data])
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                detectors.append(-len(data_qubits) + data_coord_to_order[data_wrapped])
        detectors.append(-len(data_qubits) - len(measurement_qubits) + measure_coord_to_order[measure])
        detectors.sort(reverse=True)
        tail.append_operation("DETECTOR", [stim.target_rec(x) for x in detectors], [measure.real, measure.imag, 1.0])

    # Logical observable
    obs_inc: List[int] = []
    for q in chosen_basis_observable:
        obs_inc.append(-len(data_qubits) + data_coord_to_order[q])
    obs_inc.sort(reverse=True)
    tail.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in obs_inc], 0.0)

    """
    
    # Combine to form final circuit.
    # return head + body * (params.rounds - 1) + tail
    # return head + body * (params.rounds - 1) + tail_logical_measure
    return (head + body * (params.rounds - 1) + ideal_QEC + tail_logical_measure, num_detectors_ideal)
    


###################################################
## Step III: finish the circuit construction

# function to input prep_4112_circuit, and append the surface code circuits
def complete_circuit(
        prep_4112_circuit: stim.Circuit,  # circuit generated from the function StatePrep4112()
        prep_4112_params: dict,
        rounds: int,
        distance: int = None,
        # x_distance: int = None,
        # z_distance: int = None,
        after_clifford_depolarization: float = 0.0,
        before_round_data_depolarization: float = 0.0,
        before_measure_flip_probability: float = 0.0,
        after_reset_flip_probability: float = 0.0,
        # exclude_other_basis_detectors: bool = False,
        qubit_initialization_pattern: str = 'asym',  # 'asym' or 'sym'
        top_left_2_body_meas_type: str = 'X',
) -> stim.Circuit:
    """Generates common circuits.

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
        """
    params = CircuitGenParameters(
        rounds=rounds,
        distance=distance,
        prep_4112_params = prep_4112_params,
        # x_distance=x_distance,
        # z_distance=z_distance,
        after_clifford_depolarization=after_clifford_depolarization,
        before_round_data_depolarization=before_round_data_depolarization,
        before_measure_flip_probability=before_measure_flip_probability,
        after_reset_flip_probability=after_reset_flip_probability,
        # exclude_other_basis_detectors=exclude_other_basis_detectors,
        qubit_initialization_pattern = qubit_initialization_pattern,
        top_left_2_body_meas_type = top_left_2_body_meas_type,
    )  # this is a tuple of all the required parameters
    # prep_4112_circuit: the 4112 state prep circuit, given in the following 

    circuit_4112_to_surface_expansion, num_ideal_detectors = generate_4112_to_surface_expansion_circuit_from_params(params)
    return (prep_4112_circuit + circuit_4112_to_surface_expansion, num_ideal_detectors)


# generate the overal circuit
def overall_circuit_generation(p_dep, basis, angle, distance, rounds):
    circ_0, prep_4112_params = StatePrep4112(p_dep, basis, angle)

    # To maximize the number of detectors, we should want
    top_left_2_body_meas_type = prep_4112_params["gauge_basis"]
    if prep_4112_params["gauge_basis"] == 'X':
        from_4112_to_coord = {prep_4112_params["index_to_label_4112"][0]: 1 + 1j, prep_4112_params["index_to_label_4112"][1]: 1 + 3j, prep_4112_params["index_to_label_4112"][2]: 3 + 1j, prep_4112_params["index_to_label_4112"][3]: 3 + 3j}
    elif prep_4112_params["gauge_basis"] == 'Z':
        from_4112_to_coord = {prep_4112_params["index_to_label_4112"][0]: 1 + 1j, prep_4112_params["index_to_label_4112"][2]: 1 + 3j, prep_4112_params["index_to_label_4112"][1]: 3 + 1j, prep_4112_params["index_to_label_4112"][3]: 3 + 3j}

    # top_left_2_body_meas_type = prep_4112_params["gauge_basis"]
    # if prep_4112_params["gauge_basis"] == 'X':
    #     from_4112_to_coord = {prep_4112_params["index_to_label_4112"][0]: 1 + 1j, prep_4112_params["index_to_label_4112"][1]: 3 + 1j, prep_4112_params["index_to_label_4112"][2]: 1 + 3j, prep_4112_params["index_to_label_4112"][3]: 3 + 3j}
    # elif prep_4112_params["gauge_basis"] == 'Z':
    #     from_4112_to_coord = {prep_4112_params["index_to_label_4112"][0]: 1 + 1j, prep_4112_params["index_to_label_4112"][1]: 1 + 3j, prep_4112_params["index_to_label_4112"][2]: 3 + 1j, prep_4112_params["index_to_label_4112"][3]: 3 + 3j}

    prep_4112_params["from_4112_to_coord"] = from_4112_to_coord   # save the coordinate for the 4 [4,1,1,2] data qubits

    circ_1, num_ideal_detectors = complete_circuit(
            prep_4112_circuit = circ_0,
            prep_4112_params = prep_4112_params,
            rounds = rounds,
            distance = distance,
            # after_clifford_depolarization = prep_4112_params["p_dep"],
            # before_round_data_depolarization = prep_4112_params["p_dep"],
            # before_measure_flip_probability = prep_4112_params["p_dep"],
            # after_reset_flip_probability = prep_4112_params["p_dep"],
            after_clifford_depolarization = prep_4112_params["p_dep"],
            before_round_data_depolarization = prep_4112_params["p_dep"],
            before_measure_flip_probability = prep_4112_params["p_dep"],
            after_reset_flip_probability = prep_4112_params["p_dep"],
            qubit_initialization_pattern = 'asym',
            top_left_2_body_meas_type = top_left_2_body_meas_type)
    return circ_1, num_ideal_detectors






#################################################################################################
#! reference circuit used to generate ideal QEC matching graph


## function to connect [4,1,1,2] circuit with the later surface code expansion circuit
def generate_4112_to_surface_expansion_circuit_from_params_reference(
        params: CircuitGenParameters,
) -> stim.Circuit:
    
    if params.prep_4112_params['prepare_logical_basis'] == 'X':
        is_memory_x = True
    elif params.prep_4112_params['prepare_logical_basis'] == 'Z':
        is_memory_x = False
    else:
        is_memory_x = True

    x_distance = params.distance
    z_distance = params.distance

    # Place data qubits: specified by the complex-valued coordinates q
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
    x_measure_coords: Set[complex] = set()
    z_measure_coords: Set[complex] = set()
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
            else:
                z_measure_coords.add(q)

    # Define interaction orders so that hook errors run against the error grain instead of with it.
    z_order: List[complex] = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    x_order: List[complex] = [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]

    # convert the coordinates to indicies
    # we remark that, many redundant q is introduced to keep the coord->index mapping simple
    def coord_to_index(q: complex, params: dict=None) -> int:
        q = q - math.fmod(q.real, 2) * 1j
        padded_qubit_nbr = params.prep_4112_params["total_qubit_nbr"]
        return int(q.real + q.imag * (z_distance + 0.5)) + padded_qubit_nbr

    # based on the requirement on the top left 2-body check, determine whether to flip the surface code
    # data qubit coordinates keep invariant: change the x,z parity check & observable definition
    if params.top_left_2_body_meas_type == 'Z':
        x_observable, z_observable = util.swap_sets(x_observable, z_observable)
        x_measure_coords, z_measure_coords = util.swap_sets(x_measure_coords, z_measure_coords)
        x_order, z_order = util.swap_sets(x_order, z_order)
    
    # Here, we return the circuit and number of ideal detectors generated from the function `finish_surface_code_circuit` below
    return finish_surface_code_circuit_reference(
        coord_to_index,
        data_coords,
        x_measure_coords,
        z_measure_coords,
        params,
        x_order,
        z_order,
        x_observable,
        z_observable,
        is_memory_x
    )


## detailed surface code circuit
# We first perform 3 rounds of noisy QED, then add an extra round of ideal QEC, then determine the x/y measurement based on the phi value
def finish_surface_code_circuit_reference(
        coord_to_index: Callable[[complex, dict], int],
        data_coords: Set[complex],
        x_measure_coords: Set[complex],
        z_measure_coords: Set[complex],
        params: CircuitGenParameters,
        x_order: List[complex],
        z_order: List[complex],
        x_observable: List[complex],
        z_observable: List[complex],
        is_memory_x: bool,   # determine it is either x or z memory
        *,
        exclude_other_basis_detectors: bool = False,
        wraparound_length: Optional[int] = None
) -> tuple[stim.Circuit, int]:
    if params.rounds < 1:
        raise ValueError("Need rounds >= 1")
    if params.distance is not None and params.distance < 2:
        raise ValueError("Need a distance >= 2")
    if params.x_distance is not None and (params.x_distance < 2 or
                                          params.z_distance < 2):
        raise ValueError("Need a distance >= 2")

    chosen_basis_observable = x_observable if is_memory_x else z_observable   ### need to be revised
    chosen_basis_measure_coords = x_measure_coords if is_memory_x else z_measure_coords

    # Index the measurement qubits and data qubits: store them to dictionaries.
    ###  p: coordinate ;  q: index   
    p2q: Dict[complex, int] = {}
    for q in data_coords:
        p2q[q] = coord_to_index(q, params)

    for q in x_measure_coords:
        p2q[q] = coord_to_index(q, params)

    for q in z_measure_coords:
        p2q[q] = coord_to_index(q, params)

    q2p: Dict[int, complex] = {v: k for k, v in p2q.items()}   # define a dictionary from index to coord


    data_qubits = [p2q[p] for p in data_coords]
    x_logical_qubits = [p2q[p] for p in x_observable]
    z_logical_qubits = [p2q[p] for p in z_observable]
    measurement_qubits = [p2q[p] for p in x_measure_coords]
    measurement_qubits += [p2q[p] for p in z_measure_coords]
    x_measurement_qubits = [p2q[p] for p in x_measure_coords]

    # set the surface code initialization
    # the reference circuit: directly prepare the logical 0 or + state
    initial_data_z_coord = []
    initial_data_x_coord = []
    if params.prep_4112_params['prepare_logical_basis'] == 'X':
        # if the basis choice is X, initialize in X basis
        for q in data_coords:
            initial_data_x_coord.append(q)
    elif params.prep_4112_params['prepare_logical_basis'] == 'Z':
        # otherwise we initialize in Z basis
        for q in data_coords:
            initial_data_z_coord.append(q)
    else: # params.prep_4112_params['prepare_logical_basis'] == 'Y':
        for q in data_coords:
            initial_data_x_coord.append(q)
        # logical S gate is implemented later

    # List to record the location of 4112 data qubits: (order is not important, this is only for the detector assignment usage)
    data_4112_qubits_coord = [1 + 1j, 1 + 3j, 3 + 1j, 3 + 3j]

    [initial_data_x, initial_data_z] = util.map_list_of_list([initial_data_x_coord, initial_data_z_coord], coord_to_index, params)

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
        data_coord_to_order[q2p[q]] = len(data_coord_to_order)   # this dictionary will provide plain-ordered sequence of all the data qubits: start from 1
    for q in measurement_qubits:
        measure_coord_to_order[q2p[q]] = len(measure_coord_to_order)  # this dictionary will provide plain-ordered sequence of all the measurements

    # List out CNOT gate targets using given interaction orders.
    # CNOT gates are specified by the complex-valued coordinates
    cnot_targets: List[List[int]] = [[], [], [], []]  # list it by 4 time bins
    for k in range(4):
        for measure in sorted(x_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + x_order[k]
            if data in p2q:
                cnot_targets[k].append(p2q[measure])
                cnot_targets[k].append(p2q[data])
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                cnot_targets[k].append(p2q[measure])
                cnot_targets[k].append(p2q[data_wrapped])

        for measure in sorted(z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + z_order[k]
            if data in p2q:
                cnot_targets[k].append(p2q[data])
                cnot_targets[k].append(p2q[measure])
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                cnot_targets[k].append(p2q[data_wrapped])
                cnot_targets[k].append(p2q[measure])

    # Build the repeated actions that make up the surface code cycle
    # For reference circuit: set it to be noiseless!!
    cycle_actions_ideal = stim.Circuit()
    params.append_begin_round_tick(cycle_actions_ideal, data_qubits, if_noiseless = True)
    params.append_unitary_1(cycle_actions_ideal, "H", x_measurement_qubits, if_noiseless = True)
    for targets in cnot_targets:
        cycle_actions_ideal.append_operation("TICK", [])
        params.append_unitary_2(cycle_actions_ideal, "CNOT", targets, if_noiseless = True)
        # print(targets)
    cycle_actions_ideal.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions_ideal, "H", x_measurement_qubits, if_noiseless = True)
    cycle_actions_ideal.append_operation("TICK", [])
    params.append_measure_reset(cycle_actions_ideal, measurement_qubits, if_noiseless = True)


    # Build the start of the circuit, getting a state that's ready to cycle
    # For the reference circuit: we do not set any detector for the first round of QEC
    # We do not perform 4112 qubit to surface qubit swap
    
    # assign coordinates on the surface code qubits
    # real part is the first axis, imaginary part is the second axis
    head = stim.Circuit()
    for k, v in sorted(q2p.items()):
        head.append_operation("QUBIT_COORDS", [k], [v.real, v.imag])

    # Ideal initialization of the surface code (4112 qubits not included)
    # For the reference circuit: no noise!
    params.append_reset(head, initial_data_x, "X", if_noiseless = True)
    params.append_reset(head, initial_data_z, "Z", if_noiseless = True)
    params.append_reset(head, measurement_qubits, if_noiseless = True)


    # first round of surface code syndrome check for code expansion
    head += cycle_actions_ideal
    # remark: no detector at the end!
    

    # Build the repeated body of the circuit, including the detectors comparing to previous cycles
    # for the reference circuit: this is a selective S gate based on the angle
    body = stim.Circuit()

    if params.prep_4112_params['prepare_logical_basis'] == 'Y':
        # when the basis is Y, apply logical S gate
        # z_logical_qubits = [p2q[p] for p in z_observable]
        q0 = p2q[1+1j]
        if q0 != z_logical_qubits[0]:
            raise Exception("Wrong first observable qubit index!")
        
        for q in reversed(z_logical_qubits):
            if q != q0: 
                body.append("CNOT", [q, q0])

        body.append("S", q0)

        for q in z_logical_qubits:
            if q != q0:
                body.append("CNOT", [q, q0])

    if params.prep_4112_params['angle'] == 1:
        # when the angle is np.pi/4, apply logical S gate
        # z_logical_qubits = [p2q[p] for p in z_observable]

        q0 = p2q[1+1j]
        if q0 != z_logical_qubits[0]:
            raise Exception("Wrong first observable qubit index!")
        
        for q in reversed(z_logical_qubits):
            if q != q0: 
                body.append("CNOT", [q, q0])

        body.append("S", q0)

        for q in z_logical_qubits:
            if q != q0:
                body.append("CNOT", [q, q0])

    

    # We append the single qubit depolarizing noise for the data qubits
    body.append("DEPOLARIZE1", [q for q in data_qubits], params.prep_4112_params["p_dep"])
    body.append("TICK")


    # body = cycle_actions.copy()
    # m = len(measurement_qubits)
    # body.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    # for m_index in measurement_qubits:
    #     m_coord = q2p[m_index]
    #     k = len(measurement_qubits) - measure_coord_to_order[m_coord] - 1
    #     if not exclude_other_basis_detectors or m_coord in chosen_basis_measure_coords:
    #         body.append_operation(
    #             "DETECTOR",
    #             [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
    #             [m_coord.real, m_coord.imag, 0.0]
    #         )


    #! Build the end of the circuit, getting out of the cycle state and terminating.
    # first perform one round of ideal QEC (not QED!); then measurement X,Y,Z in a non-transversal way (logical observable)
    # the detector here are all used for QEC! need a counter for the usage of later decoding: descriminate QED and QEC

    # In particular, the data measurements create detectors that have to be handled special.
    # Also, the tail is responsible for identifying the logical observable.
    

    # Build the ideal QEC, add detectors
    ideal_QEC = cycle_actions_ideal.copy()
    m = len(measurement_qubits)
    num_detectors_ideal = 0
    ideal_QEC.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    for m_index in measurement_qubits:
        m_coord = q2p[m_index]
        k = len(measurement_qubits) - measure_coord_to_order[m_coord] - 1
        # recall that measure_coord_to_order dictionary provides the plain order!
        if not exclude_other_basis_detectors or m_coord in chosen_basis_measure_coords:
            ideal_QEC.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                [m_coord.real, m_coord.imag, 0.0]
            )
            # check the parity of this detector with the former detector at the same location
            num_detectors_ideal = num_detectors_ideal + 1
        
    # Build the logical measurement and observables
    tail_logical_measure = stim.Circuit()
    if params.prep_4112_params['prepare_logical_basis'] == 'X':  # the initial state is prepared to logical +
        if params.prep_4112_params['angle'] == 0:   # phi = 0 
            # in this case, we measure logical X on the edge of logical X qubits
            # recall that 'z_observable' and 'x_observable' store the complex location of the logical data qubits!
            # 'z_logical_qubits' and 'x_logical_qubits' store the index of the qubits
            params.append_measure(tail_logical_measure, x_logical_qubits, "X", if_noiseless=True) 

            # add X observable
            tail_logical_measure.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-len(x_observable),0)], 0.0)
        else: # phi = np.pi/4
            # in this case, we measure logical Y on the edge of logical X and Z qubits
            # Here, we assume that the overlapped data qubit for logical X and Z is the one with coordinate 1+1j
            params.append_measure(tail_logical_measure, x_logical_qubits[1:], "X", if_noiseless=True)
            params.append_measure(tail_logical_measure, z_logical_qubits[1:], "Z", if_noiseless=True)
            params.append_measure(tail_logical_measure, p2q[1+1j], "Y", if_noiseless=True)

            # add logical Y observable
            tail_logical_measure.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-(2*(len(x_observable)-1)+1),0)], 0.0)
    elif  params.prep_4112_params['prepare_logical_basis'] == 'Z':  # the initial state is prepared to logical 0
        # regardless of the phi angle, here we measure logical Z
        params.append_measure(tail_logical_measure, z_logical_qubits, "Z", if_noiseless=True)
        # add X observable
        tail_logical_measure.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-len(x_observable),0)], 0.0)
    else: # params.prep_4112_params['prepare_logical_basis'] == 'Y' # the initial state is prepared to logical +i
        if params.prep_4112_params['angle'] == 1:   # phi = np.pi/4 
            # in this case, we measure logical X on the edge of logical X qubits
            # recall that 'z_observable' and 'x_observable' store the complex location of the logical data qubits!
            # 'z_logical_qubits' and 'x_logical_qubits' store the index of the qubits
            params.append_measure(tail_logical_measure, x_logical_qubits, "X", if_noiseless=True) 

            # add X observable
            tail_logical_measure.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-len(x_observable),0)], 1.0)
        else: # phi = 0
            # in this case, we measure logical Y on the edge of logical X and Z qubits
            # Here, we assume that the overlapped data qubit for logical X and Z is the one with coordinate 1+1j
            params.append_measure(tail_logical_measure, x_logical_qubits[1:], "X", if_noiseless=True)
            params.append_measure(tail_logical_measure, z_logical_qubits[1:], "Z", if_noiseless=True)
            params.append_measure(tail_logical_measure, p2q[1+1j], "Y", if_noiseless=True)

            # add logical Y observable
            tail_logical_measure.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in range(-(2*(len(x_observable)-1)+1),0)], 0.0)

    """
    # Here, we assume ideal measurements to isolate out state preparation errors.
    tail = stim.Circuit()
    params.append_measure(tail, data_qubits, "ZX"[is_memory_x], if_noiseless=True)  ### THIS BECOMES INVALID WHEN phi=np.pi/4!
    # Detectors
    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        detectors: List[int] = []
        for delta in z_order:
            data = measure + delta
            if data in p2q:
                detectors.append(-len(data_qubits) + data_coord_to_order[data])
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                detectors.append(-len(data_qubits) + data_coord_to_order[data_wrapped])
        detectors.append(-len(data_qubits) - len(measurement_qubits) + measure_coord_to_order[measure])
        detectors.sort(reverse=True)
        tail.append_operation("DETECTOR", [stim.target_rec(x) for x in detectors], [measure.real, measure.imag, 1.0])

    # Logical observable
    obs_inc: List[int] = []
    for q in chosen_basis_observable:
        obs_inc.append(-len(data_qubits) + data_coord_to_order[q])
    obs_inc.sort(reverse=True)
    tail.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in obs_inc], 0.0)

    """
    
    # Combine to form final circuit.
    # return head + body * (params.rounds - 1) + tail
    # return head + body * (params.rounds - 1) + tail_logical_measure
    return (head + body + ideal_QEC + tail_logical_measure, num_detectors_ideal)



# function to input prep_4112_circuit, and append the surface code circuits
def complete_circuit_reference(
        prep_4112_circuit: stim.Circuit,  # circuit generated from the function StatePrep4112()
        prep_4112_params: dict,
        rounds: int,
        distance: int = None,
        # x_distance: int = None,
        # z_distance: int = None,
        after_clifford_depolarization: float = 0.0,
        before_round_data_depolarization: float = 0.0,
        before_measure_flip_probability: float = 0.0,
        after_reset_flip_probability: float = 0.0,
        # exclude_other_basis_detectors: bool = False,
        qubit_initialization_pattern: str = 'asym',  # 'asym' or 'sym'
        top_left_2_body_meas_type: str = 'X',
) -> stim.Circuit:
    """Generates common circuits.

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
        """
    params = CircuitGenParameters(
        rounds=rounds,
        distance=distance,
        prep_4112_params = prep_4112_params,
        # x_distance=x_distance,
        # z_distance=z_distance,
        after_clifford_depolarization=after_clifford_depolarization,
        before_round_data_depolarization=before_round_data_depolarization,
        before_measure_flip_probability=before_measure_flip_probability,
        after_reset_flip_probability=after_reset_flip_probability,
        # exclude_other_basis_detectors=exclude_other_basis_detectors,
        qubit_initialization_pattern = qubit_initialization_pattern,
        top_left_2_body_meas_type = top_left_2_body_meas_type,
    )  # this is a tuple of all the required parameters
    # prep_4112_circuit: the 4112 state prep circuit, given in the following 

    circuit_4112_to_surface_expansion_reference, num_ideal_detectors = generate_4112_to_surface_expansion_circuit_from_params_reference(params)
    # return (prep_4112_circuit + circuit_4112_to_surface_expansion, num_ideal_detectors)
    return (circuit_4112_to_surface_expansion_reference, num_ideal_detectors)  # no need to add prep_4112_circuit for the reference circuit!!


# generate the overal circuit
def overall_circuit_generation_reference(p_dep, basis, angle, distance, rounds):
    circ_0, prep_4112_params = StatePrep4112(p_dep, basis, angle)

    # To maximize the number of detectors, we should want
    top_left_2_body_meas_type = prep_4112_params["gauge_basis"]
    if prep_4112_params["gauge_basis"] == 'X':
        from_4112_to_coord = {prep_4112_params["index_to_label_4112"][0]: 1 + 1j, prep_4112_params["index_to_label_4112"][1]: 1 + 3j, prep_4112_params["index_to_label_4112"][2]: 3 + 1j, prep_4112_params["index_to_label_4112"][3]: 3 + 3j}
    elif prep_4112_params["gauge_basis"] == 'Z':
        from_4112_to_coord = {prep_4112_params["index_to_label_4112"][0]: 1 + 1j, prep_4112_params["index_to_label_4112"][2]: 1 + 3j, prep_4112_params["index_to_label_4112"][1]: 3 + 1j, prep_4112_params["index_to_label_4112"][3]: 3 + 3j}

    # top_left_2_body_meas_type = prep_4112_params["gauge_basis"]
    # if prep_4112_params["gauge_basis"] == 'X':
    #     from_4112_to_coord = {prep_4112_params["index_to_label_4112"][0]: 1 + 1j, prep_4112_params["index_to_label_4112"][1]: 3 + 1j, prep_4112_params["index_to_label_4112"][2]: 1 + 3j, prep_4112_params["index_to_label_4112"][3]: 3 + 3j}
    # elif prep_4112_params["gauge_basis"] == 'Z':
    #     from_4112_to_coord = {prep_4112_params["index_to_label_4112"][0]: 1 + 1j, prep_4112_params["index_to_label_4112"][1]: 1 + 3j, prep_4112_params["index_to_label_4112"][2]: 3 + 1j, prep_4112_params["index_to_label_4112"][3]: 3 + 3j}

    prep_4112_params["from_4112_to_coord"] = from_4112_to_coord   # save the coordinate for the 4 [4,1,1,2] data qubits

    circ_1_reference, num_ideal_detectors = complete_circuit_reference(
            prep_4112_circuit = circ_0,
            prep_4112_params = prep_4112_params,
            rounds = rounds,
            distance = distance,
            # after_clifford_depolarization = prep_4112_params["p_dep"],
            # before_round_data_depolarization = prep_4112_params["p_dep"],
            # before_measure_flip_probability = prep_4112_params["p_dep"],
            # after_reset_flip_probability = prep_4112_params["p_dep"],
            after_clifford_depolarization = prep_4112_params["p_dep"],
            before_round_data_depolarization = 0,
            before_measure_flip_probability = 0,
            after_reset_flip_probability = prep_4112_params["p_dep"],
            qubit_initialization_pattern = 'asym',
            top_left_2_body_meas_type = top_left_2_body_meas_type)
    return circ_1_reference, num_ideal_detectors
