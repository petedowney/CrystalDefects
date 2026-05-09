import os
import sys
import time
import random
import itertools
import pickle
from collections import defaultdict
import concurrent.futures

import numpy as np
from mpi4py import MPI

import util

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(description="Distributed Tight-Binding Training")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--sample-limit", type=int, default=1000, help="Total samples across the cluster")
parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size per worker")
args = parser.parse_args()

LEARNING_RATE = 0.05
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
DECAY_LENGTH = 1.0
PERTURBATION = 1e-4
DISTANCE_THRESHOLD = 3
BASE_PATH = "./all_dos_data_full/all_dos_data_full"
NUM_SAMPLES_LIMIT = args.sample_limit

# -------------------------------------------------------------------------
# Helper Functions (Extracted from Notebook)
# -------------------------------------------------------------------------
el_orbs_cache = {}

def precompute_basis(elements):
    atom_orbitals = [el_orbs_cache[el] for el in elements]
    basis_sizes = [len(orbs) for orbs in atom_orbitals]
    total_basis = sum(basis_sizes)
    basis_idx = np.insert(np.cumsum(basis_sizes), 0, 0)[:-1]
    return atom_orbitals, total_basis, basis_idx

def build_adj(file_tuple, threshold_distance=3.0):
    poscar_path = file_tuple[2]
    with open(poscar_path, 'r') as f:
        lines = f.readlines()
        
    scale = float(lines[1].strip())
    lattice = np.array([
        [float(x) for x in lines[2].split()],
        [float(x) for x in lines[3].split()],
        [float(x) for x in lines[4].split()]
    ]) * scale
        
    element_names = lines[5].split()
    element_counts = [int(x) for x in lines[6].split()]
    
    elements = []
    for name, count in zip(element_names, element_counts):
        elements.extend([name] * count)
        
    coord_idx = 0
    while coord_idx < len(lines) and not lines[coord_idx].strip() in ["Direct", "Cartesian"]:
        coord_idx += 1
        
    coord_type = lines[coord_idx].strip()
    coord_idx += 1
    
    coords = []
    while coord_idx < len(lines):
        line = lines[coord_idx].strip()
        if not line:
            break
        parts = line.split()
        if len(parts) >= 3:
            coord = [float(parts[0]), float(parts[1]), float(parts[2])]
            coords.append(coord)
        coord_idx += 1
        
    coords = np.array(coords)
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    
    if coord_type == "Direct":
        diffs = (diffs + 0.5) % 1.0 - 0.5
        diffs = np.dot(diffs, lattice)
            
    dist_matrix = np.linalg.norm(diffs, axis=-1)
    adj_matrix = (dist_matrix < threshold_distance).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    
    return adj_matrix, dist_matrix, elements

def build_hamiltonian(adj_matrix, dist_matrix, atom_orbitals, total_basis, basis_idx, on_site_energies, hopping_energies, decay_length=1.0):
    decay_matrix = np.zeros_like(dist_matrix)
    mask = adj_matrix == 1
    decay_matrix[mask] = np.exp(-dist_matrix[mask] / decay_length)
    
    hamiltonian = np.zeros((total_basis, total_basis), dtype=np.float64)
    num_atoms = len(atom_orbitals)
    
    for i in range(num_atoms):
        orbs_i = atom_orbitals[i]
        start_i = basis_idx[i]
        
        for idx, orb in enumerate(orbs_i):
            hamiltonian[start_i + idx, start_i + idx] = on_site_energies[orb]
            
        connected_j = np.where(adj_matrix[i, i+1:] == 1)[0] + i + 1
        
        for j in connected_j:
            d_factor = decay_matrix[i, j]
            orbs_j = atom_orbitals[j]
            start_j = basis_idx[j]
            
            for idx_i, orb_i in enumerate(orbs_i):
                h_row = start_i + idx_i
                
                for idx_j, orb_j in enumerate(orbs_j):
                    t0 = hopping_energies.get((orb_i, orb_j))
                    if t0 is None:
                        t0 = hopping_energies.get((orb_j, orb_i), 0.0)
                        
                    if t0 != 0.0:
                        t_ij = t0 * d_factor
                        h_col = start_j + idx_j
                        
                        hamiltonian[h_row, h_col] = t_ij
                        hamiltonian[h_col, h_row] = t_ij
    
    return hamiltonian

def compute_mse(energies, dos_energies):
    min_len = min(len(energies), len(dos_energies))
    diffs = energies[:min_len] - dos_energies[:min_len]
    mse = np.mean(diffs ** 2)
    return mse

def compute_onsite_perturbation_grad(energies, eigen_functions, dos_energies, param_indices, base_mse, perturbation=1e-4):
    delta_energies = np.sum(np.abs(eigen_functions[param_indices, :]) ** 2, axis=0) * perturbation
    new_energies = energies + delta_energies
    new_mse = compute_mse(new_energies, dos_energies)
    return (new_mse - base_mse) / perturbation

def compute_hopping_perturbation_grad(energies, eigen_functions, dos_energies, row_indices, col_indices, decay_factors, base_mse, perturbation=1e-4):
    shifts = 2 * eigen_functions[row_indices, :] * eigen_functions[col_indices, :] * decay_factors[:, np.newaxis]
    delta_energies = np.sum(shifts, axis=0) * perturbation
    new_energies = energies + delta_energies
    new_mse = compute_mse(new_energies, dos_energies)
    return (new_mse - base_mse) / perturbation

def precompute_onsite_topology(atom_orbitals, basis_idx):
    param_to_indices = {}
    for i, orbs in enumerate(atom_orbitals):
        start_i = basis_idx[i]
        for idx, orb in enumerate(orbs):
            if orb not in param_to_indices:
                param_to_indices[orb] = []
            param_to_indices[orb].append(start_i + idx)
    return {orb: np.array(indices) for orb, indices in param_to_indices.items()}

def precompute_hopping_topology(adj_matrix, dist_matrix, atom_orbitals, basis_idx, decay_length=1.0):
    decay_matrix = np.zeros_like(dist_matrix)
    mask = adj_matrix == 1
    decay_matrix[mask] = np.exp(-dist_matrix[mask] / decay_length)
    
    param_to_instances = {}
    num_atoms = len(atom_orbitals)
    
    for i in range(num_atoms):
        orbs_i = atom_orbitals[i]
        start_i = basis_idx[i]
        connected_j = np.where(adj_matrix[i, i+1:] == 1)[0] + i + 1
        
        for j in connected_j:
            d_factor = decay_matrix[i, j]
            orbs_j = atom_orbitals[j]
            start_j = basis_idx[j]
            
            for idx_i, orb_i in enumerate(orbs_i):
                h_row = start_i + idx_i
                for idx_j, orb_j in enumerate(orbs_j):
                    h_col = start_j + idx_j
                    param = tuple(sorted([orb_i, orb_j]))
                    if param not in param_to_instances:
                        param_to_instances[param] = {"rows": [], "cols": [], "decays": []}
                    param_to_instances[param]["rows"].append(h_row)
                    param_to_instances[param]["cols"].append(h_col)
                    param_to_instances[param]["decays"].append(d_factor)
                    
    for param in param_to_instances:
        param_to_instances[param]["rows"] = np.array(param_to_instances[param]["rows"])
        param_to_instances[param]["cols"] = np.array(param_to_instances[param]["cols"])
        param_to_instances[param]["decays"] = np.array(param_to_instances[param]["decays"])
        
    return param_to_instances

def fast_onsite_grads(energies, eigen_functions, dos_energies, onsite_topology, base_mse, perturbation=1e-4):
    grads = {}
    for orb, indices in onsite_topology.items():
        grads[orb] = compute_onsite_perturbation_grad(
            energies, eigen_functions, dos_energies, indices, base_mse, perturbation
        )
    return grads

def fast_hopping_grads(energies, eigen_functions, dos_energies, hopping_topology, base_mse, perturbation=1e-4):
    grads = {}
    for param, data in hopping_topology.items():
        grad = compute_hopping_perturbation_grad(
            energies, eigen_functions, dos_energies, 
            data["rows"], data["cols"], data["decays"], 
            base_mse, perturbation
        )
        grads[param] = grad
    return grads

# -------------------------------------------------------------------------
# MPI Distributed Flow
# -------------------------------------------------------------------------
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def main():
    # Keep NumPy threads restricted so we don't oversubscribe the cluster nodes mapped to ranks
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    #print(f"rank={MPI.COMM_WORLD.Get_rank()} size={MPI.COMM_WORLD.Get_size()}", flush=True)

    if rank == 0:
        f = open('hopping_terminal.txt', 'a')
        sys.stdout = Tee(sys.stdout, f)

    # We must init properties to use elements and valence
    util.init_properties("properties.txt")
    
    # Populate orbital cache
    for el in util.getPeriodicTableDict().keys():
        try:
            el_orbs_cache[el] = [f"{el}_{orbital}_{n+1}" for n, orbital in enumerate(util.getValenceInteractions(el))]
        except KeyError:
            pass

    on_site_energies = None
    hopping_energies = None
    file_pairs = []
    
    # Setup parameters and collect data ONLY on Rank 0
    if rank == 0:
        print("Gathering files...", flush=True)
        
        spin_neutral_pkl = 'filtered_spin_neutral.pkl'
        if not os.path.exists(spin_neutral_pkl):
            spin_neutral_pkl = 'tight_binding_model/filtered_spin_neutral.pkl'
            
        spin_neutral_dirs = []
        if os.path.exists(spin_neutral_pkl):
            with open(spin_neutral_pkl, 'rb') as f_pkl:
                spin_neutral_dirs = pickle.load(f_pkl)
            print(f"Loaded {len(spin_neutral_dirs)} directories from {spin_neutral_pkl}", flush=True)
        else:
            print(f"ERROR: {spin_neutral_pkl} not found. Please run the filter script first.", flush=True)
            sys.exit(1)

        total_file_pairs = []
        for root in spin_neutral_dirs:
            
            root = root.replace("all_dos_data_full/", "", 1)           

            dos_dw = os.path.join(root, 'dos_DW')
            dos_up = os.path.join(root, 'dos_UP')
            poscar = os.path.join(root, 'POSCAR')
            if os.path.exists(dos_dw) and os.path.exists(dos_up) and os.path.exists(poscar):
                total_file_pairs.append((dos_dw, dos_up, poscar))

        # Limit to the configured count max and shuffle
        random.shuffle(total_file_pairs)
        if NUM_SAMPLES_LIMIT != -1:
            total_file_pairs = total_file_pairs[:NUM_SAMPLES_LIMIT]
        print(f"Total valid file pairs available for training: {len(total_file_pairs)}", flush=True)

        # Initialize global parameters (Load trained weights if available, else initialize random)
        full_element_list = list(util.getPeriodicTableDict().keys())
        full_list = [] 
        for el in full_element_list:
            for n, orbital in enumerate(util.getValenceInteractions(el)):
                full_list.append(f"{el}_{orbital}_{n+1}")

        if os.path.exists('trained_onsite_energies.pkl') and os.path.exists('trained_hopping_energies.pkl'):
            print("Loading pre-trained model weights from .pkl files to use as seeds...", flush=True)
            with open('trained_onsite_energies.pkl', 'rb') as f:
                on_site_energies = pickle.load(f)
            with open('trained_hopping_energies.pkl', 'rb') as f:
                hopping_energies = pickle.load(f)
        else:
            print("No pre-trained weights found, initializing with uniform random weights...", flush=True)
            on_site_energies = {el: random.uniform(0.1, 10.0) for el in full_list}
            hopping_energies = {}
            for el1, el2 in itertools.combinations_with_replacement(full_list, 2):
                energy = random.uniform(0.1, 10.0)
                hopping_energies[(el1, el2)] = energy
                if el1 != el2:
                    hopping_energies[(el2, el1)] = energy

        # Partition data for workers
        chunk_size = len(total_file_pairs) // size + 1
        partitioned_pairs = [total_file_pairs[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
    else:
        partitioned_pairs = None

    # Sync initial parameters
    on_site_energies = comm.bcast(on_site_energies, root=0)
    hopping_energies = comm.bcast(hopping_energies, root=0)
    
    # Send pairs to workers
    local_file_pairs = comm.scatter(partitioned_pairs, root=0)

    # All workers precompute topologies for purely their slice
    if rank == 0:
        print("Workers precomputing topology data...", flush=True)
        
    local_precomputed_data = []
    for fp in local_file_pairs:
        # sometimes dos wont parse correctly skipping for now but bad practice
        try:
            with open(fp[0], 'r') as f:
                dos_energies = np.array([float(line.split()[1]) for line in f if line.strip()])
            
            adj_mat, dist_mat, elems = build_adj(fp, threshold_distance=DISTANCE_THRESHOLD)
            atom_orbs, t_basis, b_idx = precompute_basis(elems)
            onsite_top = precompute_onsite_topology(atom_orbs, b_idx)
            hopping_top = precompute_hopping_topology(adj_mat, dist_mat, atom_orbs, b_idx, DECAY_LENGTH)
        
            local_precomputed_data.append({
                'dos_energies': dos_energies,
                'adj_matrix': adj_mat,
                'dist_matrix': dist_mat,
                'atom_orbitals': atom_orbs,
                'total_basis': t_basis,
                'basis_idx': b_idx,
                'onsite_topology': onsite_top,
                'hopping_topology': hopping_top,
            })
        except:
            continue

    # Prepare vectorization maps for allreduce gradients
    # Since keys are the same, order is identical everywhere as broadcasted
    onsite_keys = list(on_site_energies.keys())
    hopping_keys = list(hopping_energies.keys())
    
    if rank == 0:
        print(f"Starting MPI training over {EPOCHS} epochs with {size} ranks. Batch size per rank: {BATCH_SIZE}", flush=True)
        
    global_samples_count = comm.allreduce(len(local_precomputed_data), op=MPI.SUM)
    local_samples_count = len(local_precomputed_data)
    max_local_samples = comm.allreduce(local_samples_count, op=MPI.MAX)
    num_batches = (max_local_samples + BATCH_SIZE - 1) // BATCH_SIZE if max_local_samples > 0 else 0

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # SGD logic: Shuffle local data per epoch
        random.shuffle(local_precomputed_data)
        
        epoch_mse = 0.0
        
        for b in range(num_batches):
            # Select batch for current worker
            start_idx = b * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, local_samples_count)
            batch_data = local_precomputed_data[start_idx:end_idx]
            
            # Gradients accumulators for this mini-batch
            local_onsite_grad_arr = np.zeros(len(onsite_keys), dtype=np.float64)
            local_hopping_grad_arr = np.zeros(len(hopping_keys), dtype=np.float64)
            local_batch_mse = 0.0

            for data in batch_data:
                # 1. Forward Pass
                hamiltonian = build_hamiltonian(
                    data['adj_matrix'], data['dist_matrix'], data['atom_orbitals'],
                    data['total_basis'], data['basis_idx'], 
                    on_site_energies, hopping_energies, DECAY_LENGTH
                )
                
                # 2. Diagonalize
                energies, eigen_functions = np.linalg.eigh(hamiltonian)
                
                # 3. MSE
                base_mse = compute_mse(energies, data['dos_energies'])
                local_batch_mse += base_mse
                
                # 4. Backwards Pass
                onsite_grads = fast_onsite_grads(
                    energies, eigen_functions, data['dos_energies'], 
                    data['onsite_topology'], base_mse, PERTURBATION
                )
                
                hopping_grads = fast_hopping_grads(
                    energies, eigen_functions, data['dos_energies'], 
                    data['hopping_topology'], base_mse, PERTURBATION
                )
                
                # Accumulate manually over the batch
                for idx, key in enumerate(onsite_keys):
                    if key in onsite_grads:
                        local_onsite_grad_arr[idx] += onsite_grads[key]
                        
                for idx, key in enumerate(hopping_keys):
                    if key in hopping_grads:
                        local_hopping_grad_arr[idx] += hopping_grads[key]

            # Aggregate gradients across all nodes for this mini-batch
            global_onsite_grad_arr = np.zeros_like(local_onsite_grad_arr)
            global_hopping_grad_arr = np.zeros_like(local_hopping_grad_arr)
            
            comm.Allreduce(local_onsite_grad_arr, global_onsite_grad_arr, op=MPI.SUM)
            comm.Allreduce(local_hopping_grad_arr, global_hopping_grad_arr, op=MPI.SUM)
            
            # Aggregate the exact size of this batch globally (some ranks might have fewer or 0 items in the last batch)
            global_batch_size = comm.allreduce(len(batch_data), op=MPI.SUM)
            
            # Aggregate MSE
            global_batch_mse = comm.allreduce(local_batch_mse, op=MPI.SUM)
            epoch_mse += global_batch_mse
            
            # Apply the synchronized gradients across all nodes identically if the batch wasn't fully empty globally
            if global_batch_size > 0:
                for idx, key in enumerate(onsite_keys):
                    # Normalizing by global batch size to accurately average the gradient
                    avg_grad = global_onsite_grad_arr[idx] / global_batch_size
                    on_site_energies[key] -= LEARNING_RATE * avg_grad
                    
                for idx, key in enumerate(hopping_keys):
                    avg_grad = global_hopping_grad_arr[idx] / global_batch_size
                    k1, k2 = key
                    hopping_energies[(k1, k2)] -= LEARNING_RATE * avg_grad
                    if k1 != k2:
                        hopping_energies[(k2, k1)] -= LEARNING_RATE * avg_grad

        # Log epoch 
        if rank == 0:
            avg_epoch_mse = epoch_mse / global_samples_count if global_samples_count > 0 else 0
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{EPOCHS} | Avg Global MSE: {avg_epoch_mse:.6f} | Time: {elapsed:.2f}s", flush=True)

    # Finishing and saving on Rank 0
    if rank == 0:
        print("Training finished. Saving global weights...", flush=True)
        with open('trained_onsite_energies_mpi.pkl', 'wb') as f:
            pickle.dump(on_site_energies, f)
            
        with open('trained_hopping_energies_mpi.pkl', 'wb') as f:
            pickle.dump(hopping_energies, f)
            
        print("Saved trained parameters to 'trained_onsite_energies_mpi.pkl' and 'trained_hopping_energies_mpi.pkl'", flush=True)

if __name__ == '__main__':
    main()
