import os
import pickle
import concurrent.futures
from collections import defaultdict

def validate_files(files_dict):
    """
    Validates that the DOS files are not empty and contain properly formatted 
    energy values on their first data line, matching the behavior in train_mpi.py.
    """
    try:
        # Check if files physically have data before opening
        if os.path.getsize(files_dict['dos_DW']) == 0 or os.path.getsize(files_dict['dos_UP']) == 0:
            return False
            
        # Just check the first non-empty line to ensure valid float conversion
        for key in ['dos_DW', 'dos_UP']:
            with open(files_dict[key], 'r') as f:
                for line in f:
                    if line.strip():
                        float(line.split()[1])
                        break
        return True
    except (ValueError, IndexError, OSError):
        return False

def main():
    # Attempt to locate the spin_neutral list
    pkl_path = "spin_neutral.pkl"
    if not os.path.exists(pkl_path):
        pkl_path = "../spin_neutral.pkl"
        if not os.path.exists(pkl_path):
            print("Error: Could not find spin_neutral.pkl.")
            return

    print(f"Loading directories from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        spin_neutral_dirs = pickle.load(f)
        
    print(f"Loaded {len(spin_neutral_dirs)} potential spin-neutral directories.")

    # Group required files
    file_groups = defaultdict(list)
    for root in spin_neutral_dirs:
        if os.path.exists(root):
            files = os.listdir(root)
            if all(req in files for req in ['dos_DW', 'dos_UP', 'POSCAR']):
                file_groups[root].append(('dos_DW', os.path.join(root, 'dos_DW')))
                file_groups[root].append(('dos_UP', os.path.join(root, 'dos_UP')))
                file_groups[root].append(('POSCAR', os.path.join(root, 'POSCAR')))

    print(f"Found {len(file_groups)} directories containing all required files (dos_DW, dos_UP, POSCAR).")
    print("Running validation pool...")

    valid_dirs = []
    
    # Process validations concurrently
    max_workers = os.cpu_count() or 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for parent_dir, files in file_groups.items():
            files_dict = {name: path for name, path in files}
            futures[executor.submit(validate_files, files_dict)] = parent_dir
            
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                valid_dirs.append(futures[future])

    print(f"Filtering complete. Retained {len(valid_dirs)} fully valid samples.")

    # Save the filtered list
    output_file = "filtered_spin_neutral.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(valid_dirs, f)
        
    print(f"Saved fully validated sample paths to '{output_file}'")

if __name__ == "__main__":
    main()
