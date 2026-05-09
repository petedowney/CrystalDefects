import os
from pathlib import Path
from collections import defaultdict

def pair_dos_files(base_path):
    """
    Traverse directory structure and pair dos_DW, dos_UP, and POSCAR files.
    Returns a list of tuples (dos_DW, dos_UP, POSCAR) grouped by their parent directory.
    """
    file_groups = defaultdict(dict)
    
    # Walk through all directories and files
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file in ['dos_DW', 'dos_UP', 'POSCAR']:
                # Use parent directory as the key
                file_groups[root][file] = os.path.join(root, file)
    
    # Create tuples of complete sets (where all three files exist)
    result = []
    for directory, files in sorted(file_groups.items()):
        if len(files) == 3:  # Only include if all three files exist
            dos_dw = files.get('dos_DW')
            dos_up = files.get('dos_UP')
            poscar = files.get('POSCAR')
            if all([dos_dw, dos_up, poscar]):
                result.append((dos_dw, dos_up, poscar))
        elif len(files) > 0:
            # Print warning for incomplete sets
            print(f"Warning: Incomplete set in {directory}: {list(files.keys())}")
    
    return result

if __name__ == "__main__":
    base_path = "/home/petedowney/github/CrystalDefects/all_dos_data_full"
    
    tuples = pair_dos_files(base_path)
    
    print(f"\nFound {len(tuples)} complete file sets:\n")
    for i, (dos_dw, dos_up, poscar) in enumerate(tuples, 1):
        print(f"Set {i}:")
        print(f"  dos_DW: {dos_dw}")
        print(f"  dos_UP: {dos_up}")
        print(f"  POSCAR: {poscar}")
        print()
    
    # Optionally, save to a file
    with open("/home/petedowney/github/CrystalDefects/file_pairs.txt", "w") as f:
        for dos_dw, dos_up, poscar in tuples:
            f.write(f"({dos_dw}, {dos_up}, {poscar})\n")
    
    print(f"Results saved to file_pairs.txt")
