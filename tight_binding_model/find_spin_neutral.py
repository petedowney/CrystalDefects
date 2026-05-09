import os
import pickle

def find_spin_neutral_samples(base_path):
    """
    Looks through all DOS data directories, reads the first 5 entries 
    of dos_DW and dos_UP. If they match completely, it is considered spin neutral.
    Returns a list of the matching directory paths.
    """
    spin_neutral = []
    
    for root, dirs, files in os.walk(base_path):
        if 'dos_DW' in files and 'dos_UP' in files:
            dos_dw_path = os.path.join(root, 'dos_DW')
            dos_up_path = os.path.join(root, 'dos_UP')
            
            try:
                with open(dos_dw_path, 'r') as f_dw, open(dos_up_path, 'r') as f_up:
                    # Extract the first 5 non-empty lines
                    dw_lines = [line.strip() for line in f_dw if line.strip()][:5]
                    up_lines = [line.strip() for line in f_up if line.strip()][:5]
                    
                    # If both have at least 1 entry and match perfectly on the first 5 lines
                    if len(dw_lines) > 0 and len(dw_lines) == len(up_lines) and dw_lines == up_lines:
                        spin_neutral.append(root)
            except Exception as e:
                print(f"Error reading files in {root}: {e}")
                
    return spin_neutral

if __name__ == "__main__":
    # Point relative to tight_binding_model folder
    base_path = "./all_dos_data_full/all_dos_data_full"
    
    print(f"Scanning {base_path} for spin neutral samples...")
    spin_neutral_dirs = find_spin_neutral_samples(base_path)
    
    print(f"Found {len(spin_neutral_dirs)} spin neutral samples.")
    
    # Save the resulting list
    output_file = "spin_neutral.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(spin_neutral_dirs, f)
        
    print(f"Saved spin neutral list to {output_file}")
