# CrystalDefects

A machine learning system for predicting crystal defect properties using Graph Convolutional Networks (GCNs) for QIS applications.

## Overview

This project uses GCNs to predict various properties of crystal defects that are important for quantum information science. The system analyzes defect structures represented as graphs, where atoms are nodes and chemical bonds are edges, to predict properties like spin state, zero-phonon line energy, and transition dipole moments. 

The goal is to be able to predict these properties without the many compute hours that a traditional HSE simulation takes

## Key Features

- **Graph-based defect representation**: Crystal defects are modeled as graphs with atoms as nodes and bonds as edges
- **Multiple property prediction**: Predicts various defect properties including:
  - Spin state (classification)
  - Zero-phonon line energy (regression)
  - HSE excited state delta Q (regression)
  - Fermi energy (regression)
  - Defect level transition dipole moment (regression)
  - Total energy (regression)

## Data Pipeline

### Raw Data

- Located in `db_files/db_files/`
- Contains JSON files with defect calculation results from DFT simulations
- Each JSON contains HSE and PBE data for different charge states (-2, -1, 0, 1, 2)
- **Note**: This data is not included in the repository and must be manually placed in `db_files/db_files/` before running the preprocessing steps

### Preprocessing
1. **Parser** (`parser.ipynb`): Extracts valid HSE/PBE POSCAR files from JSON data
2. **Feature Engineering**: Converts atomic structures to graph representations:
   - Nodes: Atoms with features from `properties.txt` (atomic properties) + charge state
   - Edges: Chemical bonds with distance-based weights
   - Adjacency matrices with distance thresholding
3. **Data Processing**: Outlier removal, normalization, and train/validation splitting
4. Stored in `filtered_data/` as NumPy arrays with format `{method}_{property}_{data_type}.npy`

## Model Architecture

### Graph Convolutional Network (GCN)
- **Input**: Graph with node features (atomic properties + charge state)
- **Layers**: GATConv layers with ReLU activation
- **Pooling**: Global graph pooling via flattening
- **Output**: Linear layers for regression/classification

### Variants
- **Classifier**: Uses CrossEntropy loss for classification tasks
- **Regressor**: Uses MSE loss for regression tasks

## Usage

### Training a Model

1. Open `train.ipynb` in Jupyter
2. Configure parameters (learning rate, epochs, batch size, etc.)
3. Select target property from available options
4. Run the training cells

### Visualization

1. Open `visulizations.ipynb` in Jupyter
2. Run the plotting functions to visualize:
   - Training/validation loss curves
   - Predicted vs actual values
   - Model performance metrics

### Data Parsing

1. Open `parser.ipynb` in Jupyter
2. Run cells to:
   - Extract POSCAR files from JSON data
   - Validate HSE calculations
   - Generate file lists for further processing

## File Structure

```
CrystalDefects/
├── GCN.py                 # Regression GCN model
├── GCN_classifier.py      # Classification GCN model
├── util.py                # Utility functions for atomic 
├── loader.py              # Data loading utilities 
├── properties.txt         # Atomic properties database
├── notes.txt              # Project notes and target 
├── train.ipynb            # Main training notebook
├── visulizations.ipynb    # Visualization and analysis
├── parser.ipynb           # Data parsing and preprocessing
├── temp.ipynb             # Temporary/experimental code
├── filtered_data/         # Preprocessed training data
├── models/                # Saved model checkpoints
├── pictures/              # Generated plots and 
└── db_files/              # Raw simulation data
```

## Output and Results

Models are saved as `model_{method}_{property}{version}.pt` in the `models/` directory.

Training generates:
- Loss curves (training vs validation)
- Model checkpoints
- Normalization parameters (for regression)
```

