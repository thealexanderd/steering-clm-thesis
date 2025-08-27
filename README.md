# Drug Generation with Steering Vectors

This project implements molecular generation experiments using steering vectors to control the properties of generated drug molecules. The system uses a pre-trained  [DrugGen](https://github.com/mahsasheikh/DrugGen) model and applies steering vectors to guide the generation of SMILES strings with specific molecular properties like size, solubility, and lipophilicity.

## Project Structure

```
├── README.md
├── requirements.txt
├── experiments.py           # Main experiment runner
├── model_and_generation.py  # Core generation and steering logic
├── drugGen_generator.py     # Adapted from DrugGen repository
├── check_smiles.py         # Adapted from DrugGen repository  
├── helpers.py              # Utility functions for molecular analysis
└── steering-vectors/       # Git submodule (also available as pip package)
```

## Overview

The project consists of several key components:

- **Steering Vector Generation**: Creates steering vectors from molecular pairs to control generation
- **Molecular Generation**: Uses DrugGen model to generate SMILES strings for target proteins
- **Property Evaluation**: Analyzes generated molecules for size, solubility, and lipophilicity
- **Experimental Framework**: Compares generation with and without steering vectors

## Dependencies

### Core Requirements

The project requires several Python packages. Install them using:

```bash
pip install -r requirements.txt
```

### Key Dependencies:

- **PyTorch**: Deep learning framework for model execution
- **Transformers**: Hugging Face library for the DrugGen model
- **RDKit**: Cheminformatics toolkit for molecular analysis
- **Steering Vectors**: Custom library for steering vector implementation
- **SolTranNet**: Solubility prediction model
- **Scientific Libraries**: NumPy, Pandas, SciPy for data analysis

### Special Installation Notes

1. **Steering Vectors**: Modified version of the steering-vectors pip package, you need to install it manually:
   ```bash
   cd steering-vectors
   pip install .
   ```

2. **RDKit**: May require conda installation for some systems:
   ```bash
   conda install -c conda-forge rdkit
   ```

## File Descriptions

### Core Files

- **`experiments.py`**: Main experiment runner with configurable parameters
- **`model_and_generation.py`**: Core steering vector generation and molecule generation logic
- **`helpers.py`**: Utility functions for molecular property analysis and pair generation

### Adapted Files (from DrugGen Repository)

- **`drugGen_generator.py`**: Model loading and basic SMILES generation (adapted from DrugGen)
- **`check_smiles.py`**: SMILES validation and quality checking (adapted from DrugGen)

These files have been modified to work with the steering vector framework while maintaining compatibility with the original DrugGen model.

## Running Experiments

### Basic Usage

The main experimental framework is controlled through the `main()` function in `experiments.py`. To run experiments:

```bash
python experiments.py
```

### Customizing Experiments

Modify the `main()` function in `experiments.py` to customize your experiments:

```python
def main():
    # Load protein sequences dataset
    df = pd.read_json("hf://datasets/alimotahharynia/approved_drug_target/uniprotId_sequence_2024_11_01.json")
    uniprot_to_sequence = dict(zip(df["UniProt_id"], df["Sequence"]))

    # Define target proteins
    new_list_of_uniprot_ids = ['P07900', 'P00734', 'Q14524', ...]  # Add your protein IDs
    
    # Run experiments
    results = perform_experiments(
        protein_sequences,
        new_list_of_uniprot_ids,
        filename_prefix="data/figures/",
        duplicates=False,              # Whether to allow duplicate pairs
        num_to_generate_pairs=50,      # Number of training pairs for steering
        num_to_generate=250,           # Number of molecules to generate per condition
        experiment="size",             # Type of experiment: "size", "solubility", "logp"
        field=False,                   # Use field-based steering (True/False)
        tau=1.0,                      # Temperature for field-based steering
        multiplier=1.0,               # Steering strength
        token_index=-1,               # Token position for steering
        layer_index=11,               # Model layer for steering vector extraction
    )
```

### Experiment Types

1. **Size Control** (`experiment="size"`):
   - Controls the number of atoms in generated molecules
   - Useful for generating molecules of specific sizes

2. **Solubility Control** (`experiment="logs"`):
   - Controls predicted solubility of generated molecules
   - Uses SolTranNet for solubility predictions

3. **Lipophilicity Control** (`experiment="logp"`):
   - Controls lipophilicity (logP) values
   - Uses RDKit's Crippen module for logP calculations

### Key Parameters

- **`num_to_generate_pairs`**: Number of molecules generated to create the pairs for training the steering vector (field)
- **`num_to_generate`**: Number of molecules generated for each condition (no steering, positive steering, negative steering)
- **`field`**: Boolean flag to use field-based steering vs. standard steering vectors
- **`multiplier`**: Controls the strength of the steering effect
- **`duplicates`**: Whether to allow a molecule to appear in multiple pairs (either as a negative or positive example)

### Output

Experiments generate several outputs in the `data/figures/` directory:

- **CSV files**: Statistical results for each protein
- **PKL files**: Generated SMILES strings for each condition
- **Pair files**: Training pairs used for steering vector generation

## Example Workflow

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   cd steering-vectors && pip install . && cd ..
   ```

2. **Configure Experiment**:
   Edit the `main()` function in `experiments.py` with your desired parameters

3. **Run Experiment**:
   ```bash
   python experiments.py
   ```

4. **Analyze Results**:
   Check the generated CSV files and PKL files in `data/figures/`

## Model Information

The project uses the **DrugGen** model (`alimotahharynia/DrugGen`), a GPT-2 based model fine-tuned for drug-target interaction prediction and molecular generation.

## Citation

If you use this code in your research, please cite the relevant papers for:
- DrugGen model
- Steering Vectors methodology
- Any specific molecular property prediction models used

## License

Please check the licenses of the individual components, especially:
- DrugGen model and associated code
- Steering vectors library
- RDKit and other dependencies

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Ensure PyTorch is installed with CUDA support if using GPU
2. **RDKit Installation**: Use conda for RDKit if pip installation fails
3. **Memory Issues**: Reduce `num_to_generate` for large experiments
4. **Model Loading**: Ensure internet connection for downloading DrugGen model on first run


