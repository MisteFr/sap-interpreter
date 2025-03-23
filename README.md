# SAP Interpreter

SAP Interpreter is a toolkit for analyzing and interpreting SAP model, developed as part of a semester project at ETH Zurich.

## Installation

### Local Development Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd sap-interpreter
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Regular Installation

```bash
pip install .
```

## Key Features

- Extracting and analyzing SAP's Concept Encoder activations
- Extracting and analyzing SAP's facets violations


## Usage

### Computing Edge Violations

```bash
compute-edge-violations \
    --model_path /path/to/model \
    --trained_weights_path /path/to/weights \
    --hidden_states_path /path/to/hidden_states \
    --output_dir outputs \
    --token_level \
    --save_separate_datasets
```

### Extracting SAE Activations

```bash
extract-sae-activations \
    --model_path /path/to/model \
    --trained_weights_path /path/to/weights \
    --hidden_states_path /path/to/hidden_states \
    --output_dir outputs \
    --token_level \
    --save_separate_datasets
```