# SAP Interpreter Overview

Safety Polytope (SaP) is a geometric safety layer that defines a set of half-space constraints ("facets") in the hidden-state representation space of a language model ([Learning Safety Constraints for Large Language Models](https://arxiv.org/abs/2505.24445) the technical details).  

`sap-interpreter` provides a lightweight, **post-hoc analysis toolkit** for understanding what these facets / edges capture and whether specialization emerges inside the polytope.

The library focuses on three complementary questions:

1. **Which inputs violate which facets (at whole input and token level)?**  (`compute-edge-violations`)
2. **Which concept encoder features are activated by a given input (whole input and token level)?**  (`extract-sae-activations`)

Taken together, these scripts let you trace a single safety facet all the way from *hidden state* → *activation* → *violation* → *natural-language example*.

## End-to-End Workflow

Below is the pipeline we use in our mechanistic-interpretability studies.

1. **Extract hidden states from your dataset `crlhf`.**  

2. **Train SaP `crlhf`.**  

3. **Compute sample- / token-level facet violations.**  
   ```bash
   compute-edge-violations \
    --model_path /path/to/model \
    --trained_weights_path /path/to/weights \
    --hidden_states_path /path/to/hidden_states \
    --output_dir outputs \
    --token_level \
    --save_separate_datasets
   ```
   • Outputs compressed NPZ files with raw violation scores and a CSV with per-facet statistics.  
   • When `--token_level` is active we additionally store `token_level/*.npz` containing the token sequence, per-token violations and the original text.

4. **Extract SaP Concept Encoder activations.**  
   ```bash
   extract-sae-activations \
    --model_path /path/to/model \
    --trained_weights_path /path/to/weights \
    --hidden_states_path /path/to/hidden_states \
    --output_dir outputs \
    --token_level \
    --save_separate_datasets
   ```
   This mirrors step 3 but saves *activations* on features (before applying facet normals/thresholds) instead of violations on facets.

5. **Aggregate results across multiple runs (optional).**  
   ```bash
   python -m sap_interpreter.combine_violations \
       --input_dirs outputs/run-* \
       --output_dir outputs/combined \
       --token_level --edge_level
   ```

6. **Inspect & visualise.**  
   • Use the companion Streamlit app in `sap-interpret-frontend` to explore `edge_violations.npz`, `sae_activations.npz` and token-level files.


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

### Safety Polytope dependency

All scripts expect the upstream **Safety Polytope** package (which still registers itself under the `crlhf` namespace for backward-compatibility) to be importable. Install it once per environment:

```bash
git clone https://github.com/lasgroup/SafetyPolytope.git
cd SafetyPolytope
pip install -e .
```