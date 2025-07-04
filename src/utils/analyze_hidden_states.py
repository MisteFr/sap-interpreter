import torch
import numpy as np
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

log = logging.getLogger(__name__)

# Helper script to analyze the hidden states file

def load_hidden_states(file_path):
    """Load hidden states from a file."""
    log.info(f"Loading hidden states from {file_path}")
    return torch.load(file_path)

def analyze_hidden_states(hidden_states_data):
    """Analyze hidden states data and return statistics."""
    results = {
        "structure": {},
        "statistics": {}
    }
    
    # Analyze the structure
    def analyze_structure(data, path=""):
        if isinstance(data, dict):
            structure = {}
            for key, value in data.items():
                structure[key] = analyze_structure(value, f"{path}.{key}" if path else key)
            return structure
        elif isinstance(data, (list, tuple)):
            if len(data) > 0:
                return analyze_structure(data[0], path)
            return "empty_list"
        elif torch.is_tensor(data):
            return f"tensor(shape={data.shape}, dtype={data.dtype})"
        elif isinstance(data, np.ndarray):
            return f"ndarray(shape={data.shape}, dtype={data.dtype})"
        else:
            return type(data).__name__
    
    results["structure"] = analyze_structure(hidden_states_data)
    
    # Calculate statistics for tensors and arrays
    def analyze_tensor_stats(data, path=""):
        if torch.is_tensor(data):
            # Convert BFloat16 to float32 before numpy conversion
            if data.dtype == torch.bfloat16:
                data = data.to(torch.float32)
            data_np = data.detach().cpu().numpy()
            return {
                "shape": data.shape,
                "mean": float(np.mean(data_np)),
                "std": float(np.std(data_np)),
                "min": float(np.min(data_np)),
                "max": float(np.max(data_np)),
                "dtype": str(data.dtype)
            }
        elif isinstance(data, np.ndarray):
            return {
                "shape": data.shape,
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "dtype": str(data.dtype)
            }
        elif isinstance(data, dict):
            stats = {}
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                stats[key] = analyze_tensor_stats(value, new_path)
            return stats
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            return analyze_tensor_stats(data[0], path)
        return None
    
    results["statistics"] = analyze_tensor_stats(hidden_states_data)
    
    return results

def print_results(results):
    """Print the analysis results in a readable format."""
    print("\nHidden States Analysis Results:")
    print("=" * 50)
    
    print("\nData Structure:")
    print("-" * 30)
    def print_structure(structure, indent=0):
        for key, value in structure.items():
            print("  " * indent + f"{key}:")
            if isinstance(value, dict):
                print_structure(value, indent + 1)
            else:
                print("  " * (indent + 1) + str(value))
    
    print_structure(results["structure"])
    
    print("\nStatistics:")
    print("-" * 30)
    def print_stats(stats, indent=0):
        for key, value in stats.items():
            if value is None:
                continue
            print("  " * indent + f"{key}:")
            if isinstance(value, dict):
                if "shape" in value:  # It's a tensor/array stats
                    print("  " * (indent + 1) + f"Shape: {value['shape']}")
                    print("  " * (indent + 1) + f"Mean: {value['mean']:.6f}")
                    print("  " * (indent + 1) + f"Std: {value['std']:.6f}")
                    print("  " * (indent + 1) + f"Min: {value['min']:.6f}")
                    print("  " * (indent + 1) + f"Max: {value['max']:.6f}")
                    print("  " * (indent + 1) + f"Dtype: {value['dtype']}")
                else:
                    print_stats(value, indent + 1)
    
    print_stats(results["statistics"])

def print_first_elements(hidden_states_data):
    """Print the first elements of train and test data if they exist."""
    print("\nFirst Elements of Train and Test:")
    print("=" * 50)
    
    def print_dataset_elements(dataset_name, data):
        print(f"\n{dataset_name} first element:")
        print("-" * 30)
        if isinstance(data, dict):
            if 'hidden_states' in data:
                print("Hidden States (first element):")
                print(f"Shape: {data['hidden_states'][0].shape}")
                print(f"First few values: {data['hidden_states'][0][:5]}")
            
            if 'labels' in data:
                print("\nLabel (first element):")
                print(f"Value: {data['labels'][0]}")
            
            if 'input_texts' in data:
                print("\nInput Text (first element):")
                print(f"Text: {data['input_texts'][0]}")
        else:
            print(f"No {dataset_name} data found or invalid format")
    
    if isinstance(hidden_states_data, dict):
        if 'train' in hidden_states_data:
            print_dataset_elements("Train", hidden_states_data['train'])
        
        if 'test' in hidden_states_data:
            print_dataset_elements("Test", hidden_states_data['test'])

def main():
    parser = argparse.ArgumentParser(description="Analyze hidden states file")
    parser.add_argument("--file", type=str, required=True, help="Path to the hidden states file")
    args = parser.parse_args()
    
    # Load the file
    hidden_states_data = load_hidden_states(args.file)
    
    # Print first elements of train and test
    print_first_elements(hidden_states_data)
    
    # Analyze the data
    results = analyze_hidden_states(hidden_states_data)
    
    # Print the results
    print_results(results)

if __name__ == "__main__":
    main() 