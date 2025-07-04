import torch
import numpy as np
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

log = logging.getLogger(__name__)


# Helper script to compare two hidden states files

def load_hidden_states(file_path):
    """Load hidden states from a file."""
    log.info(f"Loading hidden states from {file_path}")
    return torch.load(file_path)

def compare_hidden_states(old_hs, new_hs):
    """Compare two hidden states dictionaries."""
    results = {
        "match": True,
        "details": {}
    }
    
    # Check if both files have train and test splits
    splits = ["train", "test"]
    for split in splits:
        if split not in old_hs or split not in new_hs:
            log.error(f"Missing {split} split in one of the files")
            results["match"] = False
            continue
            
        old_data = old_hs[split]
        new_data = new_hs[split]
        
        split_details = {}
        
        # Compare number of samples
        old_samples = len(old_data["hidden_states"])
        new_samples = len(new_data["hidden_states"])
        split_details["num_samples"] = {
            "old": old_samples,
            "new": new_samples,
            "match": old_samples == new_samples
        }
        
        # Convert PyTorch tensors to NumPy arrays for statistics if needed
        def to_numpy(data):
            if torch.is_tensor(data):
                return data.detach().cpu().numpy()
            return data
            
        old_hs_np = to_numpy(old_data["hidden_states"])
        new_hs_np = to_numpy(new_data["hidden_states"])
        
        # Compare hidden states statistics
        old_hs_mean = np.mean(old_hs_np)
        old_hs_std = np.std(old_hs_np)
        new_hs_mean = np.mean(new_hs_np)
        new_hs_std = np.std(new_hs_np)
        
        split_details["hidden_states_stats"] = {
            "old": {"mean": old_hs_mean, "std": old_hs_std},
            "new": {"mean": new_hs_mean, "std": new_hs_std},
            "mean_diff": abs(old_hs_mean - new_hs_mean),
            "std_diff": abs(old_hs_std - new_hs_std)
        }
        
        # Compare labels
        old_labels = to_numpy(old_data["labels"])
        new_labels = to_numpy(new_data["labels"])
        split_details["labels"] = {
            "old_unique": np.unique(old_labels),
            "new_unique": np.unique(new_labels),
            "match": np.array_equal(np.unique(old_labels), np.unique(new_labels))
        }
        
        # Check if new file has input texts and verify their length matches the number of samples
        split_details["input_texts"] = {
            "has_input_texts": "input_texts" in new_data,
            "length": len(new_data["input_texts"]) if "input_texts" in new_data else 0,
            "length_matches_samples": len(new_data["input_texts"]) == new_samples if "input_texts" in new_data else False
        }
        
        results["details"][split] = split_details
        
        # Update overall match status
        if not all([
            split_details["num_samples"]["match"],
            split_details["labels"]["match"],
            split_details["input_texts"]["length_matches_samples"]
        ]):
            results["match"] = False
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare two hidden states files")
    parser.add_argument("--old_file", type=str, required=True, help="Path to the old hidden states file")
    parser.add_argument("--new_file", type=str, required=True, help="Path to the new hidden states file")
    args = parser.parse_args()
    
    # Load both files
    old_hs = load_hidden_states(args.old_file)
    new_hs = load_hidden_states(args.new_file)
    
    # Print first hidden states and input texts from new file
    print("\nFirst Hidden States and Input Texts from New File:")
    print("=" * 50)
    
    for split in ["train", "test"]:
        print(f"\n{split.upper()} Split:")
        print("-" * 30)
        
        if split in new_hs:
            data = new_hs[split]
            print(f"First Hidden State Shape: {data['hidden_states'][0].shape}")
            print(f"First Hidden State:\n{data['hidden_states'][0]}")
            
            if "input_texts" in data:
                print(f"\nFirst Input Text:\n{data['input_texts'][0]}")
            else:
                print("\nNo input texts available")
        else:
            print(f"No {split} split available")
    
    # Compare the files
    results = compare_hidden_states(old_hs, new_hs)
    
    # Print results
    print("\nComparison Results:")
    print("=" * 50)
    
    for split, details in results["details"].items():
        print(f"\n{split.upper()} Split:")
        print("-" * 30)
        
        # Print number of samples
        num_samples = details["num_samples"]
        print(f"Number of samples:")
        print(f"  Old: {num_samples['old']}")
        print(f"  New: {num_samples['new']}")
        print(f"  Match: {'✓' if num_samples['match'] else '✗'}")
        
        # Print hidden states statistics
        hs_stats = details["hidden_states_stats"]
        print(f"\nHidden States Statistics:")
        print(f"  Old - Mean: {hs_stats['old']['mean']:.6f}, Std: {hs_stats['old']['std']:.6f}")
        print(f"  New - Mean: {hs_stats['new']['mean']:.6f}, Std: {hs_stats['new']['std']:.6f}")
        print(f"  Mean Difference: {hs_stats['mean_diff']:.6f}")
        print(f"  Std Difference: {hs_stats['std_diff']:.6f}")
        
        # Print labels information
        labels = details["labels"]
        print(f"\nLabels:")
        print(f"  Old unique values: {labels['old_unique']}")
        print(f"  New unique values: {labels['new_unique']}")
        print(f"  Match: {'✓' if labels['match'] else '✗'}")
        
        # Print input texts information
        texts = details["input_texts"]
        print(f"\nInput Texts:")
        print(f"  New file has input texts: {'✓' if texts['has_input_texts'] else '✗'}")
        if texts['has_input_texts']:
            print(f"  Number of input texts: {texts['length']}")
            print(f"  Length matches number of samples: {'✓' if texts['length_matches_samples'] else '✗'}")
    
    print("\nOverall Result:")
    print("=" * 50)
    print(f"Files match: {'✓' if results['match'] else '✗'}")

if __name__ == "__main__":
    main() 