import argparse
import logging
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
log = logging.getLogger(__name__)

# Helper script to combine multiple violation result files from multiple directories
# (for example to combine the results of test and train datasets)

def combine_token_level_violations(input_dirs, output_dir):
    """Combine multiple token-level violation files from multiple directories into a single file."""
    log.info("Combining token-level violations...")
    
    # Initialize combined data
    all_tokens = []
    all_violations = []
    all_texts = []
    
    # Process each input directory
    for input_dir in input_dirs:
        token_dir = os.path.join(input_dir, "token_level")
        if not os.path.exists(token_dir):
            log.warning(f"Token-level directory not found: {token_dir}")
            continue
            
        # Find all token-level violation files in this directory
        token_files = glob(os.path.join(token_dir, "*_token_level_violations.npz"))
        if not token_files:
            log.warning(f"No token-level violation files found in {token_dir}")
            continue
            
        # Process each file in this directory
        for file_path in tqdm(token_files, desc=f"Processing token-level files in {input_dir}"):
            try:
                data = np.load(file_path, allow_pickle=True)
                
                # Extract data
                tokens = data['tokens']
                violations = data['violations']
                texts = data['original_texts']
                
                # Add to combined data
                all_tokens.extend(tokens)
                all_violations.extend(violations)
                all_texts.extend(texts)
                
                log.info(f"Processed {file_path} with {len(tokens)} samples")
                
            except Exception as e:
                log.error(f"Error processing {file_path}: {str(e)}")
    
    # Save combined data
    if all_tokens:
        output_path = os.path.join(output_dir, "combined_token_level_violations.npz")
        np.savez_compressed(
            output_path,
            tokens=np.array(all_tokens, dtype=object),
            violations=np.array(all_violations, dtype=object),
            original_texts=all_texts
        )
        log.info(f"Saved combined token-level violations to {output_path}")
        log.info(f"Total samples: {len(all_tokens)}")
    else:
        log.warning("No token-level data to combine")

def combine_edge_violations(input_dirs, output_dir):
    """Combine multiple edge violation files from multiple directories into a single file."""
    log.info("Combining edge violations...")
    
    # Initialize combined data
    all_violations = []
    all_texts = []
    
    # Process each input directory
    for input_dir in input_dirs:
        facets_dir = os.path.join(input_dir, "facets")
        if not os.path.exists(facets_dir):
            log.warning(f"Facets directory not found: {facets_dir}")
            continue
            
        # Find all edge violation files in this directory
        edge_files = glob(os.path.join(facets_dir, "*_edge_violations.npz"))
        if not edge_files:
            log.warning(f"No edge violation files found in {facets_dir}")
            continue
            
        # Process each file in this directory
        for file_path in tqdm(edge_files, desc=f"Processing edge violation files in {input_dir}"):
            try:
                data = np.load(file_path, allow_pickle=True)
                
                # Extract data
                violations = data['violations']
                texts = data['original_texts']
                
                # Add to combined data
                all_violations.append(violations)
                all_texts.extend(texts)
                
                log.info(f"Processed {file_path} with {len(texts)} samples")
                
            except Exception as e:
                log.error(f"Error processing {file_path}: {str(e)}")
    
    # Save combined data
    if all_violations:
        # Concatenate all violations
        combined_violations = np.concatenate(all_violations, axis=0)
        
        output_path = os.path.join(output_dir, "combined_edge_violations.npz")
        np.savez_compressed(
            output_path,
            violations=combined_violations,
            original_texts=all_texts
        )
        log.info(f"Saved combined edge violations to {output_path}")
        log.info(f"Total samples: {len(all_texts)}")
        
        # Also save a CSV with summary statistics
        csv_path = os.path.join(output_dir, "edge_violation_summary.csv")
        mean_violations = combined_violations.mean(axis=1)
        max_violations = combined_violations.max(axis=1)
        num_pos_violations = (combined_violations > 0).sum(axis=1)
        
        summary_df = pd.DataFrame({
            "text": all_texts,
            "mean_violation": mean_violations,
            "max_violation": max_violations,
            "positive_violations_count": num_pos_violations
        })
        summary_df.to_csv(csv_path, index=False)
        log.info(f"Saved summary statistics to {csv_path}")
    else:
        log.warning("No edge violation data to combine")

def main():
    parser = argparse.ArgumentParser(description='Combine multiple violation result files from multiple directories')
    parser.add_argument('--input_dirs', type=str, nargs='+', required=True,
                      help='List of directories containing the violation files to combine')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the combined results')
    parser.add_argument('--token_level', action='store_true',
                      help='Combine token-level violations')
    parser.add_argument('--edge_level', action='store_true',
                      help='Combine edge-level violations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process token-level violations if requested
    if args.token_level:
        combine_token_level_violations(args.input_dirs, args.output_dir)
    
    # Process edge-level violations if requested
    if args.edge_level:
        combine_edge_violations(args.input_dirs, args.output_dir)
    
    log.info("Done combining violation results")

if __name__ == "__main__":
    main() 