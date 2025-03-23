import argparse
import logging
import os
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import load_data_and_model
from model_utils import compute_token_level_violations

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate edge violations from polytope safety model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the base Mistral model')
    parser.add_argument('--trained_weights_path', type=str, required=True,
                        help='Path to the trained polytope weights')
    parser.add_argument('--hidden_states_path', type=str, required=True,
                        help='Path to the hidden states data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_separate_datasets', action='store_true',
                        help='Save separate .npz files for each dataset')
    parser.add_argument('--token_level', action='store_true',
                        help='Compute token-level violations instead of sample-level')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Additional subdirectory paths
    facets_dir = os.path.join(args.output_dir, "facets")
    os.makedirs(facets_dir, exist_ok=True)
    
    if args.token_level:
        token_dir = os.path.join(args.output_dir, "token_level")
        os.makedirs(token_dir, exist_ok=True)

    log.info("Starting edge violations collection...")
    log.info(f"Arguments: {args}")

    # 1. Load data & model
    safety_model, datasets = load_data_and_model(args)

    # 2. Load the model/tokenizer if we need token-level analysis
    model = None
    tokenizer = None
    if args.token_level:
        log.info(f"Loading model from {args.model_path} for token-level analysis...")
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        safety_model.model = model
        safety_model.tokenizer = tokenizer
        model.to(safety_model.device)

    # Ensure eval mode
    safety_model.eval()

    # For combined data
    all_combined_violations = []
    all_original_texts = []

    # For combined token-level data
    all_token_lists = []
    all_token_violations = []
    all_token_texts = []

    for dataset_name, dataset in datasets.items():
        log.info(f"Processing {dataset_name} dataset...")
        if args.token_level:
            # Compute token-level
            dataset_token_violations = []
            for idx, text in enumerate(tqdm(dataset["input_texts"], desc=f"{dataset_name} token-level")):
                # Print the input text for token-level violations computation
                log.info(f"Computing token-level violations for text [{idx}]: {text}")
                tokens, token_violations = compute_token_level_violations(
                    safety_model, tokenizer, text, safety_model.device
                )
                dataset_token_violations.append((tokens, token_violations))

            # Save token-level results if desired
            if args.save_separate_datasets:
                token_level_path = os.path.join(token_dir, f"{dataset_name}_token_level_violations.npz")
                np.savez_compressed(
                    token_level_path,
                    tokens=np.array([tup[0] for tup in dataset_token_violations], dtype=object),
                    violations=np.array([tup[1].numpy() for tup in dataset_token_violations], dtype=object),
                    original_texts=dataset["input_texts"]
                )
                log.info(f"Saved {dataset_name} token-level NPZ to {token_level_path}")

            # Update combined
            all_token_lists.extend([tup[0] for tup in dataset_token_violations])
            all_token_violations.extend([tup[1].numpy() for tup in dataset_token_violations])
            all_token_texts.extend(dataset["input_texts"])

        # Now do sample-level
        all_violations = []
        dataset_texts = []
        batch_sizes = []

        num_edges = safety_model.phi.shape[0]
        log.info(f"Number of edges in polytope: {num_edges}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataset["dataloader"], desc=f"{dataset_name} sample-level")):
                inputs, _ = batch
                if batch_idx == 0:
                    start_idx = 0
                else:
                    start_idx = sum(batch_sizes[:batch_idx])
                end_idx = start_idx + inputs.size(0)
                batch_sizes.append(inputs.size(0))

                batch_texts = dataset["input_texts"][start_idx:end_idx]
                dataset_texts.extend(batch_texts)

                inputs = inputs.to(safety_model.device)
                hs_rep = safety_model.get_hidden_states_representation(inputs)
                features = safety_model.feature_extractor(hs_rep)
                
                violations = torch.matmul(safety_model.phi, features.T) - safety_model.threshold.unsqueeze(1)
                violations = violations.T.cpu()
                all_violations.append(violations)

        if len(all_violations) > 0:
            violations_t = torch.cat(all_violations, dim=0)
            all_combined_violations.append(violations_t)
            all_original_texts.extend(dataset_texts)

            # Example stats
            min_v, max_v = violations_t.min().item(), violations_t.max().item()
            mean_v = violations_t.mean().item()
            pos_ratio = (violations_t > 0).float().mean().item()

            log.info(f"{dataset_name} stats - min: {min_v:.4f}, max: {max_v:.4f}, mean: {mean_v:.4f}")
            log.info(f"{dataset_name} positive ratio: {pos_ratio:.4f}")

            # Edge violation counts
            edge_violation_counts = (violations_t > 0).sum(dim=0).numpy()
            total_samples = violations_t.shape[0]
            edge_indices = np.arange(len(edge_violation_counts))
            violation_ratio = edge_violation_counts / total_samples

            edge_df = pd.DataFrame({
                "Edge": edge_indices,
                "Violation_Count": edge_violation_counts,
                "Violation_Ratio": violation_ratio
            })
            edge_df.sort_values("Violation_Count", ascending=False, inplace=True)

            # Save CSV
            edge_counts_path = os.path.join(facets_dir, f"{dataset_name}_edge_violation_counts.csv")
            edge_df.to_csv(edge_counts_path, index=False)
            log.info(f"{dataset_name} counts saved to {edge_counts_path}")

            # Quick bar chart with Plotly
            fig = px.bar(
                edge_df, x="Edge", y="Violation_Count",
                title=f"{dataset_name} - Violating Samples per Edge",
                labels={"Edge": "Edge Index", "Violation_Count": "Count"}
            )
            barchart_path = os.path.join(facets_dir, f"{dataset_name}_edge_violation_counts.png")
            fig.write_image(barchart_path)

            # Save dataset-level NPZ if requested
            if args.save_separate_datasets:
                out_path = os.path.join(facets_dir, f"{dataset_name}_edge_violations.npz")
                np.savez_compressed(out_path, violations=violations_t.numpy(), original_texts=dataset_texts)
                log.info(f"{dataset_name} edge violations saved to {out_path}")

    # Combine all dataset violations
    if len(all_combined_violations) > 0:
        combined_violations_t = torch.cat(all_combined_violations, dim=0)
        combined_out = combined_violations_t.numpy()
        combined_path = os.path.join(facets_dir, "edge_violations.npz")
        np.savez_compressed(combined_path, violations=combined_out, original_texts=all_original_texts)
        log.info(f"Combined edge violations saved to {combined_path}")

        # Write a CSV with summary data
        csv_path = os.path.join(facets_dir, "sample_violations.csv")
        mean_violations = combined_out.mean(axis=1)
        max_violations = combined_out.max(axis=1)
        num_pos_violations = (combined_out > 0).sum(axis=1)

        data = {
            "text": all_original_texts,
            "mean_violation": mean_violations,
            "max_violation": max_violations,
            "positive_violations_count": num_pos_violations
        }
        # Optionally add columns for the first ~50 edges
        max_cols = min(50, combined_out.shape[1])
        for i in range(max_cols):
            data[f"edge_{i}"] = combined_out[:, i]

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        log.info(f"Detailed CSV saved at {csv_path}")
    else:
        log.warning("No violations found, nothing to combine.")

    # Finally, save combined token-level if needed
    if args.token_level and len(all_token_violations) > 0:
        combined_token_path = os.path.join(token_dir, "token_level_violations.npz")
        np.savez_compressed(
            combined_token_path,
            tokens=np.array(all_token_lists, dtype=object),
            violations=np.array(all_token_violations, dtype=object),
            original_texts=all_token_texts
        )
        log.info(f"Combined token-level data saved to {combined_token_path}")

    log.info("Done collecting edge violations.")

if __name__ == "__main__":
    main()
