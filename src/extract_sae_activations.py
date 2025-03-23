import argparse
import logging
import os
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_utils import load_data_and_model
from model_utils import compute_token_level_activations

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Extract SAP/SAE activations from hidden states')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the Mistral model')
    parser.add_argument('--trained_weights_path', type=str, required=True,
                        help='Path to the trained weights')
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
                        help='Compute token-level activations instead of sample-level')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Additional subdirectory for token-level data
    if args.token_level:
        token_dir = os.path.join(args.output_dir, "token_level")
        os.makedirs(token_dir, exist_ok=True)
    
    log.info("Starting SAP/SAE activations extraction...")
    log.info(f"Arguments: {args}")

    # Load data & safety model
    safety_model, datasets = load_data_and_model(args)

    # Load tokenizer and model if needed
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model if token-level analysis is required
    model = None
    if args.token_level:
        log.info(f"Loading model from {args.model_path} for token-level analysis...")
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
        model.to(safety_model.device)
        safety_model.model = model
        safety_model.tokenizer = tokenizer

    safety_model.eval()
    
    # For combined data
    all_combined_activations = []
    all_original_texts = []
    
    # For combined token-level data
    all_token_lists = []
    all_token_activations = []
    all_token_texts = []

    # Iterate over train/test sets
    for dataset_name, dataset in datasets.items():
        log.info(f"Processing {dataset_name} dataset...")
        
        # Token-level processing if requested
        if args.token_level:
            dataset_token_activations = []
            for idx, text in enumerate(tqdm(dataset["input_texts"], desc=f"{dataset_name} token-level")):
                log.info(f"Computing token-level activations for text [{idx}]: {text}")
                tokens, token_activations = compute_token_level_activations(
                    safety_model, tokenizer, text, safety_model.device
                )
                dataset_token_activations.append((tokens, token_activations))
                
            # Save token-level results if desired
            if args.save_separate_datasets:
                token_level_path = os.path.join(token_dir, f"{dataset_name}_token_level_activations.npz")
                np.savez_compressed(
                    token_level_path,
                    tokens=np.array([tup[0] for tup in dataset_token_activations], dtype=object),
                    activations=np.array([tup[1].detach().numpy() for tup in dataset_token_activations], dtype=object),
                    original_texts=dataset["input_texts"]
                )
                log.info(f"Saved {dataset_name} token-level NPZ to {token_level_path}")
                
            # Update combined
            all_token_lists.extend([tup[0] for tup in dataset_token_activations])
            all_token_activations.extend([tup[1].detach().numpy() for tup in dataset_token_activations])
            all_token_texts.extend(dataset["input_texts"])
            
        # Existing sample-level processing
        all_activations = []
        original_texts = []
        batch_sizes = []

        for batch_idx, batch in enumerate(tqdm(dataset["dataloader"], desc=f"{dataset_name} feature extraction")):
            inputs, _ = batch
            if batch_idx == 0:
                start_idx = 0
            else:
                start_idx = sum(batch_sizes[:batch_idx])
            end_idx = start_idx + inputs.size(0)
            batch_sizes.append(inputs.size(0))

            batch_texts = dataset["input_texts"][start_idx:end_idx]
            original_texts.extend(batch_texts)

            inputs = inputs.to(safety_model.device)
            # This should return the same shape as the hidden states used in training
            hs_rep = safety_model.get_hidden_states_representation(inputs)
            features = safety_model.feature_extractor(hs_rep)

            features_cpu = features.cpu().float()  # [batch_size, feature_dim]
            all_activations.append(features_cpu)

        if len(all_activations) > 0:
            activations_t = torch.cat(all_activations, dim=0)
            log.info(f"{dataset_name} final shape: {activations_t.shape}")
            
            # L0 measure: how many features > 0
            is_active = (activations_t > 0).float()
            l0_per_sample = is_active.sum(dim=-1)
            if l0_per_sample.numel() > 0:
                avg_l0 = l0_per_sample.mean().item()
                log.info(f"{dataset_name} average L0: {avg_l0:.2f}")
                # Optionally plot a histogram
                l0_vals = l0_per_sample.numpy()
                df = pd.DataFrame({"L0 (active features)": l0_vals})
                fig = px.histogram(
                    df, x="L0 (active features)", nbins=30,
                    title=f"{dataset_name.capitalize()} L0 Distribution"
                )
                hist_path = os.path.join(args.output_dir, f"{dataset_name}_l0_hist.png")
                fig.write_image(hist_path)
            else:
                log.warning(f"No samples to compute L0 for {dataset_name}.")
                
            # Save separate if needed
            if args.save_separate_datasets:
                out_path = os.path.join(args.output_dir, f"{dataset_name}_sae_activations.npz")
                np.savez_compressed(
                    out_path,
                    activations=activations_t.detach().numpy(),
                    original_texts=original_texts
                )
                log.info(f"Saved {dataset_name} activations at {out_path}")

            all_combined_activations.append(activations_t)
            all_original_texts.extend(original_texts)
        else:
            log.warning(f"No activations found for {dataset_name}; skipping.")

    # Combine everything
    if len(all_combined_activations) > 0:
        combined_activations_t = torch.cat(all_combined_activations, dim=0)
        combined_out_path = os.path.join(args.output_dir, "sae_activations.npz")
        np.savez_compressed(
            combined_out_path,
            activations=combined_activations_t.detach().numpy(),
            original_texts=all_original_texts
        )
        log.info(f"Saved combined activations to {combined_out_path}")
    else:
        log.warning("No activations from any dataset. Nothing to combine.")

    # Finally, save combined token-level data if needed
    if args.token_level and len(all_token_activations) > 0:
        combined_token_path = os.path.join(token_dir, "token_level_activations.npz")
        np.savez_compressed(
            combined_token_path,
            tokens=np.array(all_token_lists, dtype=object),
            activations=np.array(all_token_activations, dtype=object),
            original_texts=all_token_texts
        )
        log.info(f"Combined token-level data saved to {combined_token_path}")

    log.info("Done extracting SAP/SAE activations.")

if __name__ == "__main__":
    main()