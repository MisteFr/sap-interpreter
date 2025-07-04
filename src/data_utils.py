import torch
import logging

from crlhf.data.safety_data import get_hidden_states_dataloader
from crlhf.polytope.lm_constraints import PolytopeConstraint

log = logging.getLogger(__name__)

def load_data_and_model(args):
    """
    Loads the dataset(s) and a polytope safety model from disk.
    Returns the model plus a dictionary of dataloaders & input_texts.
    """
    log.info(f"Loading trained weights from {args.trained_weights_path}")
    trained_weights = torch.load(args.trained_weights_path, weights_only=False)

    log.info(f"Loading dataset from {args.hidden_states_path}")
    hs_data = torch.load(args.hidden_states_path, weights_only=False)
    
    datasets = {}
    
    # Check if the data is in the new format (direct hidden_states and input_texts)
    if "hidden_states" in hs_data and "input_texts" in hs_data:
        log.info("Detected new format (direct hidden_states and input_texts)")
        
        # Ensure hidden states are on CPU and properly formatted
        hidden_states = hs_data["hidden_states"]
        if torch.is_tensor(hidden_states):
            hidden_states = hidden_states.cpu()
        else:
            hidden_states = torch.from_numpy(hidden_states).float()
        
        # Reshape hidden states to match expected format
        # Original shape: [batch_size, seq_len, hidden_dim]
        # Expected shape: [batch_size, hidden_dim]
        # We take the last token's hidden state
        hidden_states = hidden_states[:, -1, :]  # Take last token's hidden state
        
        # Structure the data to match the old format
        test_data = {
            "hidden_states": hidden_states,
            "input_texts": hs_data["input_texts"],
            "labels": torch.zeros(len(hidden_states), dtype=torch.float32)  # Create labels on CPU
        }
        
        test_dataloader = get_hidden_states_dataloader(
            test_data,
            shuffle=False,
            batch_size=args.batch_size
        )
        
        datasets["test"] = {
            "dataloader": test_dataloader,
            "input_texts": hs_data["input_texts"]
        }
        
    else:
        log.info("Detected old format (with test/train splits)")
        # Handle test split
        test_dataloader = get_hidden_states_dataloader(
            hs_data["test"],
            shuffle=False,
            batch_size=args.batch_size
        )
        
        datasets["test"] = {
            "dataloader": test_dataloader,
            "input_texts": hs_data["test"]["input_texts"]
        }
        
        # Handle train split if available
        if "train" in hs_data:
            train_dataloader = get_hidden_states_dataloader(
                hs_data["train"],
                shuffle=False,
                batch_size=args.batch_size
            )
            
            datasets["train"] = {
                "dataloader": train_dataloader,
                "input_texts": hs_data["train"]["input_texts"]
            }
    
    # Debug logs for the first batch
    first_batch = next(iter(datasets["test"]["dataloader"]))
    log.info(f"Dataloader batch type: {type(first_batch)}")
    log.info(f"Dataloader batch content types: {[type(item) for item in first_batch]}")
    log.info(f"First batch shapes: {[item.shape if hasattr(item, 'shape') else 'No shape' for item in first_batch]}")
    
    # If the batch contains hidden states, show their statistics
    inputs = first_batch[0]
    log.info(f"Input tensor stats - min: {inputs.min().item()}, max: {inputs.max().item()}, mean: {inputs.mean().item()}")
    log.info(f"Input tensor dtype: {inputs.dtype}")
    log.info(f"Input tensor shape: {inputs.shape}")
    
    log.info(f"Loaded {len(datasets['test']['input_texts'])} test input texts")
    log.info(f"First test input text: {datasets['test']['input_texts'][0]}")
    
    if "train" in datasets:
        log.info(f"Loaded {len(datasets['train']['input_texts'])} train input texts")

    log.info("Initializing safety model...")
    safety_model = PolytopeConstraint(
        model=None,
        tokenizer=None,
        train_on_hs=True,
    )
    safety_model.phi = trained_weights.phi
    safety_model.threshold = trained_weights.threshold
    safety_model.feature_extractor = trained_weights.feature_extractor
    safety_model.to(safety_model.device)
    log.info("Safety model initialized successfully")

    return safety_model, datasets
