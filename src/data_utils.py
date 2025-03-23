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
    dataloader = get_hidden_states_dataloader(hs_data["test"], shuffle=True)
    
    print(hs_data["test"])
    input_texts = hs_data["test"]["input_texts"]
    
    # Debug logs for the first batch
    first_batch = next(iter(dataloader))
    log.info(f"Dataloader batch type: {type(first_batch)}")
    log.info(f"Dataloader batch content types: {[type(item) for item in first_batch]}")
    log.info(f"First batch shapes: {[item.shape if hasattr(item, 'shape') else 'No shape' for item in first_batch]}")
    
    inputs = first_batch[0]
    log.info(f"Input tensor stats - min: {inputs.min().item()}, max: {inputs.max().item()}, mean: {inputs.mean().item()}")
    
    log.info(f"Loaded {len(input_texts)} input texts")
    log.info(f"First input text: {input_texts[0]}")
    
    # Prepare test dataloader
    test_dataloader = get_hidden_states_dataloader(
        hs_data["test"],
        shuffle=False,
        batch_size=args.batch_size
    )
    
    datasets = {
        "test": {
            "dataloader": test_dataloader,
            "input_texts": hs_data["test"]["input_texts"]
        }
    }
    
    train_dataloader = get_hidden_states_dataloader(
        hs_data["train"],
        shuffle=False,
        batch_size=args.batch_size
    )
    first_train_batch = next(iter(train_dataloader))
    log.info(f"Train dataloader batch type: {type(first_train_batch)}")
    log.info(f"Train batch shapes: {[item.shape if hasattr(item, 'shape') else 'No shape' for item in first_train_batch]}")
    
    # Reset train dataloader (because we consumed its first batch)
    train_dataloader = get_hidden_states_dataloader(
        hs_data["train"],
        shuffle=False,
        batch_size=args.batch_size
    )
    
    log.info(f"Loaded {len(hs_data['train']['input_texts'])} train input texts")
    
    datasets["train"] = {
        "dataloader": train_dataloader,
        "input_texts": hs_data["train"]["input_texts"]
    }

    log.info("Initializing safety model...")
    safety_model = PolytopeConstraint(model=None, tokenizer=None, train_on_hs=True)
    
    # Transfer learned weights
    safety_model.phi = trained_weights.phi
    safety_model.threshold = trained_weights.threshold
    safety_model.feature_extractor = trained_weights.feature_extractor
    safety_model.to(safety_model.device)
    log.info("Safety model initialized successfully")

    return safety_model, datasets
