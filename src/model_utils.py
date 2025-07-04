import torch
import logging

log = logging.getLogger(__name__)

# facet level violations
def compute_token_level_violations(safety_model, tokenizer, text, device, steer_layer=30):
    """
    Computes token-level violation scores for each facet of the polytope model.
    Args:
        safety_model: The safety model to use
        tokenizer: The tokenizer to use
        text: Input text to analyze
        device: Device to run computations on
        steer_layer: Which transformer layer to extract hidden states from (default: 30)
    Returns:
        A tuple (tokens, violations_tensor).
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    token_ids = inputs["input_ids"][0]  # assume batch size 1
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    with torch.no_grad():
        outputs = safety_model.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[steer_layer].to(torch.float32)
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        all_violations = []
        
        for i in range(seq_len):
            token_hs = hidden_states[:, i, :].squeeze(0)  # [hidden_dim]
            token_hs = token_hs.unsqueeze(0)              # [1, hidden_dim]
            
            features = safety_model.feature_extractor(token_hs)
            token_violations = torch.matmul(safety_model.phi, features.T) - safety_model.threshold.unsqueeze(1)
            all_violations.append(token_violations.squeeze(1).cpu())
        
        violations_tensor = torch.stack(all_violations)
        return tokens, violations_tensor

# concept encoder level activations
def compute_token_level_activations(safety_model, tokenizer, text, device, steer_layer=30):
    """
    Computes token-level SAE/SAP activations for each token in the input text.
    Args:
        safety_model: The safety model to use
        tokenizer: The tokenizer to use
        text: Input text to analyze
        device: Device to run computations on
        steer_layer: Which transformer layer to extract hidden states from (default: 30)
    Returns:
        A tuple (tokens, activations_tensor).
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    token_ids = inputs["input_ids"][0]  # assume batch size 1
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    with torch.no_grad():
        outputs = safety_model.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[steer_layer].to(torch.float32)
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        all_activations = []
        
        for i in range(seq_len):
            token_hs = hidden_states[:, i, :].squeeze(0)  # [hidden_dim]
            token_hs = token_hs.unsqueeze(0)              # [1, hidden_dim]
            
            # Extract features using the safety model's feature extractor
            features = safety_model.feature_extractor(token_hs)
            all_activations.append(features.squeeze(0).cpu())
        
        activations_tensor = torch.stack(all_activations)
        return tokens, activations_tensor
