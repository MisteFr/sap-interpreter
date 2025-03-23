import torch
import logging

log = logging.getLogger(__name__)
STEER_LAYER = 36

def compute_token_level_violations(safety_model, tokenizer, text, device):
    """
    Computes token-level violation scores for each facet of the polytope model.
    Returns a tuple (tokens, violations_tensor).
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    token_ids = inputs["input_ids"][0]  # assume batch size 1
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    with torch.no_grad():
        outputs = safety_model.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[STEER_LAYER].to(torch.float32)
        
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

def compute_token_level_activations(safety_model, tokenizer, text, device):
    """
    Computes token-level SAE/SAP activations for each token in the input text.
    Returns a tuple (tokens, activations_tensor).
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    token_ids = inputs["input_ids"][0]  # assume batch size 1
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    with torch.no_grad():
        outputs = safety_model.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[STEER_LAYER].to(torch.float32)
        
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
