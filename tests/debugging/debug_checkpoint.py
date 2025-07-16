#!/usr/bin/env python3
import torch

checkpoint = torch.load('outputs/checkpoints/best_model.pt', map_location='cpu')
print('Checkpoint keys:', list(checkpoint.keys()))
for k, v in checkpoint.items():
    if k != 'model_state_dict':
        print(f'{k}: {v}')

# Check positional encoding shape
state_dict = checkpoint['model_state_dict']
pe_shape = state_dict['embedding.positional_encoding.pe'].shape
print(f'Positional encoding shape: {pe_shape}')
print(f'Max sequence length from PE: {pe_shape[1]}')