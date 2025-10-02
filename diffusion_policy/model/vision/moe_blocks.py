# filename: moe_blocks.py (DEFINITIVE WRAPPER Version)

import torch
import torch.nn as nn
import deepspeed

# This is the standard FFN module for the expert
class FFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

# This wrapper makes the DeepSpeed MoE layer compatible with timm's ViT
class DeepSpeedMoEWrapper(nn.Module):
    def __init__(self, d_model, d_ffn, num_experts=8, top_k=2):
        super().__init__()
        # We create the actual DeepSpeed MoE layer inside this wrapper
        expert = FFN(d_model=d_model, d_ffn=d_ffn)
        self.moe_layer = deepspeed.moe.layer.MoE(
            hidden_size=d_model,
            expert=expert,
            num_experts=num_experts,
            k=top_k
        )
    
    def forward(self, x):
        # Call the MoE layer, which returns a tuple (output, loss, ...)
        result = self.moe_layer(x)
        
        # IMPORTANT: We only return the FIRST element (the output tensor) to the timm block.
        # The loss is handled automatically by the DeepSpeed engine in the background.
        output = result[0]
        
        return output

def replace_ffn_with_deepspeed_moe(model, layers_to_replace, num_experts=8, top_k=2):
    """
    Replaces the FFN (Mlp) in specified ViT blocks with our DeepSpeedMoEWrapper.
    """
    for i in layers_to_replace:
        original_mlp = model.blocks[i].mlp
        d_model = original_mlp.fc1.in_features
        d_ffn = original_mlp.fc1.out_features
        
        # Replace the mlp attribute with our new wrapper class
        model.blocks[i].mlp = DeepSpeedMoEWrapper(
            d_model=d_model,
            d_ffn=d_ffn,
            num_experts=num_experts,
            top_k=top_k
        )
    return model