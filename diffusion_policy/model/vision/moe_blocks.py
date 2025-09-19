# filename: moe_blocks.py (修正版)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEFeedForward(nn.Module):
    """
    一个完整的 FFN-MoE 模块，用于替换 ViT 中的 Mlp 层。
    它实现了 Top-K 路由和负载均衡辅助损失。
    """
    def __init__(self, d_model, d_ffn, num_experts=8, top_k=2, aux_loss_weight=1e-2):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.GELU(),
                nn.Linear(d_ffn, d_model)
            ) for _ in range(num_experts)
        ])
        
        self.aux_loss = None
        self.register_buffer('expert_usage', torch.zeros(self.num_experts))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)
        
        router_logits = self.router(x_flat)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        gates = F.softmax(top_k_logits, dim=-1)

        if self.training:
            zeros = torch.zeros_like(router_logits, requires_grad=False)
            mask = zeros.scatter(-1, top_k_indices, 1)
            tokens_per_expert = torch.mean(mask.float(), dim=0)
            router_prob_per_expert = torch.mean(F.softmax(router_logits, dim=-1).float(), dim=0)
            self.aux_loss = (tokens_per_expert * router_prob_per_expert).sum() * self.num_experts
            self.aux_loss = self.aux_loss * self.aux_loss_weight
            
            flat_indices = top_k_indices.flatten()
            self.expert_usage = torch.bincount(flat_indices, minlength=self.num_experts)

        else:
            self.aux_loss = torch.tensor(0.0, device=x.device)

        output = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            token_indices_mask = (top_k_indices == i).any(dim=-1)
            
            if token_indices_mask.any():
                selected_tokens = x_flat[token_indices_mask]
                gate_indices_mask = (top_k_indices[token_indices_mask] == i)
                selected_gates = gates[token_indices_mask][gate_indices_mask].unsqueeze(-1)
                expert_output = self.experts[i](selected_tokens) * selected_gates
                
                # =========================================================
                # ==================== 这里是修改点 =======================
                # =========================================================
                # 使用标准的布尔掩码索引进行原地相加
                output[token_indices_mask] += expert_output

        output = output.view(batch_size, seq_len, -1)
        return output

def replace_ffn_with_moe(model, layers, d_ffn, num_experts=8, top_k=2):
    """
    一个辅助函数，用于替换指定ViT block的FFN (Mlp) 为 MoE-FFN 层
    """
    for i in layers:
        original_block = model.blocks[i]
        d_model = original_block.mlp.fc1.in_features
        
        original_block.mlp = MoEFeedForward(
            d_model=d_model,
            d_ffn=d_ffn,
            num_experts=num_experts,
            top_k=top_k
        )
    return model