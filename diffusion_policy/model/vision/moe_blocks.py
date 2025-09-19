# 文件: diffusion_policy/model/vision/moe_blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import VisionTransformer

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
            self.aux_loss = 0

        output = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            token_indices = (top_k_indices == i).any(dim=-1)
            if token_indices.any():
                selected_tokens = x_flat[token_indices]
                gate_indices = (top_k_indices[token_indices] == i) 
                selected_gates = gates[token_indices][gate_indices].unsqueeze(-1)
                expert_output = self.experts[i](selected_tokens)
                output[token_indices] += expert_output * selected_gates

        return output.view(batch_size, seq_len, -1)

class MoEVisionTransformer(VisionTransformer):
    """
    一个继承自 timm.VisionTransformer 的自定义类，
    它在初始化时就将指定的 FFN 层替换为 MoE 层。
    """
    def __init__(self, moe_layers: list, moe_d_ffn: int, moe_num_experts: int, moe_top_k: int, **kwargs):
        # 首先，调用父类 (原始 ViT) 的 __init__ 方法，让它构建一个标准的 ViT
        super().__init__(**kwargs)
        
        # 现在，self.blocks 已经是一个包含标准 Block 的 ModuleList 了
        # 我们可以对它进行“手术”
        for i in moe_layers:
            original_mlp = self.blocks[i].mlp
            d_model = original_mlp.fc1.in_features

            self.blocks[i].mlp = MoEFeedForward(
                d_model=d_model,
                d_ffn=moe_d_ffn,
                num_experts=moe_num_experts,
                top_k=moe_top_k
            )