import torch
import torch.nn as nn

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

        # 路由器
        self.router = nn.Linear(d_model, num_experts)

        # 专家们 (每个专家都是一个标准的前馈网络)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.GELU(),
                nn.Linear(d_ffn, d_model)
            ) for _ in range(num_experts)
        ])
        
        # 重要的属性：用于在外部访问辅助损失
        self.aux_loss = None

    def forward(self, x):
        # x shape: [batch_size, sequence_length, d_model]
        # 在ViT中，sequence_length 通常是 num_patches + 1 (for CLS token)
        
        batch_size, seq_len, _ = x.shape
        # 将输入展平以进行路由
        x_flat = x.view(-1, self.d_model) # [B * N, d_model]
        
        # 1. 路由
        router_logits = self.router(x_flat) # [B * N, num_experts]
        
        # 2. 计算 Top-K 门控值 (gates) 和 选中的专家索引
        # topk返回 (values, indices)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1) # [B * N, k]
        gates = F.softmax(top_k_logits, dim=-1) # [B * N, k]

        # 3. 计算辅助损失 (Load Balancing Loss)
        # 这是为了鼓励路由器将负载均匀分配给所有专家
        if self.training:
            # one_hot编码每个token被分配给哪些专家
            zeros = torch.zeros_like(router_logits, requires_grad=False)
            # scatter_将1填充到被选中的专家的位置
            mask = zeros.scatter(-1, top_k_indices, 1)
            
            # 每个专家被分配的token比例
            tokens_per_expert = torch.mean(mask.float(), dim=0)
            # 路由器为每个专家输出的平均概率
            router_prob_per_expert = torch.mean(F.softmax(router_logits, dim=-1).float(), dim=0)
            
            # 损失 = (每个专家处理的token比例 * 每个专家被路由到的平均概率) 的内积
            # 这个损失鼓励 tokens_per_expert 和 router_prob_per_expert 都接近均匀分布
            self.aux_loss = (tokens_per_expert * router_prob_per_expert).sum() * self.num_experts
            self.aux_loss = self.aux_loss * self.aux_loss_weight
        else:
            self.aux_loss = 0 # 推理时不需要

        # 4. 将 token 分发给对应的专家
        output = torch.zeros_like(x_flat) # [B * N, d_model]
        
        # 这个循环效率不高，但在PyTorch中实现稀疏分发的最直接方式
        # 高性能实现通常需要自定义CUDA核
        for i in range(self.num_experts):
            # 找到被分配给当前专家 i 的 token
            # (top_k_indices == i) 会产生一个布尔掩码 [B * N, k]
            # .any(dim=-1) 检查每个token是否至少有一个k指向专家i
            token_indices = (top_k_indices == i).any(dim=-1)
            
            if token_indices.any():
                # 提取这些 token
                selected_tokens = x_flat[token_indices]
                
                # 提取这些 token 对应的门控值
                # (top_k_indices == i) [B*N, k] -> gate_indices [num_selected_tokens, k]
                gate_indices = (top_k_indices[token_indices] == i) 
                selected_gates = gates[token_indices][gate_indices].unsqueeze(-1)
                
                # 让专家 i 处理这些 token
                expert_output = self.experts[i](selected_tokens)
                
                # 将专家输出乘以其门控值，并放回原位
                output[token_indices] += expert_output * selected_gates

        # 恢复原始形状
        output = output.view(batch_size, seq_len, -1)
        return output

class AdapterMoEBlock(nn.Module):
    """
    用于将标准ViT Block并联一个简单MoE Adapter模块
    可按需改造为FFN-MoE等结构。
    """
    def __init__(self, base_block, adapter_dim=192, num_experts=4):
        super().__init__()
        self.base_block = base_block
        self.adapter_dim = adapter_dim
        self.num_experts = num_experts
        d_model = base_block.mlp.fc1.in_features

        # 每个专家是一个简单的Adapter瓶颈结构
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, d_model)
            ) for _ in range(num_experts)
        ])
        # 路由器，将输入投影到专家权重
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # ViT主干前向
        h = self.base_block(x)
        # 路由 logits
        logits = self.router(h)  # [B, N, num_experts]
        probs = torch.softmax(logits, dim=-1)  # soft routing

        # MoE聚合专家输出
        moe_out = 0
        for i, expert in enumerate(self.experts):
            moe_out = moe_out + probs[..., i:i+1] * expert(h)
        # 残差连接
        return h + moe_out

def replace_vit_blocks_with_moe(model, layers, adapter_dim=192, num_experts=4):
    """
    替换指定ViT block为MoE/Adapter block
    :param model: timm创建的ViT模型
    :param layers: 需要替换的层号列表，例如[8,9,10,11]
    :param adapter_dim: Adapter隐藏层
    :param num_experts: 专家数
    """
    for i in layers:
        model.blocks[i] = AdapterMoEBlock(model.blocks[i], adapter_dim, num_experts)
    return model

# 同样放在 moe_blocks.py

def replace_ffn_with_moe(model, layers, d_ffn, num_experts=8, top_k=2):
    """
    替换指定ViT block的FFN (Mlp) 为 MoE-FFN 层
    :param model: timm创建的ViT模型
    :param layers: 需要替换的层号列表
    """
    for i in layers:
        original_block = model.blocks[i]
        d_model = original_block.mlp.fc1.in_features
        
        # 直接替换 mlp 属性
        original_block.mlp = MoEFeedForward(
            d_model=d_model,
            d_ffn=d_ffn,
            num_experts=num_experts,
            top_k=top_k
        )
    return model

# 注意: 需要从 ViT 的配置中知道 d_ffn 的大小。
# 对于 ViT-Base, d_model=768, d_ffn=3072
# 对于 ViT-Large, d_model=1024, d_ffn=4096
# 简单起见，可以写死或者从模型配置中读取。