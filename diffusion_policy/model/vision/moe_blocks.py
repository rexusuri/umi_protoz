import torch
import torch.nn as nn

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

        # 每个专家是一个简单的Adapter瓶颈结构
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_block.mlp.fc2.in_features, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, base_block.mlp.fc2.in_features)
            ) for _ in range(num_experts)
        ])
        # 路由器，将输入投影到专家权重
        self.router = nn.Linear(base_block.mlp.fc2.in_features, num_experts)

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