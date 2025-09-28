# filename: moe_blocks.py (DeepSpeed Version)

import torch.nn as nn
import deepspeed

# 这是一个标准的FFN模块，我们将把它作为“专家”传递给DeepSpeed
class FFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

# 这是我们的新替换函数
def replace_ffn_with_deepspeed_moe(model, layers_to_replace, num_experts=8, top_k=2):
    """
    一个辅助函数，用于将指定ViT block的FFN (Mlp) 替换为 DeepSpeed MoE 层。
    
    注意：DeepSpeed MoE层会自动处理负载均衡损失，我们之后需要在训练脚本中获取它。
    """
    for i in layers_to_replace:
        original_mlp = model.blocks[i].mlp
        d_model = original_mlp.fc1.in_features
        d_ffn = original_mlp.fc1.out_features
        
        # 创建一个标准的FFN作为专家模板
        expert = FFN(d_model=d_model, d_ffn=d_ffn)
        
        # 用DeepSpeed的MoE层替换原有的mlp属性
        # hidden_size: 输入/输出特征维度
        # expert: 传入一个专家模块的实例
        # num_experts: 专家总数
        # k: top_k路由
        model.blocks[i].mlp = deepspeed.moe.layer.MoE(
            hidden_size=d_model,
            expert=expert,
            num_experts=num_experts,
            k=top_k
        )
        
    return model