import copy
import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules
from .moe_blocks import replace_ffn_with_deepspeed_moe

logger = logging.getLogger(__name__)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class TransformerObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str='vit_base_patch16_clip_224.openai',
            global_pool: str='',
            transforms: list=None,
            n_emb: int=768,
            pretrained: bool=False,
            frozen: bool=False,
            use_group_norm: bool=False,
            share_rgb_model: bool=False,
            feature_aggregation: str=None,
            downsample_ratio: int=32,
            use_moe: bool=False,
            moe_layers: list=None,
            moe_num_experts: int=8,
            moe_top_k: int=2
        ):
        super().__init__()
        
        self.use_moe = use_moe
        self.moe_layers = moe_layers
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.model_name = model_name
        self.downsample_ratio = downsample_ratio
        self.feature_aggregation = feature_aggregation
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_projection_map = nn.ModuleDict()
        key_shape_map = dict()

        assert global_pool == ''
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool,
            num_classes=0
        )
        
        if self.use_moe and model_name.startswith('vit'):
            if self.moe_layers is None:
                raise ValueError("When use_moe is True, 'moe_layers' must be provided in the YAML config.")
            
            logger.info(f"Replacing ViT layers {self.moe_layers} with DeepSpeed MoE blocks.")
            model = replace_ffn_with_deepspeed_moe(
                model, 
                layers_to_replace=self.moe_layers,
                num_experts=self.moe_num_experts,
                top_k=self.moe_top_k
            )

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
            
        if model_name.startswith('vit'):
            if self.feature_aggregation is None:
                pass
            elif self.feature_aggregation != 'cls':
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = 'cls'
        
        # Hydra already instantiated the transforms for us. We just use them.
        if transforms is not None and len(transforms) > 0:
            transform = nn.Sequential(*transforms)
        else:
            transform = nn.Identity()

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                low_dim_keys.append(key)
        
        self.rgb_keys = sorted(rgb_keys)
        self.low_dim_keys = sorted(low_dim_keys)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                
                if model_name.startswith('vit'):
                    proj = nn.Identity()
                else:
                    with torch.no_grad():
                        example_img = torch.zeros((1,) + tuple(shape))
                        example_feature_map = this_model(example_img)
                        example_features = self.aggregate_feature(example_feature_map)
                        feature_size = example_features.shape[-1]
                    proj = nn.Identity()
                    if feature_size != n_emb:
                        proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                
                key_projection_map[key] = proj
                key_transform_map[key] = transform

            elif type == 'low_dim':
                dim = np.prod(shape)
                proj = nn.Identity()
                if dim != n_emb:
                    proj = nn.Linear(in_features=dim, out_features=n_emb)
                key_projection_map[key] = proj

        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_projection_map = key_projection_map
        self.key_shape_map = key_shape_map
        self.share_rgb_model = share_rgb_model

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def aggregate_feature(self, feature):
        if self.model_name.startswith('vit'):
            if self.feature_aggregation == 'cls':
                return feature[:, [0], :]
            assert self.feature_aggregation is None 
            return feature
        return feature
        
    def forward(self, obs_dict):
        embeddings = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T, C, H, W = img.shape
            img = img.reshape(B*T, C, H, W)
            img = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature)
            emb = self.key_projection_map[key](feature)
            emb = emb.reshape(B, -1, self.n_emb)
            embeddings.append(emb)

        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T, D = data.shape
            data = data.reshape(B*T, D)
            emb = self.key_projection_map[key](data)
            emb = emb.reshape(B, T, self.n_emb)
            embeddings.append(emb)
        
        result = torch.cat(embeddings, dim=1)
        return result

    @torch.no_grad()
    def output_shape(self):
        total_tokens = 0
        
        for key in self.low_dim_keys:
            total_tokens += self.shape_meta['obs'][key]['horizon']

        if len(self.rgb_keys) > 0:
            first_rgb_key = self.rgb_keys[0]
            timm_model = self.key_model_map[first_rgb_key]
            
            num_rgb_tokens_per_image = 0
            if self.model_name.startswith('vit'):
                if self.feature_aggregation == 'cls':
                    num_rgb_tokens_per_image = 1
                else:
                    num_rgb_tokens_per_image = timm_model.patch_embed.num_patches + timm_model.num_prefix_tokens
            else:
                img_shape = self.shape_meta['obs'][first_rgb_key]['shape'][1:]
                feature_map_shape = [x // self.downsample_ratio for x in img_shape]
                if self.feature_aggregation is None:
                    num_rgb_tokens_per_image = feature_map_shape[0] * feature_map_shape[1]
                else:
                    num_rgb_tokens_per_image = 1
            
            total_rgb_tokens = 0
            for key in self.rgb_keys:
                horizon = self.shape_meta['obs'][key]['horizon']
                total_rgb_tokens += num_rgb_tokens_per_image * horizon
            total_tokens += total_rgb_tokens

        return (1, total_tokens, self.n_emb)

