# 文件: diffusion_policy/model/vision/transformer_obs_encoder.py

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
from .moe_blocks import MoEVisionTransformer

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
            out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True, training=self.training, need_weights=False
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
            # MoE parameters from YAML
            use_moe: bool=False,
            moe_layers: list=None,
            moe_d_ffn: int=None,
            moe_num_experts: int=8,
            moe_top_k: int=2,
            **kwargs
        ):
        super().__init__()
        
        rgb_keys = list(); low_dim_keys = list(); key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict(); key_projection_map = nn.ModuleDict(); key_shape_map = dict()

        assert global_pool == ''

        if use_moe:
            if not model_name.startswith('vit'): raise ValueError("MoE is only supported for ViT models.")
            logger.info("Instantiating custom MoEVisionTransformer.")
            model_cfg = vars(timm.models.vision_transformer.default_cfgs[model_name])
            model_cfg.pop('tags', None)
            model_cfg.update(dict(pretrained=False, global_pool=global_pool, num_classes=0)) # pretrained is handled separately
            model_template = MoEVisionTransformer(
                moe_layers=moe_layers, moe_d_ffn=moe_d_ffn,
                moe_num_experts=moe_num_experts, moe_top_k=moe_top_k, **model_cfg
            )
            if pretrained:
                # Load pretrained weights, ignoring size mismatches in MLP layers we replaced
                model_template.load_pretrained(timm.models.vision_transformer.default_cfgs[model_name].url, strict=False)
        else:
            logger.info("Instantiating standard timm model.")
            model_template = timm.create_model(
                model_name=model_name, pretrained=pretrained,
                global_pool=global_pool, num_classes=0
            )
        self.model_name = model_name

        if frozen:
            assert pretrained
            for param in model_template.parameters(): param.requires_grad = False
        
        feature_dim = None
        temp_model_for_slicing = model_template
        if model_name.startswith('resnet'):
            if downsample_ratio == 32: modules = list(temp_model_for_slicing.children())[:-2]; feature_dim = 512
            elif downsample_ratio == 16: modules = list(temp_model_for_slicing.children())[:-3]; feature_dim = 256
            else: raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
            temp_model_for_slicing = torch.nn.Sequential(*modules)
        elif model_name.startswith('convnext'):
            if downsample_ratio == 32: modules = list(temp_model_for_slicing.children())[:-2]; feature_dim = 1024
            else: raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
            temp_model_for_slicing = torch.nn.Sequential(*modules)
        
        if use_group_norm and not pretrained:
            temp_model_for_slicing = replace_submodules(
                root_module=temp_model_for_slicing, predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=(x.num_features//16) if (x.num_features%16==0) else (x.num_features//8), num_channels=x.num_features)
            )
            
        self.feature_aggregation = feature_aggregation
        if model_name.startswith('vit'):
            if self.feature_aggregation is None: pass
            elif self.feature_aggregation != 'cls': logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!'); self.feature_aggregation = 'cls'
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb': assert image_shape is None or image_shape == shape[1:]; image_shape = shape[1:]
        
        feature_map_shape = [x // downsample_ratio for x in image_shape] if image_shape is not None else None

        if self.feature_aggregation == 'soft_attention': self.attention = nn.Sequential(nn.Linear(feature_dim, 1, bias=False), nn.Softmax(dim=1))
        elif self.feature_aggregation == 'spatial_embedding': self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0]*feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'attention_pool_2d': self.attention_pool_2d = AttentionPool2d(spacial_dim=feature_map_shape[0], embed_dim=feature_dim, num_heads=feature_dim//64, output_dim=feature_dim)
        
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'; ratio = transforms[0].ratio
            transforms = [torchvision.transforms.RandomCrop(size=int(image_shape[0]*ratio)), torchvision.transforms.Resize(size=image_shape[0], antialias=True)] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape']); obs_type = attr.get('type', 'low_dim'); key_shape_map[key] = shape
            if obs_type == 'rgb':
                rgb_keys.append(key)
                this_model = model_template if share_rgb_model else copy.deepcopy(model_template)
                key_model_map[key] = this_model
                
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_size = example_features.shape[-1]
                proj = nn.Identity()
                if feature_size != n_emb: proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj
                key_transform_map[key] = transform
            elif obs_type == 'low_dim':
                low_dim_keys.append(key); dim = np.prod(shape); proj = nn.Identity()
                if dim != n_emb: proj = nn.Linear(in_features=dim, out_features=n_emb)
                key_projection_map[key] = proj
            else: raise RuntimeError(f"Unsupported obs type: {obs_type}")
        
        self.rgb_keys = sorted(rgb_keys); self.low_dim_keys = sorted(low_dim_keys); self.n_emb = n_emb
        self.shape_meta = shape_meta; self.key_model_map = key_model_map; self.key_transform_map = key_transform_map
        self.key_projection_map = key_projection_map; self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys; self.low_dim_keys = low_dim_keys; self.key_shape_map = key_shape_map
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def aggregate_feature(self, feature):
        if self.model_name.startswith('vit'):
            if self.feature_aggregation == 'cls': return feature[:, [0], :]
            assert self.feature_aggregation is None; return feature
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d': return self.attention_pool_2d(feature)
        feature = torch.flatten(feature, start_dim=-2); feature = torch.transpose(feature, 1, 2)
        if self.feature_aggregation == 'avg': return torch.mean(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'max': return torch.amax(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'soft_attention': weight = self.attention(feature); return torch.sum(feature * weight, dim=1, keepdim=True)
        elif self.feature_aggregation == 'spatial_embedding': return torch.mean(feature * self.spatial_embedding, dim=1, keepdim=True)
        else: assert self.feature_aggregation is None; return feature
        
    def forward(self, obs_dict):
        embeddings = list(); batch_size = next(iter(obs_dict.values())).shape[0]
        for key in self.rgb_keys:
            img = obs_dict[key]; B, T = img.shape[:2]; img = img.reshape(B*T, *img.shape[2:])
            img = self.key_transform_map[key](img); raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature); emb = self.key_projection_map[key](feature)
            emb = emb.reshape(B,-1,self.n_emb); embeddings.append(emb)
        for key in self.low_dim_keys:
            data = obs_dict[key]; B, T = data.shape[:2]; data = data.reshape(B,T,-1)
            emb = self.key_projection_map[key](data); embeddings.append(emb)
        result = torch.cat(embeddings, dim=1); return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict(); obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros((1, attr['horizon']) + shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        return example_output.shape