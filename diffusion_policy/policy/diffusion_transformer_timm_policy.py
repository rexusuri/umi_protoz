from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_action_diffusion import TransformerForActionDiffusion
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.transformer_obs_encoder import TransformerObsEncoder
from diffusion_policy.model.vision.moe_blocks import MoEFeedForward


class DiffusionTransformerTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TransformerObsEncoder,
            num_inference_steps=None,
            input_pertub=0.1,
            use_dpm_solver: bool = False,
            dpm_solver_order: int = 2,
            # arch
            n_layer=7,
            n_head=8,
            n_emb=768,
            p_drop_attn=0.1,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        
        obs_shape = obs_encoder.output_shape()
        assert obs_shape[-1] == n_emb
        obs_tokens = obs_shape[-2]
        
        model = TransformerForActionDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=obs_tokens+1, # obs tokens + 1 token for time
            p_drop_attn=p_drop_attn
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.input_pertub = input_pertub
        self.kwargs = kwargs
        self.use_dpm_solver = use_dpm_solver
        self.dpm_solver_order = dpm_solver_order

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        # Optionally switch to DPM-Solver++ for faster inference
        if getattr(self, 'use_dpm_solver', False):
            # Build a DPM-Solver scheduler that reuses the training scheduler's config (betas, prediction_type, etc.)
            try:
                dpm_scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
            except Exception:
                # Fallback: construct with defaults if the training scheduler lacks a config
                dpm_scheduler = DPMSolverMultistepScheduler()
            # Enforce solver order if available (2 or 3 are typical)
            if hasattr(dpm_scheduler, 'config') and hasattr(dpm_scheduler.config, 'solver_order'):
                dpm_scheduler.config.solver_order = int(getattr(self, 'dpm_solver_order', 2))
            # DPM-Solver expects 'epsilon' or 'v_prediction'; guard against 'sample'
            if hasattr(dpm_scheduler, 'config') and getattr(dpm_scheduler.config, 'prediction_type', None) == 'sample':
                dpm_scheduler.config.prediction_type = 'epsilon'
            scheduler = dpm_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        
        # process input
        obs_tokens = self.obs_encoder(nobs)
        # (B, N, n_emb)
        
        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        
        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            cond=obs_tokens,
            **self.kwargs)
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float,
            obs_encoder_weight_decay: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=weight_decay)
        
        backbone_params = list()
        other_obs_params = list()
        for key, value in self.obs_encoder.named_parameters():
            if key.startswith('key_model_map'):
                backbone_params.append(value)
            else:
                other_obs_params.append(value)
        optim_groups.append({
            "params": backbone_params,
            "weight_decay": obs_encoder_weight_decay,
            "lr": obs_encoder_lr # for fine tuning
        })
        optim_groups.append({
            "params": other_obs_params,
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        trajectory = nactions
        
        # process input
        # 这一步前向传播会触发 MoEFeedForward 模块的 forward 方法，
        # 从而计算并缓存 self.aux_loss
        obs_tokens = self.obs_encoder(nobs)
        # (B, N, n_emb)
        
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            cond=obs_tokens
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # 计算主损失 (Diffusion Loss)
        main_loss = F.mse_loss(pred, target, reduction='none')
        main_loss = main_loss.type(main_loss.dtype)
        main_loss = reduce(main_loss, 'b ... -> b (...)', 'mean')
        main_loss = main_loss.mean()

        # ###############################################################
        # ############## START OF MODIFICATION ##########################
        # ###############################################################

        # 收集并累加所有MoE模块的辅助损失
        aux_loss = 0.0
        # self.obs_encoder 就是 TransformerObsEncoder 的实例
        # 它的 key_model_map 属性包含了 ViT 模型
        # 注意: timm ViT 模型的实例变量名是 model_name
        for model in self.obs_encoder.key_model_map.values():
            if 'vit' in getattr(model, 'model_name', ''):
                for block in model.blocks:
                    # 检查FFN层是否是我们的MoE模块
                    if isinstance(block.mlp, MoEFeedForward):
                        if hasattr(block.mlp, 'aux_loss') and block.mlp.aux_loss is not None:
                            aux_loss += block.mlp.aux_loss
        
        # 将主损失和辅助损失相加
        total_loss = main_loss + aux_loss

        # ###############################################################
        # ############### END OF MODIFICATION ###########################
        # ###############################################################

        return total_loss

    def forward(self, batch):
        return self.compute_loss(batch)
