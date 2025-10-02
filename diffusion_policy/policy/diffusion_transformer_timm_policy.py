from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_action_diffusion import TransformerForActionDiffusion
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.transformer_obs_encoder import TransformerObsEncoder

class DiffusionTransformerTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TransformerObsEncoder,
            num_inference_steps=None,
            input_pertub=0.1,
            use_dpm_solver: bool = False,
            dpm_solver_order: int = 2,
            n_layer=7,
            n_head=8,
            n_emb=768,
            p_drop_attn=0.1,
            **kwargs):
        super().__init__()

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
            max_cond_tokens=obs_tokens+1,
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
    
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        if getattr(self, 'use_dpm_solver', False):
            try:
                dpm_scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
            except Exception:
                dpm_scheduler = DPMSolverMultistepScheduler()
            if hasattr(dpm_scheduler, 'config') and hasattr(dpm_scheduler.config, 'solver_order'):
                dpm_scheduler.config.solver_order = int(getattr(self, 'dpm_solver_order', 2))
            if hasattr(dpm_scheduler, 'config') and getattr(dpm_scheduler.config, 'prediction_type', None) == 'sample':
                dpm_scheduler.config.prediction_type = 'epsilon'
            scheduler = dpm_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, cond)
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        trajectory[condition_mask] = condition_data[condition_mask]        
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        
        obs_tokens = self.obs_encoder(nobs)
        
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            cond=obs_tokens,
            **self.kwargs)
        
        action_pred = self.normalizer['action'].unnormalize(nsample)

        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer_params(self, optimizer_cfg: dict):
        # The simplified version is correct for letting DeepSpeed manage parameters.
        return [p for p in self.parameters() if p.requires_grad]

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        trajectory = nactions
        
        obs_tokens = self.obs_encoder(nobs)
        
        # Noise tensors are created as default float32, which is now correct
        # for a non-FP16 run.
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        noise_new = noise + self.input_pertub * torch.randn(
            trajectory.shape, device=trajectory.device)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()
        
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
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

        main_loss = F.mse_loss(pred, target)
        return main_loss

    def forward(self, batch):
        return self.compute_loss(batch)