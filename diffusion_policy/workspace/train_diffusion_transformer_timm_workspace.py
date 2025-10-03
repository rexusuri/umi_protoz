# filename: train_diffusion_transformer_timm_workspace.py (FINAL & CORRECTED)

if __name__ == "__main__":
    import sys, os, pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
import pickle
import wandb
import deepspeed
from contextlib import nullcontext

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_timm_policy import DiffusionTransformerTimmPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.vision.moe_blocks import DeepSpeedMoEWrapper

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerTimmWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        seed = cfg.training.seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.model: DiffusionTransformerTimmPolicy = hydra.utils.instantiate(cfg.policy)
        self.optimizer = None
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        is_main_process = not deepspeed.comm.is_initialized() or deepspeed.comm.get_rank() == 0

        if is_main_process:
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            project = wandb_cfg.pop('project', 'deepspeed_run')
            wandb.init(project=project, config=OmegaConf.to_container(cfg, resolve=True), **wandb_cfg)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if is_main_process:
            normalizer = dataset.get_normalizer()
            with open(normalizer_path, 'wb') as f: pickle.dump(normalizer, f)
        
        if deepspeed.comm.is_initialized(): torch.distributed.barrier()
        with open(normalizer_path, 'rb') as f: normalizer = pickle.load(f)

        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        
        self.model.set_normalizer(normalizer)
        
        model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            config=cfg.training.deepspeed_config,
            model_parameters=self.model.get_optimizer_params(cfg.optimizer),
        )
        
        device = model_engine.device

        # ===================================================================
        # ==================== FINAL HOOKS FOR EXPERT LOGGING =================
        # ===================================================================
        moe_expert_counts = []
        def capture_expert_counts_hook(module, input, output):
            # The router output is a tuple, and the 4th element (index 3) is the expert counts
            if isinstance(output, tuple) and len(output) > 3 and isinstance(output[3], torch.Tensor):
                moe_expert_counts.append(output[3])

        if is_main_process:
            # Find all MoE router modules and register the hook
            for module in model_engine.module.modules():
                if isinstance(module, DeepSpeedMoEWrapper):
                    router = module.moe_layer.deepspeed_moe.gate
                    router.register_forward_hook(capture_expert_counts_hook)
        # ===================================================================

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        logger_context = JsonLogger(log_path) if is_main_process else nullcontext()

        with logger_context as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                model_engine.train()
                
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                               disable=not is_main_process,
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch in tepoch:
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        loss = model_engine(batch)
                        
                        total_aux_loss = torch.tensor(0.0, device=device)
                        for module in model_engine.module.modules():
                            if isinstance(module, DeepSpeedMoEWrapper):
                                if hasattr(module.moe_layer, 'l_aux'):
                                    total_aux_loss += module.moe_layer.l_aux
                        
                        model_engine.backward(loss)
                        model_engine.step()

                        if is_main_process:
                            loss_cpu = loss.item()
                            aux_loss_cpu = total_aux_loss.item()
                            main_loss_cpu = loss_cpu - aux_loss_cpu
                            
                            tepoch.set_postfix(loss=main_loss_cpu, aux_loss=aux_loss_cpu, refresh=False)
                            
                            step_log = {
                                'train_loss': main_loss_cpu,
                                'aux_loss': aux_loss_cpu,
                                'total_loss': loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': model_engine.get_lr()[0]
                            }

                            # Process and log expert utilization periodically
                            if self.global_step > 0 and self.global_step % 100 == 0 and len(moe_expert_counts) > 0:
                                # Sum the counts from all MoE layers for this step
                                all_counts_tensor = torch.stack(moe_expert_counts).sum(dim=0).detach().cpu()
                                total_routed = all_counts_tensor.sum()

                                if total_routed > 0:
                                    expert_util_percent = (all_counts_tensor / total_routed) * 100
                                    
                                    table_data = [[f"Expert {i}", util.item()] for i, util in enumerate(expert_util_percent)]
                                    util_table = wandb.Table(columns=["Expert ID", "Utilization (%)"], data=table_data)
                                    step_log["Expert Utilization"] = wandb.plot.bar(
                                        util_table, "Expert ID", "Utilization (%)", title="Expert Utilization per 100 Steps"
                                    )
                                
                                # Clear the list for the next logging interval
                                moe_expert_counts.clear()
                            
                            wandb.log(step_log, step=self.global_step)
                            if json_logger is not None:
                                json_logger.log(step_log)
                            
                            self.global_step += 1

                self.epoch += 1