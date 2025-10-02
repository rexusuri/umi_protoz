# filename: train_diffusion_transformer_timm_workspace.py (FINAL Fix for JsonLogger)

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
from contextlib import nullcontext # <--- 新增导入

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_timm_policy import DiffusionTransformerTimmPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerTimmWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.model: DiffusionTransformerTimmPolicy = hydra.utils.instantiate(cfg.policy)
        self.optimizer = None
        self.global_step = 0
        self.epoch = 0
        if not cfg.training.resume:
            self.exclude_keys = []

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        is_main_process = not deepspeed.comm.is_initialized() or deepspeed.comm.get_rank() == 0

        if is_main_process:
            with open_dict(cfg):
                cfg.deepspeed_config = cfg.training.deepspeed_config
            
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            project = wandb_cfg.pop('project', 'deepspeed_run')
            wandb.init(project=project, config=OmegaConf.to_container(cfg, resolve=True), **wandb_cfg)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if is_main_process:
            normalizer = dataset.get_normalizer()
            with open(normalizer_path, 'wb') as f:
                pickle.dump(normalizer, f)
        
        if deepspeed.comm.is_initialized():
            torch.distributed.barrier()
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)

        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        
        model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            config=cfg.training.deepspeed_config,
            model_parameters=self.model.get_optimizer_params(cfg.optimizer),
        )
        
        device = model_engine.device

        env_runner: BaseImageRunner = None
        if is_main_process:
            env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)

        if cfg.training.resume:
            load_path, client_state = model_engine.load_checkpoint(self.output_dir)
            if load_path is not None:
                self.global_step = client_state.get('global_step', 0)
                self.epoch = client_state.get('epoch', 0)
                print(f"Resumed training from checkpoint {load_path} at global_step {self.global_step}")
        
        # ===================================================================
        # ====================  这里是核心的修改点  =====================
        # ===================================================================
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        # On main process, create a real JsonLogger. On others, create a dummy context.
        logger_context = JsonLogger(log_path) if is_main_process else nullcontext()

        with logger_context as json_logger:
            # The entire training loop is now inside the 'with' block
            for local_epoch_idx in range(cfg.training.num_epochs):
                model_engine.train()
                
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                               disable=not is_main_process,
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        # Cast batch to the model's dtype (e.g., float16)
                        if model_engine.fp16_enabled(): # check if fp16 is enabled in the DeepSpeed engine
                            batch = dict_apply(batch, lambda x: x.to(model_engine.dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x)

                        loss = model_engine(batch)
                        # In stage 0, aux_loss is not returned automatically. We can set it to 0.
                        aux_loss = torch.tensor(0.0)

                        model_engine.backward(loss)
                        model_engine.step()

                        if is_main_process:
                            raw_loss_cpu = loss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'aux_loss': aux_loss.item(),
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': model_engine.get_lr()[0]
                            }
                            
                            wandb.log(step_log, step=self.global_step)
                            json_logger.log(step_log) # json_logger is valid here
                            
                            self.global_step += 1

                if is_main_process:
                    # ... (end of epoch logic remains the same)
                    pass

                self.epoch += 1
        # ===================================================================