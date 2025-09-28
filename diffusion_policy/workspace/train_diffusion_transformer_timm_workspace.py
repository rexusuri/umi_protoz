if __name__ == "__main__":
    import sys
    import os
    import pathlib

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

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_timm_policy import DiffusionTransformerTimmPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.diffusion.ema_model import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerTimmWorkspace(BaseWorkspace):
    # DeepSpeed manages its own state for checkpointing
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerTimmPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        # The optimizer is now created by DeepSpeed during initialization
        self.optimizer = None

        # training state
        self.global_step = 0
        self.epoch = 0
        
        if not cfg.training.resume:
            self.exclude_keys = []

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        # Initialize DeepSpeed Distributed Training
        deepspeed.init_distributed()
        
        # Add deepspeed config to the main config for logging purposes
        with open_dict(cfg):
            cfg.deepspeed_config = cfg.training.deepspeed_config
        
        # Initialize WandB only on the main process (rank 0)
        if deepspeed.comm.get_rank() == 0:
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            # project name is passed directly to init
            if 'project' in wandb_cfg:
                wandb_project = wandb_cfg.pop('project')
            else:
                wandb_project = 'deepspeed_run'
            
            wandb.init(
                project=wandb_project,
                config=OmegaConf.to_container(cfg, resolve=True),
                **wandb_cfg
            )

        # Configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        
        # Normalizer should be computed on rank 0 and then loaded by all processes
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if deepspeed.comm.get_rank() == 0:
            normalizer = dataset.get_normalizer()
            with open(normalizer_path, 'wb') as f:
                pickle.dump(normalizer, f)
        
        # Synchronize all processes to make sure the file is written
        torch.distributed.barrier()
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)

        # Configure dataloaders
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        
        # DeepSpeed initialization
        model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            config=cfg.training.deepspeed_config,
            model_parameters=self.model.get_optimizer_params(cfg.optimizer),
        )
        
        device = model_engine.device

        # Configure env runner and checkpoint manager on the main process only
        env_runner: BaseImageRunner = None
        topk_manager: TopKCheckpointManager = None
        if deepspeed.comm.get_rank() == 0:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)
            
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

        # Resume training using DeepSpeed's loading mechanism
        if cfg.training.resume:
            load_path, client_state = model_engine.load_checkpoint(self.output_dir)
            if load_path is not None:
                self.global_step = client_state.get('global_step', 0)
                self.epoch = client_state.get('epoch', 0)
                print(f"Resumed training from checkpoint {load_path} at global_step {self.global_step}")

        # Training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger = JsonLogger(log_path) if deepspeed.comm.get_rank() == 0 else None
        
        for local_epoch_idx in range(cfg.training.num_epochs):
            model_engine.train()

            if cfg.training.freeze_encoder:
                # Access the original model via model_engine.module
                model_engine.module.obs_encoder.eval()
                model_engine.module.obs_encoder.requires_grad_(False)

            train_losses = list()
            # Disable tqdm on all processes except the main one
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                           disable=(deepspeed.comm.get_rank() != 0),
                           leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # Move data to the correct device
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    
                    # DeepSpeed MoE layer automatically returns (total_loss, aux_loss)
                    # We need to ensure the policy's forward method is updated for this
                    loss, aux_loss = model_engine(batch)
                    
                    # Backward pass
                    model_engine.backward(loss)
                    
                    # Optimizer step
                    model_engine.step()

                    # Logging on the main process
                    if deepspeed.comm.get_rank() == 0:
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
                        if json_logger:
                            json_logger.log(step_log)
                        
                        self.global_step += 1

            # End of epoch evaluation and checkpointing on the main process
            if deepspeed.comm.get_rank() == 0:
                train_loss_mean = np.mean(train_losses)
                epoch_end_log = {'train_loss_epoch': train_loss_mean}

                policy = model_engine.module  # Get the unwrapped model for evaluation
                policy.eval()

                # Run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and env_runner is not None:
                    runner_log = env_runner.run(policy)
                    epoch_end_log.update(runner_log)
                
                # Checkpointing
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    client_state = {
                        'global_step': self.global_step,
                        'epoch': self.epoch
                    }
                    model_engine.save_checkpoint(self.output_dir, client_state=client_state)

                wandb.log(epoch_end_log, step=self.global_step)
                if json_logger:
                    json_logger.log(epoch_end_log)
                
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerTimmWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()