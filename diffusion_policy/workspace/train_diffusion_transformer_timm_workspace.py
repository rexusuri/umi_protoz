# 文件: train_diffusion_transformer_timm_workspace.py

if __name__ == "__main__":
    import sys; import os; import pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent); sys.path.append(ROOT_DIR); os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
import pickle
import wandb

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_timm_policy import DiffusionTransformerTimmPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.vision.moe_blocks import MoEFeedForward
from accelerate import Accelerator

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerTimmWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        seed = cfg.training.seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.model: DiffusionTransformerTimmPolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionTransformerTimmPolicy = None
        if cfg.training.use_ema: self.ema_model = copy.deepcopy(self.model)
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.global_step = 0; self.epoch = 0
        if not cfg.training.resume: self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        logger_to_use = cfg.logging.get('logger', None)
        accelerator = Accelerator(log_with=logger_to_use)
        if logger_to_use == 'wandb':
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            wandb_cfg.pop('project'); wandb_cfg.pop('logger', None)
            accelerator.init_trackers(
                project_name=cfg.logging.project, config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_cfg}
            )

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file(): print(f"Resuming from checkpoint {lastest_ckpt_path}"); self.load_checkpoint(path=lastest_ckpt_path)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer(); pickle.dump(normalizer, open(normalizer_path, 'wb'))
        accelerator.wait_for_everyone(); normalizer = pickle.load(open(normalizer_path, 'rb'))

        val_dataset = dataset.get_validation_dataset(); val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema: self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler, optimizer=self.optimizer, num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader)*cfg.training.num_epochs)//cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )
        ema: EMAModel = None
        if cfg.training.use_ema: ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        env_runner: BaseImageRunner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, 'checkpoints'), **cfg.checkpoint.topk)

        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        device = self.model.device
        if self.ema_model is not None: self.ema_model.to(device)

        train_sampling_batch = None
        if cfg.training.debug:
            cfg.training.num_epochs = 2; cfg.training.max_train_steps = 3; cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1; cfg.training.checkpoint_every = 1; cfg.training.val_every = 1; cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()
                step_log = dict()
                if cfg.training.freeze_encoder: self.model.obs_encoder.eval(); self.model.obs_encoder.requires_grad_(False)
                
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None: train_sampling_batch = batch
                        train_sampling_batch = batch

                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        accelerator.backward(loss)

                        if (self.global_step + 1) % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step(); self.optimizer.zero_grad(); lr_scheduler.step()
                        
                        if cfg.training.use_ema: ema.step(accelerator.unwrap_model(self.model))

                        raw_loss_cpu = raw_loss.item(); tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {'train_loss': raw_loss_cpu, 'global_step': self.global_step, 'epoch': self.epoch, 'lr': lr_scheduler.get_last_lr()[0]}
                        
                        # Utilization Logging Block
                        try:
                            unwrapped_model = accelerator.unwrap_model(self.model)
                            total_expert_usage = None; num_moe_layers = 0
                            for model in unwrapped_model.obs_encoder.key_model_map.values():
                                if 'vit' in getattr(model, 'model_name', ''):
                                    for block in model.blocks:
                                        if type(block.mlp).__name__ == 'MoEFeedForward':
                                            if total_expert_usage is None: total_expert_usage = block.mlp.expert_usage.detach().cpu()
                                            else: total_expert_usage += block.mlp.expert_usage.detach().cpu()
                                            num_moe_layers += 1
                            
                            if total_expert_usage is not None and num_moe_layers > 0:
                                total_tokens_routed = total_expert_usage.sum()
                                if total_tokens_routed > 0:
                                    expert_util_percent = (total_expert_usage / total_tokens_routed) * 100
                                    table_columns = ["Expert ID", "Utilization (%)"]; table_data = []
                                    for i in range(len(expert_util_percent)): table_data.append([f"Expert {i}", expert_util_percent[i].item()])
                                    util_table = wandb.Table(columns=table_columns, data=table_data)
                                    step_log["Expert Utilization/Distribution Bar Chart"] = wandb.plot.bar(
                                        util_table, "Expert ID", "Utilization (%)", title="Expert Utilization Distribution per Step")
                                    for i in range(len(expert_util_percent)): step_log[f'Expert Utilization/expert_{i}_%'] = expert_util_percent[i].item()
                        except Exception as e:
                            pass # Fail silently if logging fails

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            accelerator.log(step_log, step=self.global_step); json_logger.log(step_log)
                            self.global_step += 1
                        if (cfg.training.max_train_steps is not None) and self.global_step >= cfg.training.max_train_steps: break

                if (cfg.training.max_train_steps is not None) and self.global_step >= cfg.training.max_train_steps: break
                
                train_loss = np.mean(train_losses); step_log['train_loss'] = train_loss
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema: policy = self.ema_model
                policy.eval()

                if (self.epoch % cfg.training.rollout_every) == 0: step_log.update(env_runner.run(policy))
                
                def log_action_mse(log, cat, pred, gt):
                    mse = torch.nn.functional.mse_loss(pred, gt)
                    log[f'{cat}_action_mse_error'] = mse
                
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        pred_action = policy.predict_action(obs_dict)['action_pred']
                        log_action_mse(step_log, 'train', pred_action, gt_action)
                        del batch, obs_dict, gt_action, pred_action

                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    model_ddp = self.model; self.model = accelerator.unwrap_model(self.model)
                    if cfg.checkpoint.save_last_ckpt: self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot: self.save_snapshot()
                    metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None: self.save_checkpoint(path=topk_ckpt_path)
                    self.model = model_ddp
                
                accelerator.log(step_log, step=self.global_step); json_logger.log(step_log)
                self.global_step += 1; self.epoch += 1

        accelerator.end_training()

@hydra.main(version_base=None, config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerTimmWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()