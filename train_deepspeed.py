# filename: train_deepspeed.py (FINAL Robust Version)

import sys
import hydra
from omegaconf import DictConfig
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import deepspeed
import os  # <--- 新增导入

def main() -> None:
    """
    Manually initializes Hydra and is compatible with both direct python execution
    for single-GPU debugging and the DeepSpeed launcher for multi-GPU training.
    """
    # ===================================================================
    # ====================  这里是唯一的修改点  =====================
    # ===================================================================
    # Only call init_distributed if we are running in a multi-process environment,
    # which is indicated by the presence of the WORLD_SIZE env var.
    if 'WORLD_SIZE' in os.environ:
        deepspeed.init_distributed()
    # ===================================================================
    
    original_argv = sys.argv.copy()
    sys.argv = [arg for arg in original_argv if not arg.startswith('--local_rank')]

    with hydra.initialize(config_path="diffusion_policy/config", version_base=None):
        config_name = "train"
        for arg in original_argv:
            if arg.startswith("--config-name"):
                if "=" in arg:
                    config_name = arg.split("=")[1]
                else:
                    idx = original_argv.index(arg)
                    if idx + 1 < len(original_argv):
                        config_name = original_argv[idx+1]
                break

        overrides = []
        i = 1
        while i < len(original_argv):
            arg = original_argv[i]
            if arg.startswith('--config-name'):
                i += 1
                if "=" not in arg:
                    i += 1
                continue
            overrides.append(arg)
            i += 1
        
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

    if 'workspace' in cfg:
        cls = hydra.utils.get_class(cfg.workspace._target_)
        workspace: BaseWorkspace = cls(cfg)
    else:
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        
    workspace.run()


if __name__ == "__main__":
    main()