import os
import torch

from omegaconf import OmegaConf
import transformers
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    if transformers.__version__ != "4.19.2" and "cond_stage_model.transformer.vision_model.embeddings.position_ids" in state_dict.keys():
        del state_dict["cond_stage_model.transformer.vision_model.embeddings.position_ids"]    
        print(f"delete cond_stage_model.transformer.vision_model.embeddings.position_ids from loaded state dict (transformers version : {transformers.__version__})")
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def create_model(config_path=None, config=None, **kwargs):
    # Load config if not passed directly
    if config is None:
        if config_path is None:
            raise ValueError("❌ Either config_path or config must be provided.")
        config = OmegaConf.load(config_path)

    # Basic config sanity check
    if not OmegaConf.select(config, "model") or not OmegaConf.select(config, "model.params"):
        raise ValueError("❌ Config missing required 'model' or 'model.params' section.")

    print(f"🛠 Creating model: {config.model.get('target', 'UnknownTarget')}")
    print(f"📐 Model params in config: {list(config.model.params.keys())}")

    # Instantiate model
    model = instantiate_from_config(config.model).cpu()
    print(f"Loaded model config from [{config_path}]")

    # Verify parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_nonzero = sum((p != 0).sum().item() for p in model.parameters())
    if num_nonzero == 0:
        raise RuntimeError("❌ Model parameters are all zero — weights may not be loaded!")

    print(f"✅ Model OK: {num_params:,} params, {num_nonzero:,} nonzero")
    print(f"📦 Model class: {model.__class__.__name__}")
    print(f"✅ Model set to eval mode: {not model.training}")

    return model

