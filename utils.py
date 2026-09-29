import os
from collections import OrderedDict

import torch
from torchvision.utils import make_grid

@torch.no_grad()
def load_encoders(enc_type, device):
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        architectures.append(architecture)
        encoder_types.append(encoder_type)
        if encoder_type == 'uni': # uni-vit-l
            print('Loading UNI encoder...')
            import timm
            encoder = timm.create_model("vit_large_patch16_224", img_size=512, patch_size=16, init_values=1e-5, num_classes=0)
            ckpt = torch.load(os.path.join("pretrained_enc/UNI", "pytorch_model.bin"), map_location="cpu")
            ckpt['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(encoder.pos_embed.data, [32, 32])
            encoder.load_state_dict(ckpt, strict=True)
            encoder = encoder.to(device)
            encoder.eval()

        elif encoder_type == 'conch':
            print('Loading CONCH encoder...')
            import json
            from conch.open_clip_custom import create_model_from_pretrained
            with open(os.path.join("pretrained_enc/CONCH", "config.json"), "r") as f:
                config = json.load(f)
            ckpt_path = os.path.join("pretrained_enc/CONCH", "pytorch_model.bin")
            encoder = create_model_from_pretrained(config, ckpt_path, return_transform=False)
            encoder = encoder.visual.trunk.to(device) # only use image encoder 
            encoder.eval()          
        encoders.append(encoder)
    
    return encoders, encoder_types, architectures


def array2grid(x): 
    x = make_grid(x.clamp(0, 1), nrow=4, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):  
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag