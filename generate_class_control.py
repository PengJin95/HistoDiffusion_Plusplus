import os
import math
import argparse

import torch
import torch.distributed as dist
from torchvision.transforms.v2.functional import resize, center_crop
import numpy as np
from diffusers.models import AutoencoderKL
from tqdm import tqdm, trange
from PIL import Image

from models.sit import SiT_models
from samplers import euler_sampler, euler_maruyama_sampler
from controlnet import ControlSiT

CRC_Classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
CRC_Count = [521, 522, 624, 567, 434, 662, 440, 537, 699]
BreakHis_Class = ['tubular_adenoma', 'papillary_carcinoma', 'phyllodes_tumor', 'mucinous_carcinoma', 
            'fibroadenoma', 'ductal_carcinoma', 'lobular_carcinoma', 'adenosis']
BreakHis_Count = [97, 89, 76, 138, 181, 614, 116, 79]
dataset_info = {
    'CRC': {
        'classes': CRC_Classes,
        'count': CRC_Count,
        'resolution': 224,
    },
    'BreakHis': {
        'classes': BreakHis_Class, 
        'count': BreakHis_Count,
        'resolution': (460, 700),
    },
}

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def main(args):
    """
    Run sampling.
    """
    if args.dist:
        dist.init_process_group(backend='nccl')
        RANK = dist.get_rank()
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        WORLD_SIZE = dist.get_world_size()
        
    else:
        RANK = 0
        LOCAL_RANK = 0
        WORLD_SIZE = 1
    torch.manual_seed(args.global_seed + RANK) 
    device = torch.device(f"cuda:{LOCAL_RANK}")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    torch.set_grad_enabled(False)

    setup_for_distributed(RANK == 0)

    num_classes = len(dataset_info[args.dataset]['classes'])
    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=num_classes,
        z_dims=[],
        class_dropout_prob=0.,
        skip_init=True,
        **block_kwargs,
    )
    base_ckpt = torch.load(args.base_ckpt, map_location='cpu')
    print('Loading base model')
    missing, unexpect = model.load_state_dict(base_ckpt['model'], strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpect}")

    control_ckpt = torch.load(args.control_ckpt, map_location='cpu')
    model.y_embedder.load_state_dict(control_ckpt['label_embedder'])
    model = ControlSiT(model, copy_blocks_num=args.copy_blocks_num, 
                       no_cond_embedder=True, train_label_embedder=True).to(device)
    print('Loading control model')
    missing, unexpect = model.controlnet.load_state_dict(control_ckpt['controlnet'], strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpect}")

    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained('autoencoder_diffuser.ckpt').to(device)

    sample_folder_dir = args.sample_dir
    if RANK == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
    class_labels = dataset_info[args.dataset]['classes']
    for c in class_labels:
        if RANK == 0:
            os.makedirs(f"{sample_folder_dir}/{c}", exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    for i in range(num_classes):          
        if args.dataset == 'BreakHis':
            max_samples = max(dataset_info['BreakHis']['count'])
            total_samples = max_samples if args.sync_mode == 'equal' \
                else (max_samples - dataset_info['BreakHis']['count'][i])
        else:
            total_samples = int(dataset_info[args.dataset]['count'][i] * args.sample_ratio)
        total_batches = math.ceil(total_samples / args.batch_size)
        start_batch = 0 if WORLD_SIZE == 1 else (total_batches // WORLD_SIZE) * RANK
        end_batch = (total_batches // WORLD_SIZE) * (RANK + 1) if RANK < WORLD_SIZE - 1 else total_batches
        iterator = trange(start_batch, end_batch) if RANK == 0 else range(start_batch, end_batch)
        for j in iterator:
            batch_size = min(args.batch_size, total_samples - j * args.batch_size)
            # Sample inputs:
            z = torch.randn(batch_size, model.in_channels, latent_size, latent_size, 
                            device=device, requires_grad=False) # (B, C, H, W)
            y = torch.ones(batch_size, device=device, requires_grad=False) * i # (B,)
            y = y.int()
            # Sample images:
            sampling_kwargs = dict(
                model=model, 
                latents=z,
                y=y,
                num_steps=args.num_steps, 
                heun=args.heun,
                cfg_scale=args.cfg_scale,
                guidance_low=args.guidance_low,
                guidance_high=args.guidance_high,
                path_type=args.path_type,
            )
            with torch.no_grad():
                if args.mode == "sde":
                    samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
                elif args.mode == "ode":
                    samples = euler_sampler(**sampling_kwargs).to(torch.float32)
                else:
                    raise NotImplementedError()

                latents_scale = torch.tensor(
                    [0.54207, 0.54207, 0.54207, 0.54207]
                    ).view(1, 4, 1, 1).to(device)
                latents_bias = -torch.tensor(
                    [0., 0., 0., 0.,]
                    ).view(1, 4, 1, 1).to(device)
                samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                samples = (samples + 1) / 2.
                samples = torch.clamp(255. * samples, 0, 255)
                
                if args.dataset == 'BreakHis':
                    samples = center_crop(resize(samples, 700), [460, 700])
                else:
                    samples = resize(samples, dataset_info[args.dataset]['resolution'])
                samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

                # Save samples to disk as individual .png files
                for k, sample in enumerate(samples):
                    index = j * args.batch_size + k
                    Image.fromarray(sample).save(f"{sample_folder_dir}/{class_labels[i]}/{index:04d}.png")
    print("Done.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--dist", action=argparse.BooleanOptionalAction, default=False,
                        help="Use DDP for distributed sampling.")
    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--base-ckpt", type=str, default=None, help="path to a SiT checkpoint.")
    parser.add_argument("--control-ckpt", type=str, default=None, help="path to a ControlNet checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=512)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--copy-blocks-num", type=int, default=13)

    # number of samples
    parser.add_argument("--dataset", type=str, default='CRC', choices=['CRC', 'BreakHis', 'PCam'])
    parser.add_argument("--sync-mode", type=str, default='balance', choices=['balance', 'equal'])
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=10)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=0)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    # will be deprecated
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False) # only for ode


    args = parser.parse_args()
    main(args)
