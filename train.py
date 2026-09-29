import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
import json

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import *

from diffusers.models import AutoencoderKL
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision.transforms import Normalize

from dataset import HistoTrain
from samplers import euler_sampler


logger = get_logger(__name__)
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        logger.info(f"process index: {accelerator.process_index}")

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8 # we use a VAE that downsamples by a factor of 8, so the latent size is 1/8 of the image size
    print(f"Latent size: {latent_size}")

    if args.enc_type != 'None':
        print(f"Loading encoders: {args.enc_type}")
        encoders, encoder_types, architectures = load_encoders(args.enc_type, device)
    else:
        print("No encoder used.")
        encoders, encoder_types, architectures = [], [], []
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else []
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = False,
        z_dims = z_dims,
        encoder_depth=args.encoder_depth, 
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained('autoencoder_diffuser.ckpt').to(device)
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.54207, 0.54207, 0.54207, 0.54207]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )
    if accelerator.is_main_process: 
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    train_dataset = HistoTrain(512, use_full_dataset=(not args.debug))
    print("num_processes:", accelerator.num_processes)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu', weights_only=False
            )
        clean_state_dict = {
            key.replace("module.", ""): value for key, value in ckpt['model'].items()
        }
        model.load_state_dict(clean_state_dict)
        ema.load_state_dict(ckpt['ema']) 
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config
        )
        tb_tracker = accelerator.get_tracker("tensorboard")
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        # disable=not accelerator.is_local_main_process,
        disable=not accelerator.is_main_process,
    )

    sample_batch_size = 1

    # uni use imagenet mean and std, conch use clip mean and std
    enc_mean = IMAGENET_DEFAULT_MEAN if args.enc_type == 'uni' else OPENAI_CLIP_MEAN
    enc_std = IMAGENET_DEFAULT_STD if args.enc_type == 'uni' else OPENAI_CLIP_STD

    for _ in range(0, args.epochs):
        model.train()
        for batch in train_dataloader:
            x = batch["image"] # (-1, 1)
            enc_input = Normalize(enc_mean, enc_std)(x.detach() / 2 + 0.5).to(device) 
            z = None
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.54207)
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type , _ in zip(encoders, encoder_types, architectures):
                        if encoder_type == 'uni':
                            z = encoder.forward_features(enc_input)[:, 1:]
                        elif encoder_type == 'conch':
                            z = encoder(enc_input)[:, 1:]
                        zs.append(z)

            with accelerator.accumulate(model):
                if len(zs) == 0:
                    loss = loss_fn(model, x, zs=zs)
                else:
                    loss, proj_loss = loss_fn(model, x, zs=zs)
                loss_mean = loss.mean()
                if args.proj_coeff > 0 and torch.is_tensor(proj_loss):
                    proj_loss_mean = proj_loss.mean()
                    loss = loss_mean + proj_loss_mean * args.proj_coeff
                else:
                    loss = loss_mean
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                else:
                    grad_norm = None
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
                if (global_step == 1 or (global_step % args.checkpointing_steps == 0 and global_step > 0)):
                    if accelerator.is_main_process:
                        checkpoint = {
                            # "model": model.module.state_dict(),
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": optimizer.state_dict(),
                            "args": args,
                            "steps": global_step,
                        }
                        checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        del checkpoint
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                    with torch.no_grad():
                        xT = torch.randn((sample_batch_size, 4, latent_size, latent_size), device=device)
                        samples = euler_sampler(
                            model, 
                            xT, 
                            None,
                            num_steps=50, 
                            cfg_scale=0.0,
                            guidance_low=0.,
                            guidance_high=1.,
                            path_type=args.path_type,
                            heun=False,
                        ).to(torch.float32)
                        gt_xs = x[:sample_batch_size]
                        samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                        gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                        samples = (samples + 1) / 2.
                        gt_samples = (gt_samples + 1) / 2.
                    out_samples = accelerator.gather(samples.to(torch.float32))
                    gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                    if accelerator.is_main_process:
                        tb_tracker.log_images({
                            "samples": array2grid(out_samples),
                            "gt_samples": array2grid(gt_samples)
                        }, step=global_step, dataformats='HWC')
                    del out_samples, gt_samples
                    logging.info("Generating EMA samples done.")

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
            }
            if grad_norm is not None:
                logs["grad_norm"] = accelerator.gather(grad_norm).mean().detach().item()
            if args.proj_coeff > 0 and torch.is_tensor(proj_loss_mean):
                logs['proj_loss'] = accelerator.gather(proj_loss_mean).mean().detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            torch.cuda.empty_cache()
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()      
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=512)
    parser.add_argument("--batch-size", type=int, default=4)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=180000)
    parser.add_argument("--checkpointing-steps", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=3)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='uni-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")

    parser.add_argument("--debug", action="store_true")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

"""
Example usage:
  python train.py --output-dir exp_repa/run1 \
    --allow-tf32 --exp-name=run1 --batch-size 64 \
    --max-train-steps 180000 --proj-coeff 0.5 \
    --gradient-accumulation-steps 4
"""