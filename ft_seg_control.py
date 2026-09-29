import os
import argparse
import copy
import logging
from pathlib import Path
import json
from itertools import chain

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from controlnet import ControlSiT
from models.sit import SiT_models
from loss import SILoss

from diffusers.models import AutoencoderKL
from dataset import MoNuSACDataset, CoNSePDataset
from utils import array2grid

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
    latent_size = args.resolution // 8

    z_dims = [] # do not use alignment
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        z_dims = z_dims,
        **block_kwargs
    )
    base_ckpt = torch.load(args.base_ckpt, map_location='cpu')
    missing, unexpect =model.load_state_dict(base_ckpt['model'], strict=False)
    if accelerator.is_main_process:
        logger.info(f"Missing keys: {missing}")
        logger.info(f"Unexpected keys: {unexpect}")

    cond_kwargs = { "in_channel": args.cond_in_channel }
    model = ControlSiT(model, copy_blocks_num=args.copy_blocks_num, **cond_kwargs).to(device)
    vae = AutoencoderKL.from_pretrained('autoencoder_diffuser.ckpt').to(device)

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
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )

    if accelerator.is_main_process: 
        logger.info(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

            
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        chain(model.controlnet.parameters(), model.cond_embedder.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Setup data:
    if args.dataset == "monusac":
        train_dataset = MoNuSACDataset(mode=args.dataset_suffix)
    elif args.dataset == "consep":
        train_dataset = CoNSePDataset(mode=args.dataset_suffix)
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
    model.train()
    
    # resume:
    global_step = 0

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA_Control", 
            config=tracker_config,
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

    sample_batch_size = 2
       
    for _ in range(0, args.epochs):
        model.train()
        for batch in train_dataloader:
            x = batch["image"].to(device) # (-1, 1)
            mask = batch["mask"].to(device) # (-1, 1)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.54207)
            with accelerator.accumulate(model):
                model_kwargs = dict(c=mask)
                loss = loss_fn(model, x, model_kwargs).mean()
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1         

            # set to 1 for recording the initial status or debugging
            if (global_step == 1 or (global_step % args.checkpointing_steps == 0 and global_step > 0)):
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    checkpoint = {
                        "controlnet": unwrapped_model.controlnet.state_dict(),
                        "cond_embedder": unwrapped_model.cond_embedder.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler
                with torch.no_grad():
                    xT = torch.randn((sample_batch_size, 4, latent_size, latent_size), device=device)
                    model_kwargs = dict(c=mask[:sample_batch_size])
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
                        **model_kwargs
                    ).to(torch.float32)
                    gt_xs = x[:sample_batch_size]
                    gt_masks = mask[:sample_batch_size]
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                    # gt_masks = (gt_masks + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                gt_masks = accelerator.gather(gt_masks.to(torch.float32))
                if accelerator.is_main_process:
                    tb_tracker.log_images({
                        "samples": array2grid(out_samples),
                        "gt_samples": array2grid(gt_samples),
                        "gt_masks_inst": array2grid(gt_masks[:, [0], :, :]),
                        "gt_masks_seg": array2grid(gt_masks[:, [1], :, :])
                    }, step=global_step, dataformats='HWC')

            logs = {
                "loss": accelerator.gather(loss).mean().detach().item(), 
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps_seg")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--sampling-steps", type=int, default=200)

    # model
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--base-ckpt", type=str, required=True)
    parser.add_argument("--copy-blocks-num", type=int, default=13)
    parser.add_argument("--cond-in-channel", type=int, default=1)

    # dataset
    parser.add_argument("--dataset", type=str, default="lizard")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-suffix", type=str, default='train')

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=20000)
    parser.add_argument("--checkpointing-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
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
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
