# modified from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/pixart_controlnet.py

import re
import torch.nn as nn

from copy import deepcopy
from torch.nn import Module, Linear, init
import torch.nn.functional as F

from typing import Any, Mapping, Tuple

from models.sit import SiT, SiTBlock

class ControlSiTBlock(Module):
    def __init__(self, base_block: SiTBlock, block_index: 0) -> None:
        super().__init__()
        self.copied_block = deepcopy(base_block)
        self.block_index = block_index

        for p in self.copied_block.parameters():
            p.requires_grad_(True)

        self.copied_block.load_state_dict(base_block.state_dict())
        self.copied_block.train()
        
        self.hidden_size = hidden_size = base_block.hidden_size
        if self.block_index == 0:
            self.before_proj = Linear(hidden_size, hidden_size)
            init.zeros_(self.before_proj.weight)
            init.zeros_(self.before_proj.bias)
        self.after_proj = Linear(hidden_size, hidden_size) 
        init.zeros_(self.after_proj.weight)
        init.zeros_(self.after_proj.bias)

    def forward(self, x, t, c): 
        # x is first input, t is time+label, c is the additional condition or previous output
        if self.block_index == 0:
            # the first block
            c = self.before_proj(c)
            c = self.copied_block(x + c, t)
            c_skip = self.after_proj(c)
        else:
            # load from previous output c and produce the c for skip connection
            c = self.copied_block(c, t)
            c_skip = self.after_proj(c)
        
        return c, c_skip

# reference: https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/controlnet.py
class ControlCondEmbedding(nn.Module):
    def __init__(self, in_channel: int = 1, out_channel: int = 4, block_out_channels: Tuple[int, ...] = (16, 32, 96, 256)):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channel, block_out_channels[0], kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
        
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channel, kernel_size=3, padding=1)
    
    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
    

class ControlSiT(Module):
    # only support single res model
    def __init__(self, base_model: SiT, copy_blocks_num: int = 13, 
                 no_cond_embedder=False, train_label_embedder=False, **cond_kwargs) -> None:
        super().__init__()
        self.base_model = base_model.eval()
        self.controlnet = []
        self.copy_blocks_num = copy_blocks_num
        self.total_blocks_num = len(base_model.blocks)
        if not no_cond_embedder:
            self.cond_embedder = ControlCondEmbedding(**cond_kwargs)
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        if train_label_embedder: 
            for p in self.base_model.y_embedder.parameters():
                p.requires_grad_(True)

        # Copy first copy_blocks_num block
        for i in range(copy_blocks_num):
            self.controlnet.append(ControlSiTBlock(base_model.blocks[i], i))
        self.controlnet = nn.ModuleList(self.controlnet)

    def __getattr__(self, name: str):
        if name in ['forward', 'load_state_dict']:
            return self.__dict__[name]
        elif name in ['base_model', 'controlnet', 'cond_embedder']:
            return super().__getattr__(name)
        else:
            # for directly accessing the attributes of the base_model, e.g. patch_size
            return getattr(self.base_model, name)

    # def forward(self, x, t, c, **kwargs):
    #     return self.base_model(x, t, c=self.forward_c(c), **kwargs)
    def forward(self, x, t, c=None, y=None):
        """
        Forward pass.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        t = t.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        if c is not None:
            c = c.to(self.dtype)
            c = self.cond_embedder(c)  # (N, C, H, W)
            c = self.x_embedder(c) + pos_embed
        else:
            c = x

        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        t_embed = self.t_embedder(t)                   # (N, D)
        if y is not None:
            # y = y.to(self.dtype)
            y = self.y_embedder(y, self.training)  # (N, D)
            t_embed = t_embed + y
        if c is not None:
            # update c and x
            for index in range(0, self.copy_blocks_num):
                c, c_skip = self.controlnet[index](x, t_embed, c) # original x doesn't affect the output for i > 0
                if index == 0:
                    x = self.base_model.blocks[index](x, t_embed) # first block without c_skip
                x = self.base_model.blocks[index + 1](x + c_skip, t_embed) # last one: copy_blocks_num
        
            # update x only
            for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
                x = self.base_model.blocks[index](x, t_embed)
        else:
            for index in range(0, self.total_blocks_num):
                x = self.base_model.blocks[index](x, t_embed)

        x = self.final_layer(x, t_embed)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x, t # t not used, for compatibility

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if all((k.startswith('base_model') or k.startswith('controlnet')) for k in state_dict.keys()):
            return super().load_state_dict(state_dict, strict)
        else:
            new_key = {}
            for k in state_dict.keys():
                new_key[k] = re.sub(r"(blocks\.\d+)(.*)", r"\1.base_block\2", k)
            for k, v in new_key.items():
                if k != v:
                    print(f"replace {k} to {v}")
                    state_dict[v] = state_dict.pop(k)

            return self.base_model.load_state_dict(state_dict, strict)

    @property
    def dtype(self):
        # return the dtype of the parameters
        return next(self.parameters()).dtype