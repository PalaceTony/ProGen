# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""DDPM model.

This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import torch
import torch.nn as nn
import functools
from . import utils, layers, normalization

get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

stlayer = layers.NewSpatioTemporalLayer
DownsampleST = layers.DownsampleST
UpsampleST = layers.UpsampleST
ResnetBlockDDPM = layers.ResnetST
# s4layer = layers.s4


@utils.register_model(name="ddpm")
class DDPM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act = act = get_act(config)
        self.register_buffer("sigmas", torch.tensor(utils.get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            (2 * config.data.image_size) // (2**i) for i in range(num_resolutions)
        ]

        AttnBlock = functools.partial(layers.AttnBlockST)
        self.conditional = conditional = config.model.conditional
        ResnetBlock = functools.partial(
            ResnetBlockDDPM, act=act, temb_dim=2 * nf, dropout=dropout
        )
        if conditional:
            self.temb_cond_combined_layer = nn.Conv2d(
                in_channels=3 * nf, out_channels=2 * nf, kernel_size=(3, 3), padding=1
            )
            modules = [self.temb_cond_combined_layer]
            nn.init.normal_(modules[0].weight, mean=0, std=0.01)
            nn.init.zeros_(modules[0].bias)
            self.temb_cond_1_layer = nn.Conv2d(
                in_channels=2 * nf, out_channels=2 * nf, kernel_size=(3, 3), padding=1
            )
            modules.append(self.temb_cond_1_layer)
            nn.init.normal_(modules[1].weight, mean=0, std=0.01)
            nn.init.zeros_(modules[1].bias)

        self.centered = config.data.centered
        channels = config.data.num_channels

        # Downsampling block
        modules.append(stlayer(c_in=channels, c_out=nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                modules.append(DownsampleST(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            if i_level != 0:
                modules.append(UpsampleST(channels=in_ch, with_conv=resamp_with_conv))

        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(stlayer(in_ch, channels))
        self.all_modules = nn.ModuleList(modules)

        self.scale_by_sigma = config.model.scale_by_sigma

    def forward(self, x, labels, adj, cond):
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            # timestep/scale embedding
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, self.nf)
            temb_expanded = temb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 170, 24)
            combined = torch.cat([temb_expanded, cond], dim=1)
            temb = combined
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if self.centered:
            # Input is in [-1, 1]
            h = x
        else:
            # Input is in [0, 1]
            h = 2 * x - 1.0

        # Downsampling block
        hs = [modules[m_idx](h, adj)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], adj, temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(modules[m_idx](hs[-1]))

                # Downsampling temb
                temb = modules[m_idx](
                    nn.Conv2d(temb.shape[1], hs[-1].shape[1], kernel_size=1).to(
                        temb.device
                    )(temb)
                )
                temb = nn.Conv2d(hs[-1].shape[1], 2 * self.nf, kernel_size=1).to(
                    temb.device
                )(temb)

                m_idx += 1

        h = hs[-1]
        h = modules[m_idx](h, adj, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, adj, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), adj, temb)
                m_idx += 1
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h)

                # Upsampling temb
                temb = modules[m_idx](
                    nn.Conv2d(temb.shape[1], h.shape[1], kernel_size=1).to(temb.device)(
                        temb
                    )
                )
                temb = nn.Conv2d(h.shape[1], 2 * self.nf, kernel_size=1).to(
                    temb.device
                )(temb)

                m_idx += 1

        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h, adj)
        m_idx += 1
        assert m_idx == len(modules)
        if self.scale_by_sigma:
            # Divide the output by sigmas. Useful for training with the NCSN loss.
            # The DDPM loss scales the network output by sigma in the loss function,
            # so no need of doing it here.
            used_sigmas = self.sigmas[labels, None, None, None]
            h = h / used_sigmas

        return h


if __name__ == "__main__":
    import torch
    import pickle

    def load_config(file_path):
        with open(file_path, "rb") as file:
            config = pickle.load(file)
        return config

    config_file_path = "src/model/DSTGCRN/score_sde_spatiotemporal/config.pkl"
    config = load_config(config_file_path)

    # Check if CUDA is available and set the default device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    adj = torch.randn(10, 2).to(device)

    # Create the DDPM model using the defined config
    model = DDPM(config).to(device)  # Move the model to the appropriate device

    # Create a dummy input tensor and move it to the GPU
    dummy_input = torch.randn(
        2, config.data.num_channels, 10, config.data.image_size
    ).to(device)
    # Create dummy labels and move them to the GPU
    dummy_labels = torch.randint(0, 100, (2,)).to(
        device
    )  # Assuming 100 different timesteps

    # Perform a forward pass
    output = model(dummy_input, dummy_labels, adj)

    # Print the output shape
    print("Output shape:", output.shape)  # Should match the expected output dimensions
