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

stlayer = layers.EnhancedSpatioTemporalLayer
DownsampleST = layers.DownsampleST
UpsampleST = layers.UpsampleST
ResnetBlockDDPM = layers.ResnetST
# s4layer = layers.s4


def greatest_divisor(n, max_divisor):
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


@utils.register_model(name="ddpm")
class DDPM(nn.Module):
    def __init__(self, config, adj):
        super().__init__()
        self.act = act = get_act(config)
        self.config = config

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)

        self.conditional = conditional = config.model.conditional
        ResnetBlock = functools.partial(
            ResnetBlockDDPM,
            adj=adj,
            d_k=config.stlayer.d_k,
            d_v=config.stlayer.d_v,
            hidden_size=config.stlayer.hidden_size,
            num_vertices=config.V,
            n_heads=config.stlayer.n_heads,
            K=config.stlayer.K,
            act=act,
            temb_dim=2 * nf,
            dropout=dropout,
            temporal_layer=config.model.temporal_layer,
            s4_state_dimm=config.model.s4_state_dim,
            s4_num_temporal_layers=config.model.s4_num_temporal_layers,
        )
        if conditional:
            modules = [nn.Linear(nf, nf * 2)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(nf * 2, nf * 2))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)

        self.centered = config.data.centered
        channels = config.data.num_channels

        # Downsampling block
        modules.append(
            stlayer(
                in_features=channels,
                out_features=nf,
                adj=adj,
            )
        )
        hs_c = [nf + self.config.model.pos_emb]
        in_ch = nf + self.config.model.pos_emb
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                modules.append(DownsampleST(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if i_level != 0:
                modules.append(UpsampleST(channels=in_ch, with_conv=resamp_with_conv))

        assert not hs_c
        modules.append(
            nn.GroupNorm(
                num_channels=in_ch, num_groups=greatest_divisor(in_ch, 4), eps=1e-6
            )
        )
        modules.append(
            stlayer(
                in_features=in_ch,
                out_features=channels,
                adj=adj,
            )
        )
        modules.append(nn.Linear(config.model.shape * 2, config.model.shape))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, labels, adj, cond, pos_w, pos_d):
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            # timestep/scale embedding
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, self.nf)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        points_per_day = 12 * 24  # 288 data points per day (5-minute intervals)
        points_per_week = points_per_day * 7  # 2016 data points per week
        day_enc = layers.positional_encoding(
            points_per_day, self.config.model.pos_emb
        ).to(x.device)
        week_enc = layers.positional_encoding(
            points_per_week, self.config.model.pos_emb
        ).to(x.device)
        day_pos_emb = day_enc[pos_d]  # (8, 12, 64)
        day_pos_emb = (
            day_pos_emb.unsqueeze(2)
            .expand(-1, -1, self.config.V, -1)
            .permute(0, 3, 2, 1)
        )  # (8, 64, 170, 12)
        week_pos_emb = week_enc[pos_w]  # (8, 12, 64)
        week_pos_emb = (
            week_pos_emb.unsqueeze(2)
            .expand(-1, -1, self.config.V, -1)
            .permute(0, 3, 2, 1)
        )  # (8, 64, 170, 12)
        pos_w_d = torch.cat((day_pos_emb, week_pos_emb), dim=3)  # (8, 64, 170, 24)

        h = torch.cat((x, cond), dim=3)  # (8, 1, 170, 24)

        # Downsampling block
        hs = [modules[m_idx](h, adj)]
        m_idx += 1
        hs[-1] = torch.cat((hs[-1], pos_w_d), dim=1)  # (8, nf+32, 170, 24)

        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], adj, temb)
                m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(modules[m_idx](hs[-1]))
                m_idx += 1

        h = hs[-1]
        h = modules[m_idx](h, adj, temb)
        m_idx += 1
        h = modules[m_idx](h, adj, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), adj, temb)
                m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1

        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h, adj)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)

        return h
