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
"""Common layers for defining score networks.
"""
import math
import string
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .normalization import ConditionalInstanceNorm2dPlus


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

# from score_sde_spatiotemporal.s4 import S4Model
from torch_geometric.nn import GINEConv


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, (1, kernel_size), padding=(0, kernel_size // 2)
        )

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x)


class ChebConv(nn.Module):
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        super(ChebConv, self).__init__()
        self.K = K
        self.cheb_polynomials = [
            torch.tensor(cp, dtype=torch.float32) for cp in cheb_polynomials
        ]
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.randn(in_channels, out_channels)) for _ in range(K)]
        )

    def forward(self, x, adj):
        device = x.device
        cheb_polynomials = [cp.to(device) for cp in self.cheb_polynomials]

        batch_size, num_of_timesteps, num_of_vertices, in_channels = x.shape
        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[
                :, time_step, :, :
            ]  # (batch_size, num_of_vertices, in_channels)
            output = torch.zeros(
                batch_size, num_of_vertices, self.Theta[0].size(1), device=device
            )

            for k in range(self.K):
                T_k = cheb_polynomials[k].to(
                    device
                )  # (num_of_vertices, num_of_vertices)
                theta_k = self.Theta[k].to(device)  # (in_channels, out_channels)
                rhs = torch.matmul(
                    T_k, graph_signal
                )  # (batch_size, num_of_vertices, in_channels)
                output += torch.matmul(
                    rhs, theta_k
                )  # (batch_size, num_of_vertices, out_channels)

            outputs.append(
                output.unsqueeze(1)
            )  # (batch_size, 1, num_of_vertices, out_channels)

        return F.relu(
            torch.cat(outputs, dim=1)
        )  # (batch_size, num_of_timesteps, num_of_vertices, out_channels)


class EnhancedSpatioTemporalLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        adj,
        hidden_size=64,
        K=3,
        num_of_vertices=170,
        temporal_kernel_size=3,
    ):
        super(EnhancedSpatioTemporalLayer, self).__init__()
        self.temporal_conv = TemporalConvLayer(
            in_features, hidden_size, temporal_kernel_size
        )
        self.K = K
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.num_of_vertices = num_of_vertices
        self.residual = nn.Linear(in_features, out_features)

        L_tilde = self.scaled_laplacian(adj)
        self.cheb_polynomials = [
            i.float() for i in self.cheb_polynomial(L_tilde, self.K)
        ]

        self.spatial_cheb_conv = ChebConv(
            self.K, self.cheb_polynomials, hidden_size, out_features
        )

    def scaled_laplacian(self, adj):
        adj = torch.tensor(adj, dtype=torch.float32)
        D = torch.diag(torch.sum(adj, dim=1))
        L = D - adj
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        L_norm = torch.mm(torch.mm(D_inv_sqrt, L), D_inv_sqrt)
        eigvals = torch.linalg.eigvals(L_norm)
        lambda_max = eigvals.real.max()
        L_scaled = (2 / lambda_max) * L_norm - torch.eye(L.size(0))
        return L_scaled

    def cheb_polynomial(self, L, K):
        N = L.size(0)
        cheb_polynomials = [torch.eye(N), L]
        for i in range(2, K):
            cheb_polynomials.append(
                2 * torch.mm(L, cheb_polynomials[-1]) - cheb_polynomials[-2]
            )
        return cheb_polynomials

    def forward(self, x, adj):
        batch_size, num_features, num_nodes, time_steps = x.shape

        # Reshape to [batch_size, num_features, num_nodes, time_steps] for temporal convolution
        x = x.permute(0, 1, 2, 3).contiguous()

        # Temporal Convolution
        temp_output = self.temporal_conv(x)

        # Reshape to [batch_size, num_of_timesteps, num_nodes, hidden_size] for spatial convolution
        temp_output = temp_output.permute(0, 3, 2, 1).contiguous()

        # Spatial Chebyshev Convolution
        gcn_output = self.spatial_cheb_conv(temp_output, adj)

        # Residual and Layer Normalization
        residual = self.residual(
            x.view(batch_size, num_features, num_nodes, time_steps)
            .permute(0, 3, 2, 1)
            .contiguous()
        )
        output = gcn_output + residual
        output = output.permute(0, 3, 2, 1).contiguous()

        return output


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k=2, num_node=10, embed_dim=2):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.randn(embed_dim, cheb_k, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(torch.randn(embed_dim, dim_out))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weights_pool, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights_pool)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, adj):
        B, C, N, T = x.shape
        node_embeddings = adj
        node_num = node_embeddings.shape[0]

        # Compute normalized adjacency matrix supports
        supports = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1
        )
        support_set = [torch.eye(node_num).to(supports.device), supports]

        # Generate higher-order Chebyshev polynomials
        for k in range(2, self.cheb_k):
            support_set.append(
                torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            )
        supports = torch.stack(support_set, dim=0)  # Shape: [cheb_k, N, N]

        output = []
        for t in range(T):
            x_t = x[:, :, :, t]  # Select data at time t, shape [B, C, N]
            x_t = x_t.permute(0, 2, 1).contiguous()  # Shape: [B, N, C]

            # Compute weights and bias
            weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
            bias = torch.matmul(node_embeddings, self.bias_pool)

            # Graph convolution
            x_g = torch.einsum("knm,bmc->bknc", supports, x_t)
            x_g = x_g.permute(0, 2, 1, 3).contiguous()  # Shape: [B, N, cheb_k, C]
            x_gconv = torch.einsum("bnki,nkio->bno", x_g, weights) + bias

            # Add skip connection
            x_gconv = x_gconv + x_t  # Shape: [B, N, dim_out]
            output.append(x_gconv)

        output = torch.stack(output, dim=3)  # Shape: [B, N, dim_out, T]
        return output


class s4(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        state_dim=64,
        channels=1,
        num_temporal_layers=2,
        dropout=0.1,
        prenorm=False,
        max_seq_len=12,
        bidirectional=False,
        postact=None,
    ):
        super().__init__()
        self.t_model = S4Model(
            d_input=input_dim,
            d_model=hidden_dim,
            d_state=state_dim,
            channels=channels,
            n_layers=num_temporal_layers,
            dropout=dropout,
            prenorm=prenorm,
            l_max=max_seq_len,
            bidirectional=bidirectional,
            postact=postact,  # none or 'glu'
            add_decoder=False,
            pool=False,  # hard-coded
            temporal_pool=None,
        )
        self.shortcut = (
            nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
            if input_dim != hidden_dim
            else None
        )

    def forward(self, x):
        B, C, N, T = x.shape
        x_tmp = x.permute(0, 2, 3, 1).reshape(B * N, T, C).contiguous()
        x_tmp = self.t_model(x_tmp)
        x_tmp = x_tmp.reshape(B, N, T, -1).permute(0, 3, 1, 2).contiguous()
        x_skip = x if self.shortcut is None else self.shortcut(x)
        return x_tmp + x_skip


class NewSpatioTemporalLayer(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        s4_state_dim,
        s4_num_temporal_layers,
        layer_type="gru",
    ):
        super(NewSpatioTemporalLayer, self).__init__()

        if layer_type == "s4":
            self.temporal_block = s4(
                input_dim=c_in,
                hidden_dim=c_out,
                state_dim=s4_state_dim,
                num_temporal_layers=s4_num_temporal_layers,
            )
        elif layer_type == "gru":
            self.temporal_block = nn.GRU(
                input_size=c_in,
                hidden_size=c_out,
                batch_first=True,
            )
        self.layer_type = layer_type

        self.spatial_block = AVWGCN(
            dim_in=c_out,  # Output from temporal block becomes input here
            dim_out=c_out,  # Maintain same dimension for output
        )

    def forward(self, x, adj):
        if self.layer_type == "s4":
            # x: [b, d, n, t] -> Expected input
            # First pass through the temporal block
            temporal_output = self.temporal_block(x)
            # Now pass the output through the spatial block
            st_output = self.spatial_block(temporal_output, adj)
            st_output = st_output.permute(0, 2, 1, 3).contiguous()
        elif self.layer_type == "gru":
            # x: [b, c_in, n, t] -> Expected input
            # Reshape and permute x to match GRU input requirements: [b, n, t, c_in]
            b, c_in, n, t = x.shape
            x = (
                x.permute(0, 2, 3, 1).reshape(b * n, t, c_in).contiguous()
            )  # [b*n, t, c_in]

            # Process each node sequence through the GRU
            temporal_output, _ = self.temporal_block(x)  # [b*n, t, c_out]

            # Reshape back to original dimensions
            temporal_output = (
                temporal_output.reshape(b, n, t, -1).permute(0, 3, 1, 2).contiguous()
            )  # [b, c_out, n, t]

            # Now pass the output through the spatial block
            st_output = self.spatial_block(temporal_output, adj)
            st_output = st_output.permute(
                0, 2, 1, 3
            ).contiguous()  # Adjust dimensions if needed for subsequent layers

        return st_output


# class NewSpatioTemporalLayer(nn.Module):
#     def __init__(self, c_in, c_out):
#         super(NewSpatioTemporalLayer, self).__init__()

#         self.temporal_block = nn.GRU(
#             input_size=c_in,
#             hidden_size=c_out,
#             batch_first=True,  # This will allow input/output of the shape [batch, seq, feature]
#         )

#         self.spatial_block = AVWGCN(
#             dim_in=c_out,  # Output from GRU becomes input here
#             dim_out=c_out,  # Maintain the same dimension for output
#         )

#     def forward(self, x, adj):
#         # x: [b, c_in, n, t] -> Expected input
#         # Reshape and permute x to match GRU input requirements: [b, n, t, c_in]
#         b, c_in, n, t = x.shape
#         x = x.permute(0, 2, 3, 1).reshape(b * n, t, c_in).contiguous()  # [b*n, t, c_in]

#         # Process each node sequence through the GRU
#         temporal_output, _ = self.temporal_block(x)  # [b*n, t, c_out]

#         # Reshape back to original dimensions
#         temporal_output = (
#             temporal_output.reshape(b, n, t, -1).permute(0, 3, 1, 2).contiguous()
#         )  # [b, c_out, n, t]

#         # Now pass the output through the spatial block
#         st_output = self.spatial_block(temporal_output, adj)
#         st_output = st_output.permute(
#             0, 2, 1, 3
#         ).contiguous()  # Adjust dimensions if needed for subsequent layers

#         return st_output


class SpatialBlock(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatialBlock, self).__init__()
        self.theta = nn.Parameter(torch.randn(c_in, c_out, ks))
        self.b = nn.Parameter(torch.randn(1, c_out, 1, 1))
        self.conv1x1 = nn.Conv2d(
            c_in, c_out, kernel_size=1
        )  # 1x1 convolution to expand feature dimension
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        # x: [b, c_in, time, n_nodes]
        # Lk: [3, n_nodes, n_nodes]

        # Random generate (2, 170, 170) as Lk

        if len(Lk.shape) == 2:  # if supports_len == 1:
            Lk = Lk.unsqueeze(0)
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = (
            torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        )  # [b, c_out, time, n_nodes]
        if x.shape[1] != x_gc.shape[1]:
            x = self.conv1x1(x)

        return torch.relu(x_gc + x)


class TcnBlock(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size,
        dilation_size=1,
        droupout=0.0,
        stride=1,
        padding_override=None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.padding = 1 if padding_override is None else padding_override

        self.conv = nn.Conv2d(
            c_in,
            c_out,
            stride=stride,
            kernel_size=kernel_size,
            padding=(1, self.padding) if padding_override is None else padding_override,
            dilation=dilation_size,
        )

        self.drop = nn.Dropout(droupout)

        self.net = nn.Sequential(self.conv, self.drop)

        if padding_override is None:
            self.shortcut = (
                nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else None
            )
        else:
            self.shortcut = nn.Conv2d(
                c_in, c_out, kernel_size=(1, 2), stride=(1, 2), padding=0
            )

    def forward(self, x):
        # x: (B, C_in, V, T) -> (B, C_out, V, T)
        out = self.net(x)
        x_skip = x if self.shortcut is None else self.shortcut(x)

        return out + x_skip


class SpatioTemporalLayer(nn.Module):
    def __init__(
        self, c_in, c_out, dropout=0.0, stride=1, padding_override=None, kernel_size=3
    ):
        super(SpatioTemporalLayer, self).__init__()

        # Initialize the Spatial Block
        self.spatial_block = SpatialBlock(ks=2, c_in=c_in, c_out=c_out)

        # Initialize the Temporal Block
        self.tcn_block = TcnBlock(
            c_in=c_out,
            c_out=c_out,
            kernel_size=kernel_size,
            dilation_size=1,
            droupout=dropout,
            stride=stride,
            padding_override=padding_override,
        )
        self.Lk = torch.rand(2, 170, 170)

    def forward(self, x):
        # x: [b, c_in, time, n_nodes]
        # Lk: Laplacian matrix [3, n_nodes, n_nodes] or [n_nodes, n_nodes]

        # First pass through the spatial block
        Lk = self.Lk
        x = x.permute(0, 1, 3, 2).contiguous()
        spatial_output = self.spatial_block(x, Lk)
        spatial_output = spatial_output.permute(0, 1, 3, 2).contiguous()

        # Now pass the output through the temporal block
        # Note: Ensure dimensions match, modify if necessary
        spatio_temporal_output = self.tcn_block(spatial_output)

        return spatio_temporal_output


def get_act(config):
    """Get activation functions from the config file."""

    if config.model.nonlinearity.lower() == "elu":
        return nn.ELU()
    elif config.model.nonlinearity.lower() == "relu":
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == "lrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == "swish":
        return nn.SiLU()
    else:
        raise NotImplementedError("activation function does not exist!")


def ncsn_conv1x1(
    in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=0
):
    """1x1 convolution. Same as NCSNv1/v2."""
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=bias,
        dilation=dilation,
        padding=padding,
    )
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


def variance_scaling(
    scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device="cpu"
):
    """Ported from JAX."""

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode)
            )
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (
                torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0
            ) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, "fan_avg", "uniform")


class Dense(nn.Module):
    """Linear layer with `default_init`."""

    def __init__(self):
        super().__init__()


def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.0, padding=0):
    """1x1 convolution with DDPM initialization."""
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def ncsn_conv3x3(
    in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1
):
    """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        stride=stride,
        bias=bias,
        dilation=dilation,
        padding=padding,
        kernel_size=3,
    )
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


def ddpm_conv3x3(
    in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1
):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv

    ###########################################################################
    # Functions below are ported over from the NCSNv1/NCSNv2 codebase:
    # https://github.com/ermongroup/ncsn
    # https://github.com/ermongroup/ncsnv2
    ###########################################################################


class CRPBlock(nn.Module):
    def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        if maxpool:
            self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        else:
            self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        self.act = act

    def forward(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.pool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class CondCRPBlock(nn.Module):
    def __init__(self, features, n_stages, num_classes, normalizer, act=nn.ReLU()):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.normalizer = normalizer
        for i in range(n_stages):
            self.norms.append(normalizer(features, num_classes, bias=True))
            self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))

        self.n_stages = n_stages
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x, y):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
            path = self.pool(path)
            path = self.convs[i](path)

            x = path + x
        return x


class RCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU()):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(
                    self,
                    "{}_{}_conv".format(i + 1, j + 1),
                    ncsn_conv3x3(features, features, stride=1, bias=False),
                )

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, "{}_{}_conv".format(i + 1, j + 1))(x)

            x += residual
        return x


class CondRCUBlock(nn.Module):
    def __init__(
        self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.ReLU()
    ):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(
                    self,
                    "{}_{}_norm".format(i + 1, j + 1),
                    normalizer(features, num_classes, bias=True),
                )
                setattr(
                    self,
                    "{}_{}_conv".format(i + 1, j + 1),
                    ncsn_conv3x3(features, features, stride=1, bias=False),
                )

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        self.normalizer = normalizer

    def forward(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, "{}_{}_norm".format(i + 1, j + 1))(x, y)
                x = self.act(x)
                x = getattr(self, "{}_{}_conv".format(i + 1, j + 1))(x)

            x += residual
        return x


class MSFBlock(nn.Module):
    def __init__(self, in_planes, features):
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = nn.ModuleList()
        self.features = features

        for i in range(len(in_planes)):
            self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))

    def forward(self, xs, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = F.interpolate(h, size=shape, mode="bilinear", align_corners=True)
            sums += h
        return sums


class CondMSFBlock(nn.Module):
    def __init__(self, in_planes, features, num_classes, normalizer):
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.features = features
        self.normalizer = normalizer

        for i in range(len(in_planes)):
            self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))
            self.norms.append(normalizer(in_planes[i], num_classes, bias=True))

    def forward(self, xs, y, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = F.interpolate(h, size=shape, mode="bilinear", align_corners=True)
            sums += h
        return sums


class RefineBlock(nn.Module):
    def __init__(
        self, in_planes, features, act=nn.ReLU(), start=False, end=False, maxpool=True
    ):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(RCUBlock(in_planes[i], 2, 2, act))

        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act)

        if not start:
            self.msf = MSFBlock(in_planes, features)

        self.crp = CRPBlock(features, 2, act, maxpool=maxpool)

    def forward(self, xs, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)

        return h


class CondRefineBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        features,
        num_classes,
        normalizer,
        act=nn.ReLU(),
        start=False,
        end=False,
    ):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act)
            )

        self.output_convs = CondRCUBlock(
            features, 3 if end else 1, 2, num_classes, normalizer, act
        )

        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer)

        self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act)

    def forward(self, xs, y, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]

        h = self.crp(h, y)
        h = self.output_convs(h, y)

        return h


class ConvMeanPool(nn.Module):
    def __init__(
        self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False
    ):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=biases,
            )
            self.conv = conv
        else:
            conv = nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=biases,
            )

            self.conv = nn.Sequential(nn.ZeroPad2d((1, 0, 1, 0)), conv)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = (
            sum(
                [
                    output[:, :, ::2, ::2],
                    output[:, :, 1::2, ::2],
                    output[:, :, ::2, 1::2],
                    output[:, :, 1::2, 1::2],
                ]
            )
            / 4.0
        )
        return output


class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=biases,
        )

    def forward(self, inputs):
        output = inputs
        output = (
            sum(
                [
                    output[:, :, ::2, ::2],
                    output[:, :, 1::2, ::2],
                    output[:, :, ::2, 1::2],
                    output[:, :, 1::2, 1::2],
                ]
            )
            / 4.0
        )
        return self.conv(output)


class UpsampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=biases,
        )
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inputs):
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        output = self.pixelshuffle(output)
        return self.conv(output)


class ConditionalResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_classes,
        resample=1,
        act=nn.ELU(),
        normalization=ConditionalInstanceNorm2dPlus,
        adjust_padding=False,
        dilation=None,
    ):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == "down":
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ConvMeanPool(
                    input_dim, output_dim, 3, adjust_padding=adjust_padding
                )
                conv_shortcut = partial(
                    ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding
                )

        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = nn.Conv2d
                self.conv1 = ncsn_conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim)
        else:
            raise Exception("invalid resample value")

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim, num_classes)

    def forward(self, x, y):
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        resample=None,
        act=nn.ELU(),
        normalization=nn.InstanceNorm2d,
        adjust_padding=False,
        dilation=1,
    ):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == "down":
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(
                    input_dim, output_dim, 3, adjust_padding=adjust_padding
                )
                conv_shortcut = partial(
                    ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding
                )

        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                # conv_shortcut = nn.Conv2d ### Something wierd here.
                conv_shortcut = partial(ncsn_conv1x1)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim)
        else:
            raise Exception("invalid resample value")

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)

    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


###########################################################################
# Functions below are ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
###########################################################################


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _einsum(a, b, c, x, y):
    einsum_str = "{},{}->{}".format("".join(a), "".join(b), "".join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape) : len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(
            default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True
        )
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2).contiguous()


class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""

    def __init__(self, channels):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=0.0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        return x + h


class AttnBlockST(nn.Module):
    """Spatiotemporal self-attention block for BDNT shaped data."""

    def __init__(self, channels):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=0.0)

    def forward(self, x):
        B, D, N, T = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum("bdnt,bdij->bntij", q, k) * (int(D) ** (-0.5))
        w = torch.reshape(w, (B, N, T, N * T))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, N, T, N, T))
        h = torch.einsum("bntij,bdij->bdnt", w, v)
        h = self.NIN_3(h)
        return x + h


# class NIN(nn.Module):
#     def __init__(self, in_dim, num_units, init_scale=0.1):
#         super().__init__()
#         self.W = nn.Parameter(
#             torch.randn(in_dim, num_units) * init_scale, requires_grad=True
#         )
#         self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)  # Reorder dimensions to bring feature dimension last
#         y = torch.einsum("bdnt,dp->bdnp", x, self.W) + self.b
#         return y.permute(0, 3, 1, 2)  # Reorder back


class Upsample(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = ddpm_conv3x3(channels, channels)
        self.with_conv = with_conv

    def forward(self, x):
        B, C, H, W = x.shape
        h = F.interpolate(x, (H * 2, W * 2), mode="nearest")
        if self.with_conv:
            h = self.Conv_0(h)
        return h


# class UpsampleST(nn.Module):
#     def __init__(self, channels, with_conv=False):
#         super().__init__()
#         if with_conv:
#             self.Conv_0 = SpatioTemporalLayer(channels, channels)
#         self.with_conv = with_conv

#     def forward(self, x):
#         B, C, N, T = x.shape
#         h = F.interpolate(x, (N, T * 2), mode="nearest")
#         if self.with_conv:
#             h = self.Conv_0(h)
#         return h


class UpsampleST(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.Conv_0 = nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 4),
                stride=(1, 2),
                padding=(0, 1),
            )

    def forward(self, x):
        B, C, N, T = x.shape
        if self.with_conv:
            h = self.Conv_0(x)
        else:
            h = F.interpolate(x, (N, T * 2), mode="nearest")
        return h


class Downsample(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = ddpm_conv3x3(channels, channels, stride=2, padding=0)
        self.with_conv = with_conv

    def forward(self, x):
        B, C, H, W = x.shape
        # Emulate 'SAME' padding
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1))
            x = self.Conv_0(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

        assert x.shape == (B, C, H // 2, W // 2)
        return x


class DownsampleST(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = nn.Conv2d(
                channels,
                channels,
                stride=(1, 2),
                padding=(0, 0),
                kernel_size=(1, 2),
            )
        self.with_conv = with_conv

    def forward(self, x):
        B, C, N, T = x.shape
        # Emulate 'SAME' padding
        if self.with_conv:
            x = self.Conv_0(x)

        assert x.shape == (B, C, N, T // 2)
        return x


class ResnetBlockDDPM(nn.Module):
    """The ResNet Blocks used in DDPM."""

    def __init__(
        self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.act = act
        self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = ddpm_conv3x3(out_ch, out_ch, init_scale=0.0)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = ddpm_conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        B, C, H, W = x.shape
        assert C == self.in_ch
        out_ch = self.out_ch if self.out_ch else self.in_ch
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if C != out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        return x + h


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetST(nn.Module):

    def __init__(
        self,
        act,
        in_ch,
        adj,
        d_k,
        d_v,
        hidden_size,
        num_vertices,
        n_heads,
        K,
        s4_state_dimm,
        s4_num_temporal_layers,
        out_ch=None,
        temb_dim=None,
        conv_shortcut=False,
        dropout=0.1,
        temporal_layer="gru",
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=(in_ch // 4), num_channels=in_ch, eps=1e-6
        )
        self.act = act
        self.Conv_0 = EnhancedSpatioTemporalLayer(
            in_features=in_ch,
            out_features=out_ch,
            adj=adj,
            hidden_size=hidden_size,
            num_of_vertices=num_vertices,
        )
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=(out_ch // 4), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)

        self.Conv_1 = EnhancedSpatioTemporalLayer(
            in_features=out_ch,
            out_features=out_ch,
            adj=adj,
            hidden_size=hidden_size,
            num_of_vertices=num_vertices,
        )

        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = EnhancedSpatioTemporalLayer(
                    in_features=in_ch,
                    out_features=out_ch,
                    adj=adj,
                    hidden_size=hidden_size,
                    num_of_vertices=num_vertices,
                )
            else:
                self.NIN_0 = NIN(in_ch, out_ch)

        self.out_ch = out_ch
        self.in_ch = in_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, adj, temb=None):
        B, C, T, N = x.shape
        assert C == self.in_ch
        out_ch = self.out_ch if self.out_ch else self.in_ch

        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h, adj)

        if temb is not None:
            h = h + self.Dense_0(self.act(temb))[:, :, None, None]

        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h, adj)

        if C != out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x, adj)
            else:
                x = self.NIN_0(x)

        return x + h


def greatest_divisor(n, max_divisor):
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1
