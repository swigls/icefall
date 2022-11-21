# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

import torch
import torch.nn as nn
from scaling import ScaledLinear

import torch.distributions as D
import torch.nn.functional as F
from scaling import DoubleSwish

from typing import Optional
# from icefall.utils import is_jit_tracing

import numpy as np


class Pronouncer(nn.Module):
    def __init__(
        self,
        joint_input_dim: int,
        cond_input_dim: int = 128,
    ):
        super().__init__()

        self.joint_input_dim = joint_input_dim
        self.cond_input_dim = cond_input_dim

        self.joint_input_proj = ScaledLinear(joint_input_dim, cond_input_dim)
        self.maf = MAF(
            n_blocks=1,
            input_size=4*80,
            hidden_size=256,
            n_hidden=1,
            cond_label_size=self.cond_input_dim,
            activation="DoubleSwish",
            input_order="sequential",
        )

    def forward(
        self,
        joint_input: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          joint_input:
            Intermediate output from the joiner. Its shape is (N, T, U, J)
          x_target:
            A 3-D tensor of shape (N, T, D').
        Returns:
          Return a tensor of shape (N, T, U).
        """
        assert joint_input.ndim in (4,)
        assert x_target.ndim in (3,)
        assert x_target.size(-1) == 320
        device = joint_input.device
        N, T, U, _ = joint_input.shape

        # Tile x_target to match shape
        x_target = torch.tile(
            x_target.unsqueeze(2), (1, 1, U, 1)
        )  # [N, T_h, U, 4D]

        cond_input = self.joint_input_proj(joint_input)  # [N, T_h, U, C]
        logp = self.maf.log_prob(
            x=x_target,
            y=cond_input,
        )  # [N, T_h, U]
        return logp


#######################
# Flow definitions
#######################
def create_masks(
    input_size,
    hidden_size,
    n_hidden,
    input_order="sequential",
    input_degrees=None,
):
    # MADE paper sec 4:
    # degrees of connections between layers
    # -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args
    # (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order
    # (sequential or random)
    if input_order == "sequential":
        degrees += (
            [torch.arange(input_size)]
            if input_degrees is None
            else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += (
            [torch.arange(input_size) % input_size - 1]
            if input_degrees is None
            else [input_degrees % input_size - 1]
        )

    elif input_order == "random":
        degrees += (
            [torch.randperm(input_size)]
            if input_degrees is None
            else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [
                torch.randint(min_prev_degree, input_size, (hidden_size,))
            ]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += (
            [torch.randint(min_prev_degree, input_size, (input_size,)) - 1]
            if input_degrees is None
            else [input_degrees - 1]
        )

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


# class MaskedLinear(nn.Linear):
class MaskedLinear(ScaledLinear):
    """MADE building block layer"""

    # def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
    def __init__(self, input_size, n_outputs, mask, cond_label_size=0):
        # super().__init__(input_size, n_outputs)
        super().__init__(input_size + cond_label_size, n_outputs)

        self.register_buffer("mask", mask)

        self.cond_label_size = cond_label_size
        self.input_size = input_size
        # if cond_label_size is not None:
        #     self.cond_weight = nn.Parameter(
        #         torch.rand(n_outputs, cond_label_size)
        #         / math.sqrt(cond_label_size)
        #     )

    def forward(self, x, y=None):
        # out = F.linear(x, self.weight * self.mask, self.bias)
        # x: [N, T, U, D], y: [N, T, U, C]
        # out: [N, T, U, O]
        w = self.get_weight()  # [O, D+C]
        if self.input_size==320:
            print('ML w:', w)
            print('ML weight:', self.weight)
            print('ML weight_scale:', self.weight_scale)
        if y is None:
            out = F.linear(x, w * self.mask, self.get_bias())
        else:
            w1 = w[:, :-self.cond_label_size]  # [O, D]
            w2 = w[:, -self.cond_label_size:]  # [O, C]
            out = F.linear(x, w1 * self.mask, self.get_bias())
            out = out + F.linear(y, w2)
        # if y is not None:
        #     out = out + F.linear(y, self.cond_weight)
        return out


class FlowSequential(nn.Sequential):
    """Container for layers of a normalizing flow"""

    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = (
                sum_log_abs_det_jacobians + log_abs_det_jacobian
            )
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = (
                sum_log_abs_det_jacobians + log_abs_det_jacobian
            )
        return u, sum_log_abs_det_jacobians


class MADE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="DoubleSwish",
        input_order="sequential",
        input_degrees=None,
    ):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the
                           autoregressive masks (sequential|random) or the
                           order flipped from the previous layer in a
                           stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(
            input_size, hidden_size, n_hidden, input_order, input_degrees
        )

        # setup activation
        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "tanh":
            activation_fn = nn.Tanh()
        elif activation == "DoubleSwish":
            activation_fn = DoubleSwish()
        else:
            raise ValueError("Check activation function.")

        # construct model
        self.net_input = MaskedLinear(
            input_size, hidden_size, masks[0], cond_label_size
        )
        self.net = []
        for m in masks[1:-1]:
            self.net += [
                activation_fn,
                MaskedLinear(hidden_size, hidden_size, m),
            ]
        self.net += [
            activation_fn,
            MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1)),
        ]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = -loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
            x[:, i] = u[:, i] * torch.exp(loga[:, i]) + m[:, i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(
            self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=-1
        )


class MAF(nn.Module):
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="DoubleSwish",
        input_order="sequential",
        # batch_norm=False,
    ):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [
                MADE(
                    input_size,
                    hidden_size,
                    n_hidden,
                    cond_label_size,
                    activation,
                    input_order,
                    self.input_degrees,
                )
            ]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            # modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(
            self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=-1
        )
