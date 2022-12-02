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

# from icefall.utils import is_jit_tracing

import numpy as np


class Pronouncer(nn.Module):
    def __init__(
        self,
        joint_input_dim: int,
    ):
        super().__init__()
        self.joint_input_dim = joint_input_dim
        self.output_linear = ScaledLinear(2 * joint_input_dim, 1)

    def forward(
        self,
        joint_input: torch.Tensor,
        joint_input_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          joint_input:
            Intermediate output from the joiner. Its shape is (N, T, U, J)
        Returns:
          Return a tensor of shape (N, T, U).
        """
        assert joint_input.ndim in (4,)
        assert joint_input.shape == joint_input_next.shape
        device = joint_input.device

        # shift inputs by chunk-size(=1)
        x_input = torch.cat(
            [joint_input, joint_input_next], dim=-1
        )  # [N, T, U, 2J]
        r_logp = self.output_linear(x_input)[:, :, :, 0]  # [N, T, U]
        return r_logp
