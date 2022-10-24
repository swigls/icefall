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

class Pronouncer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = 2 * 4 * 80  # (mu,logsigma) * time_reduce * feat-dim

        self.output_linear = ScaledLinear(input_dim, self.output_dim)


    def forward(
        self,
        joint_input:
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          joint_input:
            Intermediate output from the joiner. Its shape is (N, T, U+1, J)
          x:
            A 3-D tensor of shape (N, T_orig, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
        Returns:
          Return a tensor of shape (N, T, U+1).
        """
        assert joint_input.ndim in (4,)
        assert x.ndim in (3,)
        assert x.size(-1) == 80
        # if not is_jit_tracing():
        #    assert encoder_out.ndim == decoder_out.ndim
        #    assert encoder_out.ndim in (2, 4)
        #    assert encoder_out.shape == decoder_out.shape

        joint_emb = encoder_out + decoder_out  # (N, T, U, J)

        activations = torch.tanh(joint_emb)

        logits = self.output_linear(activations)  # (N, T, U, V)

        # calculating log-prob of x
        # NOTE: encoder out frame h_t corresponsd to x_(4t:4t+8)
        #       (check the Conv2dSubsampling)
        #       so, x_(4t+9:4t+12) should be estimated from h_t = Enc(x(1:4t+8))

        # x_logp = 

        return x_logp
