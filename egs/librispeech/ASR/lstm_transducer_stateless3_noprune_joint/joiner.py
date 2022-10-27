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

from pronouncer import Pronouncer
from typing import Optional


class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
        kmeans_model: str,
    ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim)
        self.output_linear = ScaledLinear(joiner_dim, vocab_size)

        self.pronouncer = Pronouncer(
            joiner_dim,
            kmeans_model=kmeans_model,
        )

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        x: torch.Tensor,
        x_lens: Optional[torch.Tensor] = None,
        h_lens: Optional[torch.Tensor] = None,
        project_input: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, self.encoder_dim)
            or (N, T, 1, self.joiner_dim).
          decoder_out:
            Output from the decoder. Its shape is (N, U, self.decoder_dim)
            or (N, 1, U, self.joiner_dim).
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
        Returns:
          Return a tensor of shape (N, T, U, C).
        """
        assert encoder_out.size(0) == decoder_out.size(0)
        assert encoder_out.size(-1) == self.encoder_dim
        assert decoder_out.size(-1) == self.decoder_dim
        assert encoder_out.ndim in (3, 4)
        assert encoder_out.ndim == decoder_out.ndim
        # if not is_jit_tracing():
        #    assert encoder_out.ndim == decoder_out.ndim
        #    assert encoder_out.ndim in (2, 4)
        #    assert encoder_out.shape == decoder_out.shape

        if encoder_out.ndim == 3:
            encoder_out = encoder_out.unsqueeze(2)  # (N, T, 1, E)
            decoder_out = decoder_out.unsqueeze(1)  # (N, 1, U, D)

        if project_input:
            encoder_out = self.encoder_proj(encoder_out)  # (N, T, 1, J)
            decoder_out = self.decoder_proj(decoder_out)  # (N, 1, U, J)

        joint_emb = encoder_out + decoder_out  # (N, T, U, J)

        activations = torch.tanh(joint_emb)

        logits = self.output_linear(activations)  # (N, T, U, V)

        # calculating log-prob of x
        # NOTE: encoder out frame h_t corresponsd to x_(4t:4t+8)
        #       (check the Conv2dSubsampling)
        #       so, x_(4t+9:4t+12) should be estimated from h_t = Enc(x(1:4t+8))
        x_logp = self.pronouncer(
            activations,
            x,
            x_lens,
            h_lens,
        )
        return logits, x_logp
