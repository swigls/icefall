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

from icefall.utils import is_jit_tracing


class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim)
        self.output_linear = ScaledLinear(joiner_dim, vocab_size)

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool = True,  # unused
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, self.encdoer_dim).
          decoder_out:
            Output from the decoder. Its shape is (N, U, self.decoder_dim).
        Returns:
          Return a tensor of shape (N, T, U, C).
        """
        assert encoder_out.size(0) == decoder_out.size(0)
        assert encoder_out.size(2) == self.encoder_dim
        assert decoder_out.size(2) == self.decoder_dim
        #if not is_jit_tracing():
        #    assert encoder_out.ndim == decoder_out.ndim
        #    assert encoder_out.ndim in (2, 4)
        #    assert encoder_out.shape == decoder_out.shape

        encoder_out = encoder_out.unsqueeze(2)  # (N, T, 1, E)
        decoder_out = decoder_out.unsqueeze(1)  # (N, 1, U, D)

        encoder_out = self.encoder_proj(encoder_out)  # (N, T, 1, J)
        decoder_out = self.decoder_proj(decoder_out)  # (N, T, 1, J)

        x = encoder_out + decoder_out  # (N, T, U, V)

        activations = torch.tanh(x)

        logits = self.output_linear(activations)

        return logits
