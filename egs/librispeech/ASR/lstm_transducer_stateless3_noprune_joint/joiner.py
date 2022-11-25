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
        pronouncer_stop_gradient: bool,
        pronouncer_lambda: float,
        pronouncer_normalize: bool,
    ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim)
        self.output_linear = ScaledLinear(joiner_dim, vocab_size)

        self.pronouncer_stop_gradient = pronouncer_stop_gradient
        self.pronouncer_lambda = pronouncer_lambda
        self.pronouncer_normalize = pronouncer_normalize
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
        x_target: Optional[torch.Tensor] = None,
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
          x_target:
            A 3-D tensor of shape (N, T, D').
        Returns:
          Return a tensor of shape (N, T, U, C).
        """
        assert encoder_out.size(0) == decoder_out.size(0)
        assert encoder_out.size(-1) == self.encoder_dim
        assert decoder_out.size(-1) == self.decoder_dim
        assert encoder_out.ndim in (3, 4)
        assert encoder_out.ndim == decoder_out.ndim

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
        if x_target is None:
            x_logp = None
        else:
            if self.pronouncer_stop_gradient:
                activations = activations.detach()
            x_logp = self.pronouncer(
                activations,
                x_target,
            )  # [N, T, U]
            # If pronouncer is to be normalized,
            # get unconditional probability on x using
            if self.pronouncer_normalize:
                '''
                norm_activations = torch.tanh(encoder_out)  # [N, T, 1, J]
                if self.pronouncer_stop_gradient:
                    norm_activations = norm_activations.detach()
                x_logp_denom = self.pronouncer(
                    norm_activations,
                    x_target,
                )  # [N, T, 1]
                '''
                x_logp_denom = x_logp[:, :, 0:1]  # [N, T, 1]
                # If the gradient of denom is not stopped,
                # the trained model just minimizes P_theta(x_t+1 | x_t)
                # which is not an intended behaviour
                x_logp_denom = x_logp_denom.detach()
                if self.training:
                    print('x_logp[0:1]:', x_logp[0:1])
                x_logp = x_logp - x_logp_denom  # [N, T, U]
                if self.training:
                    print('x_logp_normed[0:1]:', x_logp[0:1])

            x_logp = x_logp * self.pronouncer_lambda
        return logits, x_logp
