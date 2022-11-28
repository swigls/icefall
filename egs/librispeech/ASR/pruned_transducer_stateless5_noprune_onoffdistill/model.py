# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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


from typing import Tuple

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from scaling import ScaledLinear

from icefall.utils import add_sos


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and
            (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output
            contains unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        warmup: float = 1.0,
        reduction: str = "sum",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
          warmup:
            A value warmup >= 0 that determines which modules are active, values
            warmup > 1 "are fully warmed up" and all modules will be active.
          reduction:
            "sum" to sum the losses over all utterances in the batch.
            "none" to return the loss in a 1-D tensor for each utterance
            in the batch.
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert reduction in ("sum", "none"), reduction
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        encoder_out_off, _ = self.encoder(
            x,
            x_lens,
            warmup=warmup,
            no_dynamic_chunk=True
        )
        encoder_out_on, x_lens = self.encoder(
            x,
            x_lens,
            warmup=warmup
        )
        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (x.size(0), 4), dtype=torch.int64, device=x.device
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        logits_off = self.joiner(
            encoder_out_off, 
            decoder_out, 
            offline=True,
        )  # [N, T, U, V]
        logits_on, r = self.joiner(
            encoder_out_on, 
            decoder_out, 
        )  # [N, T, U, V]

        with torch.cuda.amp.autocast(enabled=False):
            loss_off = k2.rnnt_loss(
                logits=logits_off.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction=reduction,
                py_add=None,
            )

            loss_on = k2.rnnt_loss(
                logits=logits_on.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction=reduction,
                py_add=r.detach(),
            )

            loss_on_r = k2.rnnt_loss(
                logits=logits_on.detach().float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction=reduction,
                py_add=r,
            )
            loss_r = torch.norm(
                loss_on_r - loss_off.detach(),
                p=1,
            )

            loss_out = torch.zeros_like(loss_r)  # TODO

        return (
            loss_off,
            loss_on,
            loss_r,
            loss_out
        )