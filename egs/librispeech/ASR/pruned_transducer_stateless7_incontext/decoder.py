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
import torch.nn.functional as F

from zipformer import ZipformerEncoder, ZipformerEncoderLayer
from typing import Optional

class Decoder(nn.Module):
    """This class modifies the stateless decoder from the following paper:

        RNN-transducer with stateless prediction network
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419

    It removes the recurrent connection from the decoder, i.e., the prediction
    network. Different from the above paper, it adds an extra Conv1d
    right after the embedding layer.

    TODO: Implement https://arxiv.org/pdf/2109.07513.pdf
    """

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int,
        #context_size: int,
        #encoder: nn.Module,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          decoder_dim:
            Dimension of the input embedding, and of the decoder output.
          blank_id:
            The ID of the blank symbol.
          context_size:
            Number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        """
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=decoder_dim,
            padding_idx=blank_id,
        )
        self.blank_id = blank_id

        #assert context_size >= 1, context_size
        #self.context_size = context_size
        self.vocab_size = vocab_size
        '''
        if context_size > 1:
            self.conv = nn.Conv1d(
                in_channels=decoder_dim,
                out_channels=decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=decoder_dim // 4,  # group size == 4
                bias=False,
            )
        '''
        zipformer_layer = ZipformerEncoderLayer(
            d_model=decoder_dim,  # 384
            attention_dim=decoder_dim//2,  # 192
            nhead=8,
            feedforward_dim=1024,
            dropout=0.1,
            cnn_module_kernel=1,
        )
        self.zipformer = ZipformerEncoder(
            encoder_layer=zipformer_layer,
            num_layers=3,
            dropout=0.1,
            warmup_begin=4000.,
            warmup_end=5000.,
        )
        self.output_linear = nn.Linear(
            decoder_dim,
            vocab_size,
        )

    def forward(
        self,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          y:
            A 3-D tensor of shape (N, T+U, D).
          mask:
            A 3-D tensor of shape (T+U, T+U).
          src_key_mask:
            A 3-D tensor of shape (N, T+U)
        Returns:
          Return a tensor of shape (N, T+U, vocab_size).
        """
        src = y.permute(1, 0, 2)  # (T+U, N, D)
        zipformer_out = self.zipformer(
            src=src,
            mask=mask,
            src_key_padding_mask=src_key_mask,
        )  # (T+U, N, D)
        zipformer_out = zipformer_out.permute(1, 0, 2)  # (N, T+U, D)
        out = self.output_linear(zipformer_out)  # (N, T+U, vocab_size)
        return out
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          need_pad:
            True to left pad the input. Should be True during training.
            False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        '''
        y = y.to(torch.int64)
        # this stuff about clamp() is a temporary fix for a mismatch
        # at utterance start, we use negative ids in beam_search.py
        if torch.jit.is_tracing():
            # This is for exporting to PNNX via ONNX
            embedding_out = self.embedding(y)
        else:
            embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(-1)
        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)
            if need_pad is True:
                embedding_out = F.pad(embedding_out, pad=(self.context_size - 1, 0))
            else:
                # During inference time, there is no need to do extra padding
                # as we only need one output
                assert embedding_out.size(-1) == self.context_size
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
        embedding_out = F.relu(embedding_out)
        return embedding_out
        '''
