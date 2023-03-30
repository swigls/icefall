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
from conformer import ConformerEncoder, ConformerEncoderLayer, PositionalEncoding, RelPositionalEncoding
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
        #self.decoder_pos = RelPositionalEncoding(decoder_dim, 0.)
        self.decoder_pos = PositionalEncoding(decoder_dim, 0.)
        layer = ConformerEncoderLayer(
            d_model=decoder_dim,
            nhead=8,
            dim_feedforward=1024,  # 1536?
            dropout=0.1,
            cnn_module_kernel=15,  # 31?
            causal=True,
            rel_pos=False,
        )
        self.model = ConformerEncoder(
            encoder_layer=layer,
            num_layers=3,
        )
        '''
        layer = ZipformerEncoderLayer(
            d_model=decoder_dim,  # 384
            attention_dim=256,  # 192
            nhead=8,
            feedforward_dim=1536,
            dropout=0.1,
            cnn_module_kernel=15,
            rel_pos=False,
            whiten_output=False,
        )
        self.model = ZipformerEncoder(
            encoder_layer=layer,
            num_layers=4,
            dropout=0.1,
            warmup_begin=0.,
            warmup_end=0.,
        )
        '''
        self.output_linear = nn.Linear(
            decoder_dim,
            vocab_size,
        )

    def forward(
        self,
        input: torch.Tensor,
        T: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        src_key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          input:
            A 3-D tensor of shape (N, T+U, D).
          attn_mask:
            A 3-D tensor of shape (T+U, T+U).
          src_key_mask:
            A 3-D tensor of shape (N, T+U)
        Returns:
          Return a tensor of shape (N, T+U, vocab_size).
        """
        #src, pos_emb = self.decoder_pos(y[:, :])  # src:(N, T+U, D), pos_emb:(N, 2(T+U)-1, D)
        x, _ = self.decoder_pos(input[:, :T])  # x:(N, T, D)
        y, _ = self.decoder_pos(input[:, T:])  # y:(N, U, D)
        src = torch.cat([x, y], dim=1)  # (N, T+U, D)
        src = src.permute(1, 0, 2)  # (T+U, N, D)
        model_out = self.model(
            src=src,
            pos_emb=None,
            mask=attn_mask,
            src_key_padding_mask=src_key_mask,
        )  # (T+U, N, D)
        model_out = model_out.permute(1, 0, 2)  # (N, T+U, D)
        out = self.output_linear(model_out)  # (N, T+U, vocab_size)
        return out
    
    def chunk_forward(
        self,
        input: torch.Tensor,
        T: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        src_key_mask: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          input:
            A 3-D tensor of shape (N, T+U, D).
          attn_mask:
            A 3-D tensor of shape (T+U, T+U).
          src_key_mask:
            A 3-D tensor of shape (N, T+U)
          states:
            Len==2 list of 3-D tensors of shape [(left_context, N, attention_dim), (kernel_size-1, N, conv_dim)].
        Returns:
          Return a tensor of shape (N, T+U, vocab_size).
        """
        #src, pos_emb = self.decoder_pos(y[:, :])  # src:(N, T+U, D), pos_emb:(N, 2(T+U)-1, D)
        x, _ = self.decoder_pos(input[:, :T])  # x:(N, T, D)
        y, _ = self.decoder_pos(input[:, T:])  # y:(N, U, D)
        src = torch.cat([x, y], dim=1)  # (N, T+U, D)
        src = src.permute(1, 0, 2)  # (T+U, N, D)

        if states is None:
            device = src.device
            states = [[torch.zeros(0, src.size(1), 384, device=device) for _ in range(3)],\
                      [torch.zeros(14, src.size(1), 384, device=device) for _ in range(3)]]
        model_out, states = self.model.chunk_forward(
            src=src,
            pos_emb=None,
            states=states,
            mask=attn_mask,
            src_key_padding_mask=src_key_mask,
            left_context=9999,
        )  # (T+U, N, D)
        model_out = model_out.permute(1, 0, 2)  # (N, T+U, D)
        out = self.output_linear(model_out)  # (N, T+U, vocab_size)
        return out, states
