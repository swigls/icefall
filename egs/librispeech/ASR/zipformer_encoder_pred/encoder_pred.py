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
from scaling import ScaledLinear, ScaledConv1d, SwooshL, BiasNorm
from zipformer import ConvolutionModule


class EncoderPred(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        pred_bottleneck_dim: int,
        pred_kernel_size: int,
        pred_num_layers: int,
        pred_detach: bool,
        pred_l2_to_logp: str,
        pred_logp_scale: float,
        pred_logp_ratio_clamp: float,
        pred_l2_norm_loss: bool,
    ):
        super().__init__()

        self.encoder = encoder
        
        self.encoder_proj = ScaledLinear(encoder_dim, pred_bottleneck_dim, initial_scale=0.25)
        self.decoder_proj = ScaledLinear(decoder_dim, pred_bottleneck_dim, initial_scale=0.25)
        # assert len(encoder.chunk_size) == 1, encoder.chunk_size
        # chunk_size = encoder.chunk_size[0]
        # half_kernel_size = (pred_kernel_size + 1) // 2
        # num_context_chunk_per_layer = \
        #     (half_kernel_size-1 + chunk_size-1) // chunk_size
        # self.pred_context_chunk = num_context_chunk_per_layer * pred_num_layers

        self.pred_layers = []
        for _ in range(pred_num_layers):
            self.pred_layers.append(
                ConvolutionModule(
                    channels=pred_bottleneck_dim,
                    kernel_size=pred_kernel_size,  # 7 in data2vec 2.0, where frame shift was 20 ms.
                    causal=True,
                ))
            self.pred_layers.append(BiasNorm(pred_bottleneck_dim))
            self.pred_layers.append(SwooshL())
        self.pred_layers.append(nn.Linear(pred_bottleneck_dim, encoder_dim))
        self.pred = nn.Sequential(*self.pred_layers)

        self.pred_l2_norm_loss = pred_l2_norm_loss
        self.pred_l2_to_logp = pred_l2_to_logp
        self.pred_logp_scale = pred_logp_scale
        self.pred_logp_ratio_clamp = pred_logp_ratio_clamp

        self.pred_detach = pred_detach

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, E).
          encoder_out_lens:
            Lengths of the encoder output. Its shape is (N,).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, D).
          project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          logp_ratio:
            The log probability ratio. Its shape is (N, T, s_range).
          l2_numer:
            The L2 loss between the predicted encoder features and the 
            target encoder features, where encoder features are predicted
            from the current chunk and the decoder state.
          l2_denom:
            The L2 loss between the predicted encoder features and the
            target encoder features, where encoder features are predicted
            from the current chunk only.
        """
        assert encoder_out.ndim == decoder_out.ndim, (encoder_out.shape, decoder_out.shape)
        assert len(self.encoder.chunk_size) == 1, self.encoder.chunk_size
        chunk_size = self.encoder.chunk_size[0]
        assert chunk_size > 0, chunk_size

        if project_input:
            # pruned encoder_out is duplicates of the same features on the label axis.
            if self.pred_detach:
                encoder_out = encoder_out.detach()
                decoder_out = decoder_out.detach()

            input_denom = self.encoder_proj(encoder_out[:, :, 0:1, :])  # (N, T, 1, B)
            input_numer = input_denom + self.decoder_proj(decoder_out)  # (N, T, s_range, B)

            # # roll the inputs to the right by one chunk
            # input_denom_roll = torch.roll(input_denom, shifts=chunk_size, dims=1)
            # input_numer_roll = torch.roll(input_numer, shifts=chunk_size, dims=1)
            # # mask out the first chunk
            # input_denom_roll[:, 0:chunk_size, :, :] = 0
            # input_numer_roll[:, 0:chunk_size, :, :] = 0

            # There are two options of implementation
            # 1. predict the next chunk from the current chunk (no masking)
            # 2. predict the next chunk with the zero-masking on the next chunk values
            # Here, we implement the first option for simplicity
            # The second option is more complicated because we need to
            # feedforward encoder_pred on the masked input for the next chunk
            # which is computationally heavy.
            N, T, s_range, B = input_numer.shape
            E = encoder_out.shape[-1]
            input_denom = input_denom.permute(1, 0, 2, 3).reshape(T, N, B)
            input_numer = input_numer.permute(1, 0, 2, 3).reshape(T, N*s_range, B)

            pred_next_denom = self.pred(input_denom)  # (T, N, 1, E)
            pred_next_numer = self.pred(input_numer)  # (T, N, s_range, E)

            pred_next_denom = pred_next_denom.reshape(T, N, 1, -1).permute(1, 0, 2, 3)
            pred_next_numer = pred_next_numer.reshape(T, N, s_range, -1).permute(1, 0, 2, 3)
            print('pred_next_denom norm', torch.norm(pred_next_denom[0,:5,0], p=2, dim=-1))
            # print('pred_next_denom value', pred_next_denom[0,:5,0,0:5])
            # print('pred_next_numer norm', torch.norm(pred_next_numer[0,:5,0], p=2, dim=-1))

            # roll the target encoder features to the left by one chunk
            encoder_target = torch.roll(encoder_out, shifts=-chunk_size, dims=1)  # (N, T, s_range, E)
            encoder_target[:, -chunk_size:, :, :] = 0
            encoder_target = encoder_target / E**0.5
            encoder_target = encoder_target.detach()
            print('encoder_target norm', torch.norm(encoder_target[0,:5,0], p=2, dim=-1))
            # print('encoder_target value', encoder_target[0,:5,0,0:5])
            # print('encoder_target', encoder_target[0,:10,0])

            # calculate L2 distance for each position (t,u) in the rnnt grid
            # L2 distance is divided by bottleneck_dim
            l2_denom = torch.norm((pred_next_denom - encoder_target), p=2, dim=-1)  # (N, T, 1)
            l2_numer = torch.norm((pred_next_numer - encoder_target), p=2, dim=-1)  # (N, T, s_range)

            # mask out except t=1:(T_i - C) (T_i: length of the encoder output for i-th sample)
            # not that the last chunk_size frames are not valid prediction,
            # because we don't have the target encoder features for them.
            target_mask = (torch.arange(encoder_out.shape[1], device=encoder_out.device) \
                          .expand(encoder_out.shape[0], encoder_out.shape[1]) \
                          < encoder_out_lens.unsqueeze(1) - chunk_size) \
                          .to(encoder_out.dtype)  # (N, T)
            l2_denom = l2_denom * target_mask.unsqueeze(-1)  # (N, T, 1)
            l2_numer = l2_numer * target_mask.unsqueeze(-1)  # (N, T, s_range)
            # print('l2_denom', l2_denom[0,:10,0])
            # print('l2_numer', l2_numer[0,:10,0])

            # calculate log-prob. ratio
            if self.pred_l2_to_logp.startswith("Gaussian"):
                # Assumed variance of 1
                logp_denom = -0.5 * l2_denom ** 2  # (N, T, 1)
                logp_numer = -0.5 * l2_numer ** 2  # (N, T, s_range)
                ratio = logp_numer - logp_denom  # (N, T, s_range)

                # At chunk endpoint, the logp_ratio value equals to the
                # sum of the logp_ratio values of the next chunk
                # Note that except t % C == -1, the logp_ratio value is zero
                # because information gain occurs only at the end of every chunk
                logp_ratio = torch.zeros_like(ratio)
                ratio = torch.nn.functional.pad(ratio, (0,0,0,-T%chunk_size))  # (N, [T], s_range)
                ratio = ratio.reshape(N, -1, chunk_size, s_range)
                ratio = torch.sum(ratio, dim=2)  # (N, [T//C], s_range)
                logp_ratio[:, chunk_size-1::chunk_size, :] = ratio
            else:
                assert False, "not implemented"

            # scale and clamp the log-prob. ratio
            logp_ratio = logp_ratio * self.pred_logp_scale
            print('logp_ratio after scale', logp_ratio[0,chunk_size-1:chunk_size*3:chunk_size,0])
            if self.pred_logp_ratio_clamp > 0:
                logp_ratio = torch.clamp(logp_ratio,
                                        min=-self.pred_logp_ratio_clamp,
                                        max=self.pred_logp_ratio_clamp)
            
            # calculate loss functions for training encoder_prod
            # DEPRECATED: mean over valid time frames
            # NEW: mean over s_range, sum over valid time frames
            # l2_denom = torch.sum(
            #     torch.sum(l2_denom, dim=(-2,-1)) / torch.sum(encoder_mask, dim=-1))  # scalar
            # l2_numer = torch.sum(
            #     torch.sum(l2_numer, dim=(-2,-1)) / torch.sum(encoder_mask, dim=-1))  # scalar
            if self.pred_l2_norm_loss:
                l2_denom = torch.sum(torch.mean(l2_denom, dim=-1))
                l2_numer = torch.sum(torch.mean(l2_numer, dim=-1))
            else:
                l2_denom = torch.sum(torch.mean(l2_denom ** 2, dim=-1))  # scalar
                l2_numer = torch.sum(torch.mean(l2_numer ** 2, dim=-1))  # scalar
        else:
            assert False, "not implemented"

        return logp_ratio, l2_denom, l2_numer
