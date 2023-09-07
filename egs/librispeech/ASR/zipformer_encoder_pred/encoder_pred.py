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
from collections import OrderedDict

from scaling_lstm import ScaledLSTM

from typing import Optional

import normflows as nf


class EncoderPred(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        pred_bottleneck_dim: int,
        pred_kernel_size: int,
        pred_num_layers: int,
        pred_detach: int,
        pred_l2_to_logp: str,
        pred_logp_scale: float,
        pred_logp_ratio_clamp: float,
        pred_l2_norm_loss: bool,
        pred_enc_in_rnn: bool,
        pred_dec_in_rnn: bool,
        pred_dec_in_raw: bool,
        pred_flow_depth: int,
        pred_flow_num_blocks: int,
        pred_flow_hidden_dim: int,
    ):
        super().__init__()

        self.encoder = encoder
        
        self.encoder_proj = ScaledLinear(encoder_dim, pred_bottleneck_dim, initial_scale=0.25)
        self.decoder_proj = ScaledLinear(decoder_dim, pred_bottleneck_dim, initial_scale=0.25)
        if pred_enc_in_rnn:
            self.pred_enc_in_rnn = ScaledLSTM(
                input_size=encoder_dim,
                hidden_size=encoder_dim,
                proj_size=0,
                num_layers=1,
                dropout=0.0,
                bidirectional=False,
            )
        else:
            self.pred_enc_in_rnn = None
        if pred_dec_in_rnn:
            self.pred_dec_in_rnn = ScaledLSTM(
                input_size=decoder_dim,
                hidden_size=decoder_dim,
                proj_size=0,
                num_layers=1,
                dropout=0.0,
                bidirectional=False,
            )
        else:
            self.pred_dec_in_rnn = None

        self.pred_dec_in_raw = pred_dec_in_raw

        assert len(encoder.chunk_size) == 1, encoder.chunk_size
        chunk_size = encoder.chunk_size[0]
        half_kernel_size = (pred_kernel_size + 1) // 2
        num_context_chunk_per_layer = \
            (half_kernel_size-1 + chunk_size-1) // chunk_size
        self.pred_context_chunk = num_context_chunk_per_layer * pred_num_layers

        #assert pred_bottleneck_dim==encoder_dim, (pred_bottleneck_dim, encoder_dim)
        self.pred_convs = []
        self.pred_norms = []
        #self.pred_layers.append(('SwooshL_proj', SwooshL()))
        for i in range(pred_num_layers):
            self.pred_convs.append(('ConvModule%d'%i,
                ConvolutionModule(
                    channels=pred_bottleneck_dim,
                    kernel_size=pred_kernel_size,  # 7 in data2vec 2.0, where frame shift was 20 ms.
                    causal=True,
                    fixed_chunk_size=chunk_size,
                )))
            # self.pred_layers.append(('Conv1d_%d'%i,
            #     ScaledConv1d(
            #         in_channels=pred_bottleneck_dim,
            #         out_channels=pred_bottleneck_dim,
            #         kernel_size=pred_kernel_size,  # 7 in data2vec 2.0, where frame shift was 20 ms.
            #         bias=True,
            #         padding=pred_kernel_size // 2,
            #         groups=1,
            #         dilation=1,
            #         stride=1,
            #         initial_scale=0.25,
            #     )))
            #self.pred_norms.append(('BiasNorm%d'%i, BiasNorm(pred_bottleneck_dim)))
            self.pred_norms.append(('LayerNorm%d'%i, torch.nn.LayerNorm(pred_bottleneck_dim)))
        # self.pred_layers.append(('out_linear',nn.Linear(pred_bottleneck_dim, encoder_dim)))
        #self.pred = nn.Sequential(OrderedDict(self.pred_layers))
        self.pred_convs_dummy = nn.Sequential(OrderedDict(self.pred_convs))
        self.pred_norms_dummy = nn.Sequential(OrderedDict(self.pred_norms))

        self.pred_flow_depth = pred_flow_depth
        if self.pred_flow_depth == 0:
            self.pred_out = nn.Linear(pred_bottleneck_dim, encoder_dim)  # Just predict mean of Gaussian
        else:
            flows = []
            for i in range(pred_flow_depth):
                flows += [nf.flows.MaskedAffineAutoregressive(encoder_dim,
                                                              pred_flow_hidden_dim, 
                                                              context_features=pred_bottleneck_dim, 
                                                              num_blocks=pred_flow_num_blocks)]
                flows += [nf.flows.LULinearPermute(encoder_dim)]
            q0 = nf.distributions.DiagGaussian(encoder_dim, trainable=False)
            self.flow = nf.ConditionalNormalizingFlow(q0, flows)

        assert pred_l2_norm_loss == False, "deprecated"
        self.pred_l2_to_logp = pred_l2_to_logp
        self.pred_logp_scale = pred_logp_scale
        self.pred_logp_ratio_clamp = pred_logp_ratio_clamp

        self.pred_detach = pred_detach

    def pred(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """           
        Args:
            x:
                Input to the prediction network. Its shape is (T, N, B).
        Returns:
            pred_next:
                The predicted next chunk features. Its shape is (T, N, E).
        """
        res = x
        for i in range(len(self.pred_convs)):
            x = res
            x = self.pred_convs[i][1](x)  # (T, N, B)
            x = self.pred_norms[i][1](x)
            x = torch.nn.functional.gelu(x)
            res = res + x
        return res  # (T, N, B)
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool = True,
        encoder_out_post: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, E).
          encoder_out_lens:
            Lengths of the encoder output. Its shape is (N,).
          decoder_out:
            Output from the decoder. Maybe post-processed by RNN. Its shape is (N, T, s_range, D).
          project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
          encoder_out_post:
            Output from the encoder. Maybe post-processed by RNN. Its shape is (N, T, s_range, E).
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
            if encoder_out_post is None:
                encoder_out_post = encoder_out
            input_denom = self.encoder_proj(encoder_out_post[:, :, 0:1, :])  # (N, T, 1, B)
            input_numer = input_denom + self.decoder_proj(decoder_out)   # (N, T, s_range, B)

            # There are two options of implementation
            # 1. predict the next chunk from the current chunk (no masking)
            # 2. predict the next chunk with the zero-masking on the next chunk values
            # Here, we implement the first option for simplicity
            # The second option is more complicated because we need to
            # feedforward encoder_pred on the masked input for the next chunk
            # which is computationally heavy.
            N, T, s_range, B = input_numer.shape
            E = encoder_out.shape[-1]
            input_denom = input_denom.permute(1, 0, 2, 3).reshape(T, N*1, B)
            input_numer = input_numer.permute(1, 0, 2, 3).reshape(T, N*s_range, B)

            # roll the target encoder features to the left by one chunk
            encoder_target = torch.roll(encoder_out[:,:,0:1], shifts=-chunk_size, dims=1)  # (N, T, 1, E)
            encoder_target[:, -chunk_size:, :, :] = 0

            # Feed-forward prediction network to obtain the next-chunk context
            pred_context_denom = self.pred(input_denom)  # (T, N*1, B)
            pred_context_numer = self.pred(input_numer)  # (T, N*s_range, B)
            pred_context_denom = pred_context_denom.reshape(T, N, 1, -1).permute(1, 0, 2, 3)  # (N, T, 1, B)
            pred_context_numer = pred_context_numer.reshape(T, N, s_range, -1).permute(1, 0, 2, 3)  # (N, T, s_range, B)

            if self.pred_flow_depth == 0:
                # Predicting mean of Gaussian
                pred_next_denom = self.pred_out(pred_context_denom)  # (N, T, 1, E)
                pred_next_numer = self.pred_out(pred_context_numer)  # (N, T, s_range, E)
                # print('pred_next_denom norm', torch.norm(pred_next_denom[0,:5,0], p=2, dim=-1))
                # print('pred_next_numer norm', torch.norm(pred_next_numer[0,:5,0], p=2, dim=-1))
                if encoder_target.get_device() == 0:
                    print('pred_next_denom value', pred_next_denom[0,:5,0,0:5])
                    print('encoder_target norm', torch.norm(encoder_target[0,:5,0], p=2, dim=-1))
                    print('pred_next_denom norm', torch.norm(pred_next_denom[0,:5,0], p=2, dim=-1))
                    print('pred_next_denom diff norm', torch.norm(pred_next_denom[0,:5,0] - encoder_target[0,:5,0], p=2, dim=-1))
                    # print('encoder_target value', encoder_target[0,:5,0,0:5])
                    # print('encoder_target', encoder_target[0,:10,0])

                # calculate L2 distance for each position (t,u) in the rnnt grid
                # L2 distance is divided by bottleneck_dim
                logp_denom = -0.5 * torch.sum((pred_next_denom - encoder_target)**2, dim=-1)  # (N, T, 1)
                logp_numer = -0.5 * torch.sum((pred_next_numer - encoder_target)**2, dim=-1)  # (N, T, s_range)
            else:
                # Predicting arbitrary PDF with normalizing flow
                h_denom = encoder_target.reshape(N*T*1, E)
                h_numer = encoder_target.tile(1,1,s_range,1).reshape(N*T*s_range,E)
                c_denom = pred_context_denom.reshape(N*T*1, -1)  # (N*T*1, B)
                c_numer = pred_context_numer.reshape(N*T*s_range, -1)  # (N*T*s_range, B)
                z_denom, logabsdet_denom = self.flow.forward_and_log_det(h_denom, c_denom)
                z_numer, logabsdet_numer = self.flow.forward_and_log_det(h_numer, c_numer)
                z_denom = z_denom.reshape(N, T, 1, E)
                z_numer = z_numer.reshape(N, T, s_range, E)
                logabsdet_denom = logabsdet_denom.reshape(N, T, 1)
                logabsdet_numer = logabsdet_numer.reshape(N, T, s_range)
                logpz_denom = -0.5 * torch.sum(z_denom ** 2, dim=-1)  # (N, T, 1)
                logpz_numer = -0.5 * torch.sum(z_numer ** 2, dim=-1)  # (N, T, s_range)
                logp_denom = logpz_denom + logabsdet_denom  # (N, T, s_range)
                logp_numer = logpz_numer + logabsdet_numer  # (N, T, s_range)


            # Divide logp values by encoder_dim
            logp_denom = logp_denom
            logp_numer = logp_numer        
            # At chunk endpoint, the logp_ratio value equals to the
            # sum of the logp_ratio values of the next chunk
            # Note that except t % C == -1, the logp_ratio value is zero
            # because information gain occurs only at the end of every chunk
            ratio = logp_numer - logp_denom  # (N, T, s_range)
            logp_ratio = torch.zeros_like(ratio)
            if T % chunk_size != 0:
                # Prune the last chunk_size frames out before reshape & summing
                ratio = ratio[:, :-(T%chunk_size)]  # (N, [T], s_range)
            ratio = ratio.reshape(N, -1, chunk_size, s_range)  # (N, [T//C], C, s_range)
            ratio = torch.sum(ratio, dim=2)  # (N, [T//C], s_range)
            logp_ratio[:, chunk_size-1::chunk_size, :] = ratio

            # scale and clamp the log-prob. ratio
            if logp_ratio.get_device() == 0:
                print('logp_ratio before scale-clamp', logp_ratio[0,chunk_size-1:chunk_size*5:chunk_size,0])
            logp_ratio = logp_ratio * self.pred_logp_scale
            if self.pred_logp_ratio_clamp > 0:
                logp_ratio = torch.clamp(logp_ratio,
                                        min=-self.pred_logp_ratio_clamp,
                                        max=self.pred_logp_ratio_clamp)
            if logp_ratio.get_device() == 0:
                print('logp_ratio after scale-clamp', logp_ratio[0,chunk_size-1:chunk_size*5:chunk_size,0])
            
            # calculate loss functions for training encoder_prod
            # DEPRECATED: mean over valid time frames
            # NEW: mean over s_range, sum over valid time frames   
            #          
            # mask out except t=1:(T_i - C) (T_i: length of the encoder output for i-th sample)
            # not that the last chunk_size frames are not valid prediction,
            # because we don't have the target encoder features for them.
            target_mask = (torch.arange(encoder_out.shape[1], device=encoder_out.device) \
                          .expand(encoder_out.shape[0], encoder_out.shape[1]) \
                          < encoder_out_lens.unsqueeze(1) - chunk_size) \
                          .to(encoder_out.dtype)  # (N, T)
            logp_denom = logp_denom * target_mask.unsqueeze(-1)  # (N, T, 1)
            logp_numer = logp_numer * target_mask.unsqueeze(-1)  # (N, T, s_range)
            loss_denom = -torch.sum(torch.mean(logp_denom, dim=-1)) / E  # scalar
            loss_numer = -torch.sum(torch.mean(logp_numer, dim=-1)) / E  # scalar
            # NOTE: loss_denom should be at least 2x smaller than l2 loss before.
        else:
            assert False, "not implemented"

        return logp_ratio, loss_denom, loss_numer
