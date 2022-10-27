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
from scaling import ScaledLinear

# from icefall.utils import is_jit_tracing

from scaling import DoubleSwish
from typing import Optional
import faiss
from faiss.contrib.torch_utils import swig_ptr_from_FloatTensor
import numpy as np


class Pronouncer(nn.Module):
    def __init__(
        self,
        joint_input_dim: int,
        kmeans_model: str,
    ):
        super().__init__()

        self.joint_input_dim = joint_input_dim

        centroids = np.load(kmeans_model)  # [ncentroids, acoustic_dim]
        self.index = faiss.IndexFlatL2(centroids.shape[1])
        self.index.add(centroids)
        self.index_gpu = None

        self.output_linear = ScaledLinear(joint_input_dim, centroids.shape[0])
        self.loss_ce = nn.CrossEntropyLoss(reduction='none')

    def forward(
        self,
        joint_input: torch.Tensor,
        x: torch.Tensor,
        x_lens: Optional[torch.Tensor] = None,
        h_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          joint_input:
            Intermediate output from the joiner. Its shape is (N, T_h, U, J)
          x:
            A 3-D tensor of shape (N, T, D).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          h_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in
            `joint_input` before padding.
        Returns:
          Return a tensor of shape (N, T, U).
        """
        assert joint_input.ndim in (4,)
        assert x.ndim in (3,)
        assert x.size(-1) == 80
        N, T, D = x.shape
        device = x.device
        _, T_h, U, _ = joint_input.shape
        # if not is_jit_tracing():
        #    assert encoder_out.ndim == decoder_out.ndim
        #    assert encoder_out.ndim in (2, 4)
        #    assert encoder_out.shape == decoder_out.shape

        # calculating log-prob of x
        # NOTE: encoder out frame h_t corresponsd to x_(4t:4t+8)
        #       (check the Conv2dSubsampling)
        #       so, x_(4t+9:4t+12) should be estimated from h_t = Enc(x(1:4t+8))
        frames_in_first_h = 9  # Frames of first embedding (in Conv2dSampling)
        x_target = x[:, frames_in_first_h:]  # [N, T-9, D]
        r = x_target.size(1) % 4  # Stride=4 (in Conv2dSampling)
        x_target = x_target[:, : x_target.size(1) - r]  # [N, T-9-r, D]
        x_target = x_target.view(N, -1, 4 * D)  # [N, T_t=(T-9+r)/4, 4D]
        T_t = x_target.size(1)
        assert T_t + 1 == T_h, "T_t=%d, T_h=%d" % (T_t, T_h)
        x_target = F.pad(x_target, (0, 0, 0, 1))  # [N, T_h, 4D]
        x_target = torch.tile(
            x_target.unsqueeze(2), (1, 1, U, 1)
        )  # [N, T_h, U, 4D]

        # model estimates the centroid index of x_target
        logits = self.output_linear(joint_input)  # [N, T_h, U, ncentroids]
        logits = logits.permute(0, 3, 1, 2)  # [N, ncentroids, T_h, U]

        # get logp by K-means index cross-entropy
        x_target_agg = x_target.view(-1, 4 * D)  # [N * T_h * U, 4D]
        if self.index_gpu is None:
            res = faiss.StandardGpuResources()
            self.index_gpu = faiss.index_cpu_to_gpu(
                res, device.index, self.index
            )
        print(type(x_target_agg))
        _, centroid_index = self.index_gpu.search(
            x_target_agg,
            1
        )  # [N * T_h * U]
        # centroid_index = torch.from_numpy(centroid_index).to(device)
        centroid_index = centroid_index.view(N, T_h, U)  # [N, T_h, U]
        logp = -self.loss_ce(logits, centroid_index)  # [N, T_h, U]

        # Last time index should be ignored, since no valid x_target left.
        # This is implemented by masking
        x_target_mask = torch.arange(
            T_h, device=joint_input.device, dtype=h_lens.dtype
        )[None, :] < (
            h_lens[:, None] - 1
        )  # [N, T_h]
        x_target_mask = x_target_mask.unsqueeze(2)  # [N, T_h, 1]
        logp = logp * x_target_mask  # [N, T_h, U]
        # print("x_target[0:2]", x_target[0:2])
        print("h_lens[0:2]", h_lens[0:2])
        print("logp[0:2]", logp[0:2])  # [T_h, U]
        # print("x_target_mask", x_target_mask)
        # print("x", x[0, :13, 0])
        return logp
