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
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          joint_input:
            Intermediate output from the joiner. Its shape is (N, T, U, J)
          x_target:
            A 3-D tensor of shape (N, T, D').
        Returns:
          Return a tensor of shape (N, T, U).
        """
        assert joint_input.ndim in (4,)
        assert x_target.ndim in (3,)
        assert x_target.size(-1) == 320
        device = joint_input.device
        N, T, U, _ = joint_input.shape

        # model estimates the centroid index of x_target
        logits = self.output_linear(joint_input)  # [N, T_h, U, ncentroids]
        logits = logits.permute(0, 3, 1, 2)  # [N, ncentroids, T_h, U]

        # Tile x_target to match shape
        x_target = torch.tile(
            x_target.unsqueeze(2), (1, 1, U, 1)
        )  # [N, T_h, U, 4D]
        # get logp by K-means index cross-entropy
        x_target_agg = x_target.view(-1, x_target.size(3))  # [N * T_h * U, 4D]
        if self.index_gpu is None:
            res = faiss.StandardGpuResources()
            self.index_gpu = faiss.index_cpu_to_gpu(
                res, device.index, self.index
            )
        _, centroid_index = self.index_gpu.search(
            x_target_agg,
            1
        )  # [N * T_h * U]
        centroid_index = centroid_index.view(N, T, U)  # [N, T_h, U]
        logp = -self.loss_ce(logits, centroid_index)  # [N, T_h, U]
        return logp
