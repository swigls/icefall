#!/usr/bin/env python3
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


"""
This file computes encodec embedding (ecemb) features of the LibriSpeech dataset.
It looks for manifests in the directory data/manifests.

The generated ecemb features are saved in data/ecemb.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
import k2
from filter_cuts import filter_cuts
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor

from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.features.kaldi.layers import Wav2LogFilterBank
from lhotse.utils import Seconds, asdict_nonull, compute_num_frames
from typing import Any, Dict, Union, Sequence, List
from dataclasses import asdict, dataclass
import torchaudio
import numpy as np

from encodec.encodec.model import EncodecModel
# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class Wav2EcEmb(torch.nn.Module):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their log-Mel filter bank energies (also known as "fbank").
    Example::
        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2LogFilterBank()
        >>> t(x).shape
        torch.Size([1, 100, 80])
    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_filters)``.
    """

    @torch.no_grad()
    def __init__(
        self,
        frame_shift: Seconds = 0.04/3,
        feature_dim: int = 128,
        target_bandwidth: float = 6.0,  #kbps
        sampling_rate: int = 24000,
    ):
        super().__init__()
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(target_bandwidth)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, t) 16000 Hz
        # encoded_frames: list, but length 1 (whole utterance)
        encoded_frames = self.model.encode(x.unsqueeze(1))
        assert len(encoded_frames) == 1, len(encoded_frames)
        codes, scale = encoded_frames[0]
        # codes: (B, 32, T)
        assert scale is None
        codes = codes.transpose(0, 1)  # (32, B, T)
        emb = self.model.quantizer.decode(codes)  # (B, 128, T)
        emb = emb.transpose(1, 2)  # (B, T, 128)
        return emb

@dataclass
class EcEmbConfig:
    frame_shift: float = 0.04 / 3
    feature_dim: int = 128
    target_bandwidth: float = 6.0  # kbps
    sampling_rate: int = 24000  # Hz
    device: str = "cpu"
    def __init__(self, target_bandwidth=6.0, device='cpu'):
        assert target_bandwidth in [1.5, 3.0, 6.0, 12.0, 24.0]
        self.target_bandwidth = target_bandwidth
        self.device = device
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EcEmbConfig":
        return EcEmbConfig(**data)

@register_extractor
class EcEmb(FeatureExtractor):
    name = 'ecemb'
    config_type = EcEmbConfig
    def __init__(self, config: Optional[EcEmbConfig] = None) -> None:
        super().__init__(config=config)
        config_dict = self.config.to_dict()
        config_dict.pop('device')
        self.extractor = Wav2EcEmb(**config_dict).to(self.device).eval()
        #self.extractor = Wav2LogFilterBank(sampling_rate=24000, frame_shift=0.04/3).to(self.device).eval()

    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
        lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
        return lhotse.features.kaldi.extractors._extract_batch(
            self.extractor,
            samples,
            sampling_rate,
            frame_shift=self.frame_shift,
            lengths=lengths,
            device=self.device,
        )

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:
        assert sampling_rate == self.config.sampling_rate, (
            f"Fbank was instantiated for sampling_rate "
            f"{self.config.sampling_rate}, but "
            f"sampling_rate={sampling_rate} was passed to extract(). "
            "Note you can use CutSet/RecordingSet.resample() to change the audio sampling rate."
        )

        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        # samples: (B, t)
        feats = self.extractor(samples.to(self.device))[0]
        # feats: (B, T, D)
        expected_num_frames = compute_num_frames(
            duration=samples.size(1) / self.config.sampling_rate,
            frame_shift=self.frame_shift,
            sampling_rate=self.config.sampling_rate,
        )
        feats = feats[:expected_num_frames]
        #assert False, (feats.shape, expected_num_frames)

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.feature_dim

    @staticmethod
    def mix(
        features_a: np.ndarray, features_b: np.ndarray, energy_scaling_factor_b: float
    ) -> np.ndarray:
        assert False  # not implemented
        return np.log(
            np.maximum(
                # protection against log(0); max with EPSILON is adequate since these are energies (always >= 0)
                EPSILON,
                np.exp(features_a) + energy_scaling_factor_b * np.exp(features_b),
            )
        )

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        assert False  # not implemented
        return float(np.sum(np.exp(features)))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to the bpe.model. If not None, we will remove short and
        long utterances before extracting features""",
    )

    parser.add_argument(
        "--target-bandwidth",
        type=float,
        default=6.0,
        help="""Target bandwidth in [1.5, 3.0, 6.0, 12.0, 24.0] kHz.""",
    )

    return parser.parse_args()


def compute_ecemb_tedlium(bpe_model: Optional[str] = None,
    target_bandwidth: float = 6.0):

    src_dir = Path("data/manifests")
    output_dir = Path(f"data/ecemb_{target_bandwidth}")
    #num_jobs = min(15, os.cpu_count())
    #num_jobs = min(15, os.cpu_count(), torch.cuda.device_count())
    num_jobs = 1
    #num_mel_bins = 80

    dataset_parts = (
        #"train",
        "dev",
        "test",
    )
    prefix = "tedlium"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    ## swigls: Fbank is substituted by Encodec 
    #extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))
    #extractor = EcEmb(EcEmbConfig(target_bandwidth=target_bandwidth, device='cuda:0'))
    extractor = EcEmb(EcEmbConfig(target_bandwidth=target_bandwidth, device='cpu'))

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
            if (output_dir / cuts_filename).is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )
            if bpe_model:
                cut_set = filter_cuts(cut_set, sp)

            if "train" in partition:
                cut_set = (
                    cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)
                )
            ## swigls: added for 24 kHz encoded model
            cut_set = cut_set.resample(24000)

            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
            cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_ecemb_tedlium(bpe_model=args.bpe_model, target_bandwidth=args.target_bandwidth)
