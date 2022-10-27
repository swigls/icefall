#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang,
#                                                  Mingshuang Luo,)
#                                                  Zengwei Yao)
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
Usage:

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./lstm_transducer_stateless3_noprune_joint/cluster.py \
  --exp-dir lstm_transducer_stateless3_noprune_joint/exp \
"""

import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from decoder import Decoder
from joiner import Joiner
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from lstm import RNN
from model import Transducer
from optim import Eden, Eve
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    display_and_save_batch,
    setup_logger,
    str2bool,
)
import faiss
import os
import numpy as np

def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=int,
        default=12,
        help="Number of RNN encoder layers..",
    )

    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=512,
        help="Encoder output dimesion.",
    )

    parser.add_argument(
        "--rnn-hidden-size",
        type=int,
        default=1024,
        help="Hidden dim for LSTM layers.",
    )

    parser.add_argument(
        "--aux-layer-period",
        type=int,
        default=0,
        help="""Peroid of auxiliary layers used for randomly combined during training.
        If set to 0, will not use the random combiner (Default).
        You can set a positive integer to use the random combiner, e.g., 3.
        """,
    )

    parser.add_argument(
        "--grad-norm-threshold",
        type=float,
        default=25.0,
        help="""For each sequence element in batch, its gradient will be
        filtered out if the gradient norm is larger than
        `grad_norm_threshold * median`, where `median` is the median
        value of gradient norms of all elememts in batch.""",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=40,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="lstm_transducer_stateless3_noprune_joint/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--kmeans-model",
        type=str,
        default="lstm_transducer_stateless3_noprune_joint/exp/kmeans.pkl",
        help="""The K-means model file path.
        """,
    )

    parser.add_argument(
        "--ncentroids",
        type=int,
        default=500,
        help="""Number of K-means centroids
        """,
    )

    parser.add_argument(
        "--niter",
        type=int,
        default=25,
        help="""Number of K-means training iterations
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--initial-lr",
        type=float,
        default=0.003,
        help="""The initial learning rate. This value should not need to be
        changed.""",
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate decreases.
        We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=10,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network)"
        "part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=4000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=20,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=100,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - warm_step: The warm_step for Noam optimizer.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            "dim_feedforward": 2048,
            # parameters for decoder
            "decoder_dim": 512,
            # parameters for joiner
            "joiner_dim": 512,
            # parameters for Noam
            "model_warm_step": 3000,  # arg given to model, not for lrate
            "env_info": get_env_info(),
        }
    )

    return params


def run(args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))
    os.makedirs(args.exp_dir, exist_ok=True)

    fix_random_seed(params.seed)

    setup_logger(f"{params.exp_dir}/log/log-cluster")
    logging.info("Clustering started")

    logging.info(params)

    logging.info("About to create model")

    librispeech = LibriSpeechAsrDataModule(args)

    train_cuts = librispeech.train_clean_100_cuts()
    if params.full_libri:
        train_cuts += librispeech.train_clean_360_cuts()
        train_cuts += librispeech.train_other_500_cuts()

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        return 1.0 <= c.duration <= 20.0

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    train_dl = librispeech.train_dataloaders(
        train_cuts, sampler_state_dict=None
    )

    # NOTE: 80-dim. feature * 4 stride = 320-dim.
    # NOTE: 100,000,000 frames (w/ stride 4) amounts to approaximately
    #       1,111 h audio, which is larger than librispeech (960 h)
    input_tensor = np.zeros([100000000, 320], dtype=np.float32)
    i = 0
    for batch_idx, batch in enumerate(train_dl):
        x = batch["inputs"]  # [N, T, C]
        x_lens = batch["supervisions"]["num_frames"]  # [N]
        assert x.ndim == 3
        N = x.size(0)
        for n in range(N):
            # NOTE: The first encoder output h_0 is an output determined
            #   by x_0:9, so the next prediction should be imposed
            #   on x_9:12. Because stride=4 in Conv2dSubsampling,
            #   time-frame iteration should be given as below
            for t in range(9, int(x_lens[n].item()) - 3, 4):
                input_tensor[i, 0:80] = x[n, t + 0, :]
                input_tensor[i, 80:160] = x[n, t + 1, :]
                input_tensor[i, 160:240] = x[n, t + 2, :]
                input_tensor[i, 240:320] = x[n, t + 3, :]
                i += 1

        if (batch_idx + 1) % 100 == 0:
            logging.info(
                "%d frames (about %.2f h) are gathered."
                % (i, i * 0.01 * 1 / 3600)
            )
        if i > 100 * 10000:
            break

    input_tensor = input_tensor[:i]
    logging.info(
        "Total %d frames (about %.2f h) are gathered."
        % (i, i * 0.01 * 1 / 3600)
    )

    kmeans = faiss.Kmeans(
        320, args.ncentroids, niter=args.niter, verbose=True, gpu=True
    )

    logging.info(
        "Total %d frames (about %.2f h) are gathered."
        % (i, i * 0.01 * 1 / 3600)
    )

    logging.info("K-means learning started.")
    kmeans.train(input_tensor)
    logging.info("K-means learning finished.")

    np.save(args.kmeans_model, kmeans.centroids)
    logging.info("K-means model is saved in %s" % (args.kmeans_model))

    logging.info("Done!")


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    run(args=args)


if __name__ == "__main__":
    main()
