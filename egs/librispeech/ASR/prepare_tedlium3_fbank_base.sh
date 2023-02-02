#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=3
stop_stage=3  # 100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/tedlium3
#      You can find data, doc, legacy, LM, etc, inside it.
#      You can download them from https://www.openslr.org/51
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bpe_xxx,
# data/lang_bpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  5000
  2000
  1000
  500
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for tedlium3"

  if [ ! -e data/fbank_base/.tedlium3.done ]; then
    mkdir -p data/fbank_base

    python3 ./local/compute_fbank_base_tedlium.py

    # gunzip -c data/fbank/tedlium_cuts_train.jsonl.gz | shuf | \
    # gzip -c > data/fbank_base/tedlium_cuts_train-shuf.jsonl.gz
    # mv data/fbank_base/tedlium_cuts_train-shuf.jsonl.gz \
    #    data/fbank_base/tedlium_cuts_train.jsonl.gz

    touch data/fbank_base/.tedlium3.done
  fi
fi
