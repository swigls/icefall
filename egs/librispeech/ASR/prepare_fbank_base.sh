#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LibriSpeech
#      You can find BOOKS.TXT, test-clean, train-clean-360, etc, inside it.
#      You can download them from https://www.openslr.org/12
#
#  - $dl_dir/lm
#      This directory contains the following files downloaded from
#       http://www.openslr.org/resources/11
#
#        - 3-gram.pruned.1e-7.arpa.gz
#        - 3-gram.pruned.1e-7.arpa
#        - 4-gram.arpa.gz
#        - 4-gram.arpa
#        - librispeech-vocab.txt
#        - librispeech-lexicon.txt
#        - librispeech-lm-norm.txt.gz
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

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for librispeech"
  mkdir -p data/fbank_base
  if [ ! -e data/fbank_base/.librispeech.done ]; then
    ./local/compute_fbank_base_librispeech.py
    touch data/fbank_base/.librispeech.done
  fi

  if [ ! -f data/fbank_base/librispeech_cuts_train-all-shuf.jsonl.gz ]; then
    cat <(gunzip -c data/fbank_base/librispeech_cuts_train-clean-100.jsonl.gz) \
      <(gunzip -c data/fbank_base/librispeech_cuts_train-clean-360.jsonl.gz) \
      <(gunzip -c data/fbank_base/librispeech_cuts_train-other-500.jsonl.gz) | \
      shuf | gzip -c > data/fbank_base/librispeech_cuts_train-all-shuf.jsonl.gz
  fi

  if [ ! -e data/fbank_base/.librispeech-validated.done ]; then
    log "Validating data/fbank_base for LibriSpeech"
    parts=(
      train-clean-100
      train-clean-360
      train-other-500
      test-clean
      test-other
      dev-clean
      dev-other
    )
    for part in ${parts[@]}; do
      python3 ./local/validate_manifest.py \
        data/fbank_base/librispeech_cuts_${part}.jsonl.gz
    done
    touch data/fbank_base/.librispeech-validated.done
  fi
fi

