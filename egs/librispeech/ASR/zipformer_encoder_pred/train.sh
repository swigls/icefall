dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" ./$dir/train.py \
  --world-size 6 \
  --start-epoch 71 \
  --num-epochs 100 \
  --causal 1 \
  --full-libri 1 \
  --use-fp16 1 \
  --exp-dir $dir/exp-small-from50-d-nl1-ls0.01-decrawrnn-flow-dep2-dim256-from70-lps0.005-sls0.1-pr20 \
  --chunk-size "8" \
  --rnnt-type "regular" \
  --prune-range 20 \
  --simple-loss-scale 0.1 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,768,768,768,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --train-in-eval-mode 1 \
  --use-encoder-pred 1 \
  --encoder-pred-detach 1 \
  --encoder-pred-bottleneck-dim 256 \
  --encoder-pred-kernel-size 17 \
  --encoder-pred-num-layers 1 \
  --encoder-pred-loss-scale 0.01 \
  --encoder-pred-logp-scale 0.005 \
  --encoder-pred-logp-ratio-clamp 0.0 \
  --encoder-pred-enc-in-rnn 0 \
  --encoder-pred-dec-in-rnn 1 \
  --encoder-pred-dec-in-raw 1 \
  --encoder-pred-flow-depth 2 \
  --encoder-pred-flow-num-blocks 2 \
  --encoder-pred-flow-hidden-dim 256 \
  --max-duration 350 \
  --master-port 12345

# Gaussian
# logp-scale: 512 / variance  (e.g., if stddev=1/20, variance=1/400, then logp-scale=512*400=204800)
# logp-scale: 512 / variance  (e.g., if stddev=1/10, variance=1/100, then logp-scale=512*100=51200)
# logp-scale: 512 / variance  (e.g., if stddev=1, variance=1, then logp-scale=512=512)
# logp-scale: 512 / variance  (e.g., if stddev=2, variance=4, then logp-scale=512/4=128)
# logp-scale: 512 / variance  (e.g., if stddev=4, variance=16, then logp-scale=512/16=32)
# logp-scale: 512 / variance  (e.g., if stddev=8, variance=64, then logp-scale=512/64=8)
# logp-scale: 512 / variance  (e.g., if stddev=16, variance=256, then logp-scale=512/256=2)
  
#  --exp-dir $dir/exp \
#  --max-duration 750 \
#  --base-lr 0.045 \
