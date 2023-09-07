dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" ./$dir/train.py \
  --world-size 6 \
  --start-epoch 26 \
  --num-epochs 70 \
  --causal 1 \
  --full-libri 1 \
  --use-fp16 1 \
  --exp-dir $dir/exp-chunk8newdec-from25-pred-d-lyr4-flow-dep4 \
  --chunk-size "8" \
  --rnnt-type "regular" \
  --train-in-eval-mode 1 \
  --use-encoder-pred 1 \
  --encoder-pred-detach 0 \
  --encoder-pred-bottleneck-dim 384 \
  --encoder-pred-kernel-size 17 \
  --encoder-pred-num-layers 2 \
  --encoder-pred-l2-norm-loss 0 \
  --encoder-pred-loss-scale 1.0 \
  --encoder-pred-l2-to-logp Gaussian \
  --encoder-pred-logp-scale 0 \
  --encoder-pred-logp-ratio-clamp 0.0 \
  --encoder-pred-enc-in-rnn 0 \
  --encoder-pred-dec-in-rnn 1 \
  --encoder-pred-dec-in-raw 1 \
  --encoder-pred-flow-depth 4 \
  --encoder-pred-flow-num-blocks 2 \
  --encoder-pred-flow-hidden-dim 512 \
  --max-duration 600 \
  --master-port 12345
  
#  --encoder-pred-
#  --exp-dir $dir/exp \
#  --max-duration 750 \
#  --base-lr 0.045 \
