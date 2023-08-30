dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" ./$dir/train.py \
  --world-size 6 \
  --start-epoch 26 \
  --num-epochs 70 \
  --causal 1 \
  --full-libri 1 \
  --use-fp16 1 \
  --exp-dir $dir/exp-chunk8-dur600-eval-from25-pred-d-nl-lyr1-nopreact-encdecrawrnn \
  --chunk-size "8" \
  --rnnt-type "regular" \
  --train-in-eval-mode 1 \
  --use-encoder-pred 1 \
  --encoder-pred-detach 1 \
  --encoder-pred-bottleneck-dim 512 \
  --encoder-pred-kernel-size 17 \
  --encoder-pred-num-layers 1 \
  --encoder-pred-l2-norm-loss 1 \
  --encoder-pred-loss-scale 0.25 \
  --encoder-pred-l2-to-logp Gaussian \
  --encoder-pred-logp-scale 0 \
  --encoder-pred-logp-ratio-clamp 0.0 \
  --encoder-pred-enc-in-rnn 1 \
  --encoder-pred-dec-in-rnn 1 \
  --encoder-pred-dec-in-raw 1 \
  --max-duration 600 \
  --master-port 12345
  
#  --encoder-pred-
#  --exp-dir $dir/exp \
#  --max-duration 750 \
#  --base-lr 0.045 \
