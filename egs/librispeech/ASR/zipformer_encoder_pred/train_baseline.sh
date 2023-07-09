dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" ./$dir/train.py \
  --world-size 6 \
  --start-epoch 1 \
  --num-epochs 40 \
  --causal 1 \
  --full-libri 1 \
  --use-fp16 1 \
  --exp-dir $dir/exp-chunk8-dur600 \
  --chunk-size "8" \
  --train-in-eval-mode 0 \
  --use-encoder-pred 0 \
  --encoder-pred-detach 0 \
  --encoder-pred-bottleneck-dim 384 \
  --encoder-pred-kernel-size 17 \
  --encoder-pred-num-layers 2 \
  --encoder-pred-loss-scale 0.25 \
  --encoder-pred-l2-norm-loss 0 \
  --encoder-pred-l2-to-logp Gaussian \
  --encoder-pred-logp-scale 1.0 \
  --encoder-pred-logp-ratio-clamp 0.0 \
  --max-duration 600 \
  --master-port 12345
  
#  --exp-dir $dir/exp \
#  --max-duration 750 \
#  --base-lr 0.045 \
