dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

../../../icefall/rnn_lm/train.py \
    --start-epoch 0 \
    --world-size 6 \
    --num-epochs 100 \
    --exp-dir $dir/exp-rnnlm \
    --use-fp16 0 \
    --tie-weights 0 \
    --embedding-dim 800 \
    --hidden-dim 200 \
    --num-layers 2 \
    --batch-size 400

# ../../../icefall/transformer_lm/train.py \
#         --start-epoch 0 \
#         --world-size 6 \
#         --num-epochs 100 \
#         --exp-dir $dir/exp-transformerlm \
#         --use-fp16 0 \
#         --num-layers 12 \
#         --batch-size 400
