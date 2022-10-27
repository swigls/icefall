export CUDA_VISIBLE_DEVICES="0,1,2,3,4"

dir=$( dirname -- "$0"; )
./$dir/cluster.py \
  --exp-dir $dir/exp \
  --enable-musan 0  \
  --kmeans-model $dir/exp/kmeans_500.npy \
  --niter 50 \
  --ncentroids 500 \
