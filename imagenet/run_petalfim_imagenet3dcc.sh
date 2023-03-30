for i in {0..9}
do
    CUDA_VISIBLE_DEVICES=1 python -u imagenet3dcc_petal.py --cfg cfgs/10orders_imagenet3d/petalfim/petalfim$i.yaml
done