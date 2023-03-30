for i in {0..9}
do
    CUDA_VISIBLE_DEVICES=1 python -u imagenetc_petal.py --cfg cfgs/10orders/petalfim/petalfim$i.yaml
done