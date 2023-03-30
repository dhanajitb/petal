for i in {0..9}
do
    CUDA_VISIBLE_DEVICES=0 python -u cifar10c_gradual_petal.py --cfg cfgs/10orders/petalfim/petalfim$i.yaml
done



