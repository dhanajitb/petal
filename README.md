***
# A Probabilistic Framework for Lifelong Test-Time Adaptation
Test-time adaptation (TTA) is the problem of updating a pre-training source model at inference time given test input(s) from a different target domain. Most existing TTA approaches assume the setting in which the target domain is *stationary*, i.e., all the test inputs come from a single target domain. However, in many practical settings, the target domain distribution might exhibit a lifelong/continual shift over time. Moreover, existing TTA approaches also lack the ability to provide reliable uncertainty estimates, which is crucial when distribution shifts occur between the source and target domain. To address these issues, we present PETAL (Probabilistic lifElong Test-time Adaptation with seLf-training prior), which solves lifelong TTA using a probabilistic approach, and naturally results in (1) a student-teacher framework, where the teacher model is an exponential moving average of the student model, and (2) regularizing the model updates at inference time using the source model as a regularizer. To prevent model drift in the lifelong/continual TTA setting, we also propose a data-driven parameter restoration technique that contributes to reducing the error accumulation and maintaining the knowledge of recent domains by restoring only the irrelevant parameters.  In terms of predictive error rate as well as uncertainty based metrics such as Brier score and negative log-likelihood, our method achieves better results than the current state-of-the-art for online lifelong test-time adaptation across various benchmarks, such as CIFAR-10C, CIFAR-100C, ImageNetC, and ImageNet3DCC datasets.
***

## Setting up the environment
Create the following conda envrionment:
```bash
conda env create -f environment.yml
```
Then activate the environment:
```bash
conda activate petal
```

## CIFAR10/100
For CIFAR10-to-CIFAR10C/CIFAR100-to-CIFAR100C experiments, first change the directory using:
```bash
cd cifar
```

### CIFAR10-to-CIFAR10C Experiments
1. Download the pre-trained WideResNet-28 from RobustBench from the link (url is obtained from `robustbench/model_zoo/cifar10.py`):
https://drive.google.com/file/d/1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC/view
2. Rename the downloaded file "natural.pt.tar" to "Standard.pt" and move inside the directory `ckpt/cifar10/corruptions/`
3. Run `python train-swag-diagonal-cifar10.py` for further training on source domain training data. This will create the files Standard_cov.pt and Standard_swa.pt inside the directory `ckpt/cifar10/corruptions/`
4. Now, run the CIFAR10-to-CIFAR10C adaptation experiment for PETAL (FIM) by running:
```bash
bash run_cifar10_petalfim.sh
```
5. Run CIFAR10-to-CIFAR10C adaptation experiment for PETAL (SRes) by running:
```bash
bash run_cifar10_petalsres.sh
```

### CIFAR100-to-CIFAR100C Experiments
1. Run "python train-swag-diagonal-cifar100.py" for further training on source domain training data. This will create the files `Hendrycks2020AugMix_ResNeXt_cov.pt` and `Hendrycks2020AugMix_ResNeXt_swa.pt` inside the directory `ckpt/cifar100/corruptions/`
2. Now, run the CIFAR100-to-CIFAR100C adaptation experiment for PETAL (FIM) by running:
```bash
bash run_cifar100_petalfim.sh
```
3. Run CIFAR100-to-CIFAR100C adaptation experiment for PETAL (SRes) by running:
```bash
bash run_cifar100_petalsres.sh
```

### CIFAR10-to-CIFAR10C gradual corruption severity level change experiments
1. Download the pre-trained WideResNet-28 from RobustBench from the link (url is obtained from `robustbench/model_zoo/cifar10.py`):
https://drive.google.com/file/d/1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC/view
2. Rename the downloaded file `natural.pt.tar` to `Standard.pt` and move inside the directory `ckpt/cifar10/corruptions/`
3. Run `python train-swag-diagonal-cifar10.py` for further training on source domain training data. This will create the files `Standard_cov.pt` and `Standard_swa.pt` inside the directory `ckpt/cifar10/corruptions/`
4. Now, run CIFAR10-to-CIFAR10C gradual adaptation experiment for PETAL (FIM) by running:
```bash
bash run_cifar10_gradual.sh
```

## ImageNet
For ImageNet-to-ImageNetC experiments, first change the directory using: 
```bash
cd imagenet
```

### ImageNet-to-ImageNetC Experiments
1. Download the ImageNet train dataset from: [link](https://image-net.org/download.php). Upon extracting `ILSVRC2012_img_train.tar`, the `ILSVRC2012_img_train` directory will be extracted. Move `ILSVRC2012_img_train` directory inside the `data/` directory.
2. Run `python train-swag-diagonal-imagenet.py` for further training on source domain training data. This will create the files `Standard_R50_cov.pt` and `Standard_R50_swa.pt` inside the directory `ckpt/cifar10/corruptions/`
3. Download ImageNet-C from: [link](https://zenodo.org/record/2235448#.Yj2RO_co_mF). Keep the files inside the directory `data/ImageNet-C/`
4. Now, run ImageNet-to-ImageNetC adaptation experiment for PETAL (FIM) by running:
```bash
bash run_petalfim.sh
```
5. Run the ImageNet-to-ImageNetC adaptation experiment for PETAL (SRes) by running:
```bash
bash run_petalsres.sh
```
6. Run `python eval_corruptionwise.py` to get final averaged results.

## ImageNet
For ImageNet-to-ImageNet3DCC experiments, first change the directory using: 
```bash
cd imagenet
```

### ImageNet-to-ImageNet3DCC Experiments
1. Download the ImageNet train dataset from: [link](https://image-net.org/download.php). Upon extracting `ILSVRC2012_img_train.tar`, the `ILSVRC2012_img_train` directory will be extracted. Move `ILSVRC2012_img_train` directory inside the `data/` directory.
2. Run `python train-swag-diagonal-imagenet.py` for further training on source domain training data. This will create the files `Standard_R50_cov.pt` and `Standard_R50_swa.pt` inside the directory `ckpt/cifar10/corruptions/`
3. Download ImageNet-3DCC by running: `download_imagenet3d.sh` located inside the directory `data/ImageNet-3D/`. Ensure that the files are stored inside the directory `data/ImageNet-3DCC/`
4. Now, run ImageNet-to-ImageNet3DCC adaptation experiment for PETAL (FIM) by running:
```bash
bash run_petalfim_imagenet3dcc.sh
```
5. Run `python eval_corruptionwise_img3d.py` to get final averaged results.


### This codebase uses the following repositories:
+ CoTTA [link](https://github.com/qinenergy/cotta)
+ TENT [link](https://github.com/DequanWang/tent)
+ KATANA for augmentation. [link](https://github.com/giladcohen/KATANA) 
+ Robustbench [link](https://github.com/RobustBench/robustbench) 
+ 3DCommonCorruptions [link](https://github.com/EPFL-VILAB/3DCommonCorruptions) for download_imagenet3d.sh

## Citation
Please use the following to cite our work.
```bibtex
@inproceedings{brahma2023probabilistic,
  title={A Probabilistic Framework for Lifelong Test-Time Adaptation},
  author={Brahma, Dhanajit and Rai, Piyush},
  booktitle={Proceedings of Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
