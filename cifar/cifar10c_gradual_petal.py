import logging

import torch
import torch.optim as optim

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model, load_model_bayes

from conf import cfg, load_cfg_fom_args

from tqdm import tqdm
import os
logger = logging.getLogger(__name__)

import petal

all_metrics = True

if all_metrics:
    from robustbench.utils import clean_metrics as metrics
else:
    from robustbench.utils import clean_accuracy as accuracy

def evaluate(description):
    load_cfg_fom_args(description)
    # configure model
    base_model, base_mean_model, base_cov_model = load_model_bayes(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
    base_model, base_mean_model, base_cov_model = base_model.cuda(), base_mean_model.cuda(), base_cov_model.cuda()

    if cfg.MODEL.ADAPTATION == "petalfim":
        logger.info("test-time adaptation: PETAL (FIM)")
        model= setup_petalfim(base_model, base_mean_model, base_cov_model)
    if cfg.MODEL.ADAPTATION == "petalsres":
        logger.info("test-time adaptation: PETAL (SRes)")
        model= setup_petalsres(base_model, base_mean_model, base_cov_model)
    # evaluate on each severity and type of corruption in turn
    prev_ct = "x0"
    err_dict = {}
    bri_dict = {}
    nll_dict = {}
    for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
        # continual adaptation for all corruption 
        if i_c == 0:
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
            severities = [5,4,3,2,1] 
            #To simulate the large gap between the source and first target domain, we start from severity 5.
        else:
            severities = [1,2,3,4,5,4,3,2,1]
            logger.info("not resetting model")
        for severity in severities:     
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            if all_metrics:
                acc, bri, nll = metrics(model, x_test, y_test, cfg.TEST.BATCH_SIZE, num_classes=10)
                err = 1. - acc
                logger.info(f"[{corruption_type}{severity}] % error : {err:.2%}, brier : {bri:.4}, nll : {nll:.4}")
                err_dict[corruption_type+str(severity)] = err
                bri_dict[corruption_type+str(severity)] = bri
                nll_dict[corruption_type+str(severity)] = nll
            else:
                acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
                err = 1. - acc
                logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
    with open(os.path.join(cfg.SAVE_DIR, "results_" + cfg.LOG_DEST),"w") as f:
        f.write("Error\n")
        for corr in err_dict:
            f.write(corr+","+str("{:.2%}".format(err_dict[corr]))+"\n")
        f.write("\nBrier\n")
        for corr in bri_dict:
            f.write(corr+","+str("{:.4}".format(bri_dict[corr]))+"\n")
        f.write("\nNLL\n")
        for corr in nll_dict:
            f.write(corr+","+str("{:.4}".format(nll_dict[corr]))+"\n")


def setup_petalfim(model, mean_model, cov_model):
    """Set up PETAL (FIM) adaptation.
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then adapt the model.
    """
    model = petal.configure_model(model)
    mean_model = petal.configure_model(mean_model)
    cov_model = petal.configure_model(cov_model)
    params, param_names = petal.collect_params(model)
    optimizer = setup_optimizer(params)
    petalfim_model = petal.PETALFim(model, mean_model, cov_model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           ap=cfg.OPTIM.AP, 
                           spw=cfg.OPTIM.SPW, 
                           perc=cfg.OPTIM.PERC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return petalfim_model

def setup_petalsres(model, mean_model, cov_model):
    """Set up PETAL (SRes) adaptation.
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then adapt the model.
    """
    model = petal.configure_model(model)
    mean_model = petal.configure_model(mean_model)
    cov_model = petal.configure_model(cov_model)
    params, param_names = petal.collect_params(model)
    optimizer = setup_optimizer(params)
    petalsres_model = petal.PETALSRes(model, mean_model, cov_model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           ap=cfg.OPTIM.AP, 
                           spw=cfg.OPTIM.SPW)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return petalsres_model


def setup_optimizer(params):
    """Set up optimizer for PETAL adaptation.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')
