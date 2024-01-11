# -*- coding = utf-8 -*-
# @File : main.py
# @Software : PyCharm
import functools
import logging

import torch
from torch.utils.data import DataLoader
from transformers import logging as tflogging

from METS.METS import METS
from train import ssl_train
from utils.dataset import SSLECGTextDataset, ZeroShotTestECGTextDataset
from utils.utils import get_smallest_loss_model_path, make_log
from zero_shot_classification import zero_shot_classification

# close BERT pretrain file loaded warnings
tflogging.set_verbosity_error()

if __name__ == "__main__":
    make_log()

    num_samples = 100
    ecg_length = 1000
    train_dataset = SSLECGTextDataset(num_samples, ecg_length)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = SSLECGTextDataset(num_samples, ecg_length)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    test_dataset = ZeroShotTestECGTextDataset(num_samples, ecg_length)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    model = METS(stage="train")
    loss_fn = functools.partial(model.contrastive_loss, tau=0.07)
    optimizer = torch.optim.Adam(model.ecg_encoder.parameters(), lr=0.001)
    # metrics_dict = {"acc": Accuracy(task="multiclass")}

    train_dfhistory = ssl_train(model,
                                optimizer,
                                loss_fn,
                                metrics_dict=None,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                epochs=10,
                                patience=0.05,
                                monitor="val_loss",
                                mode="min")
    logging.info("\n" + train_dfhistory.to_string())

    ckpt_path = get_smallest_loss_model_path("./checkpoint")
    test_dfhistory = zero_shot_classification(model, ckpt_path, test_dataset, test_dataloader)
    logging.info("\n" + test_dfhistory.to_string())
