# -*- coding = utf-8 -*-
# @File : zero_shot_classification.py
# @Software : PyCharm
import logging

import pandas as pd
import torch
from torchmetrics import Precision, Accuracy, Recall, F1Score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zero_shot_classification(model, ckpt_path, test_dataset, test_dataloader):
    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    model.zero_shot_precess_text(test_dataset.categories)
    model.stage = "test"

    precision = Precision(task="multiclass", num_classes=5)
    accuracy = Accuracy(task="multiclass", num_classes=5)
    recall = Recall(task="multiclass", num_classes=5)
    f1 = F1Score(task="multiclass", num_classes=5)

    test_history = {}
    logging.warning("=" * 25 + "Start Zero-Shot Testing" + "=" * 25)

    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        for i, (input, target) in enumerate(test_dataloader):
            max_probability_class = model(input, None)
            predictions.append(max_probability_class.flatten())
            targets.append(target.flatten())

        predictions = torch.cat(predictions).long()
        targets = torch.cat(targets).long()

        test_history["acc"] = accuracy(predictions, targets)
        test_history["pre"] = precision(predictions, targets)
        test_history["recall"] = recall(predictions, targets)
        test_history["f1"] = f1(predictions, targets)

    for key in test_history:
        test_history[key] = [test_history[key].item()]

    return pd.DataFrame(test_history, index=[0])
