# -*- coding = utf-8 -*-
# @File : zero_shot_classification.py
# @Software : PyCharm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics import Precision, Accuracy, Recall, F1Score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classify(model, input):
    ecg_representation = model(input, None)
    class_text_rep_tensor = torch.stack(list(model.class_text_representation.values()))

    # 计算相似度
    similarities = [F.cosine_similarity(elem.unsqueeze(0), class_text_rep_tensor) for elem in ecg_representation]
    similarities = torch.stack(similarities).to(device)

    probabilities = F.softmax(similarities, dim=1).cpu().numpy()
    max_probability_class = np.argmax(probabilities, axis=1)
    # max_probabilities = np.max(probabilities, axis=1)

    max_probability_class = torch.tensor(max_probability_class).long()

    return max_probability_class, probabilities


def zero_shot_classification(model, ckpt_path, test_dataset, test_dataloader):
    model.load_state_dict(torch.load(ckpt_path))
    model.zero_shot_precess_text(test_dataset.categories)
    model.stage = "test"

    precision = Precision(task="multiclass", average='macro', num_classes=5)
    accuracy = Accuracy(task="multiclass", num_classes=5)
    recall = Recall(task="multiclass", average='macro', num_classes=5)
    f1 = F1Score(task="multiclass", num_classes=5)

    test_history = {}

    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        for i, (input, target) in enumerate(test_dataloader):
            max_probability_class, probabilities = classify(model, input)
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
