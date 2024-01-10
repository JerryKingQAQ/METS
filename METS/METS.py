# -*- coding = utf-8 -*-
# @File : METS.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

from METS.ECG_encoder.resnet1d import resnet18_1d


class FrozenLanguageModel(nn.Module):
    def __init__(self):
        super(FrozenLanguageModel, self).__init__()
        self.language_model = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        for param in self.language_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_representation = outputs.last_hidden_state[:, 0, :]
        return sentence_representation


class METS(nn.Module):
    def __init__(self, stage="train"):
        super(METS, self).__init__()
        self.text_encoder = FrozenLanguageModel()
        self.embedding_dim = self.text_encoder.language_model.config.hidden_size
        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.ecg_encoder = resnet18_1d(in_channels=16, projection_size=self.embedding_dim)
        self.class_text_representation = None
        self.stage = stage

    def ssl_process_text(self, text_data):
        ssl_text_prompt = "The report of the ECG is that {text}"
        prompt_list = [ssl_text_prompt.replace("{text}", report) for report in text_data]
        tokens = self.tokenizer(prompt_list, padding=True, truncation=True, return_tensors='pt', max_length=100)
        return tokens

    def zero_shot_precess_text(self, categories):
        zero_shot_text_prompt = "The ECG of {label}, a type of diagnostic."
        prompt_list = [zero_shot_text_prompt.replace("{label}", label) for label in categories]
        tokens = self.tokenizer(prompt_list, padding=True, truncation=True, return_tensors='pt', max_length=100)
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
        text_representation = self.text_encoder(input_ids, attention_mask)
        self.class_text_representation = {
            label: feature for label, feature in zip(categories, text_representation)
        }

    def forward(self, ecg_data, text_data):
        if self.stage == 'train':
            tokens = self.ssl_process_text(text_data)
            input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
            text_representation = self.text_encoder(input_ids, attention_mask)
            ecg_representation = self.ecg_encoder(ecg_data)
            return ecg_representation, text_representation

        elif self.stage == 'test':
            ecg_representation = self.ecg_encoder(ecg_data)
            return ecg_representation

    def contrastive_loss(self, ecg_representation, text_representation, tau=0.07):
        positive_similarity = F.cosine_similarity(ecg_representation, text_representation, dim=-1) / tau
        negative_similarity = F.cosine_similarity(
            ecg_representation.unsqueeze(1).repeat(1, ecg_representation.size(0), 1),
            text_representation.unsqueeze(0).repeat(ecg_representation.size(0), 1, 1),
            dim=-1
        ) / tau
        negative_similarity.fill_diagonal_(-float('inf'))  # 将对角线设为负无穷，以忽略正样本对

        loss_ecg_to_text = -torch.log(
            torch.exp(positive_similarity) /
            torch.sum(torch.exp(negative_similarity), dim=-1)
        ).mean()
        loss_text_to_ecg = -torch.log(
            torch.exp(positive_similarity) /
            torch.sum(torch.exp(negative_similarity), dim=0)
        ).mean()

        # TTODO 归一化等改进
        loss = (loss_ecg_to_text + loss_text_to_ecg) / 2
        return loss
