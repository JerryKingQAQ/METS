# -*- coding = utf-8 -*-
# @File : LSTM.py
# @Software : PyCharm
import torch
from torch import nn
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, projection_size=768):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, projection_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.projection(hidden[-1])