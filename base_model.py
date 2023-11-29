import torch
import torch.nn as nn
import numpy as np
from typing import List

def init_weights(module: nn.Module):
    """Initializes a single module depending on its type"""
    for param in module.parameters():
        torch.nn.init.xavier_uniform_(param.data)

class Base_RNN(nn.Module):

    def __init__(self, input_len, hidden_len, output_len):
        super(Base_RNN, self).__init__()
        self.hidden_len = hidden_len
        self.V = nn.Parameter(torch.FloatTensor(hidden_len, input_len + hidden_len))
        self.b = nn.Parameter(torch.FloatTensor(hidden_len, 1))
        self.W = nn.Parameter(torch.FloatTensor(output_len, hidden_len))
        self.c = nn.Parameter(torch.FloatTensor(output_len, 1))
        init_weights(self)
    
    def forward(self, input_sequence: list[torch.FloatTensor], horizon=20, device=torch.device("cpu")):
        hidden = torch.zeros((self.hidden_len,1)).to(device) # initialize hidden state with all zeros
        outputs = torch.FloatTensor(1,0).to(device)
        
        # start on existing data points
        for x in input_sequence:
            if not torch.is_tensor(x):
                x = torch.tensor(x).to(device)
            xh = torch.vstack([x, hidden]).to(device)
            hidden = torch.tanh(torch.matmul(self.V, xh) + self.b)
            outputs = torch.hstack([outputs, torch.matmul(self.W, hidden) + self.c])

        # continue sequence
        for _ in range(horizon):
            xh = torch.vstack([outputs[0][-1], hidden]).to(device)
            hidden = torch.tanh(torch.matmul(self.V, xh) + self.b)
            outputs = torch.hstack([outputs, torch.matmul(self.W, hidden) + self.c])
        return outputs.view(-1)[-horizon:] # only return last n elements -> n..horizon