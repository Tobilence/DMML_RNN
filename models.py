import torch
import torch.nn as nn
import numpy as np
from typing import List

def init_weights(module: nn.Module):
    """Initializes a single module depending on its type"""
    for param in module.parameters():
        torch.nn.init.xavier_uniform_(param.data)

class Base_RNN(nn.Module):

    @property
    def name(self):
        return f"{self._name if self._name is not None else 'Base RNN'}"

    def __init__(self, input_len, hidden_len, output_len, name=None):
        super(Base_RNN, self).__init__()
        self._name = name
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
    

class LSTM_RNN(nn.Module):
    @property
    def name(self):
        return f"{self._name if self._name is not None else 'LSTM RNN'}"
        
    def __init__(self, input_len, hidden_len, output_len):
        super(LSTM_RNN, self).__init__()

        self.hidden_len = hidden_len

        # gates
        self.f_Wx = nn.Parameter(torch.FloatTensor(hidden_len, input_len))
        self.f_Wh = nn.Parameter(torch.FloatTensor(hidden_len, hidden_len))
        self.f_b = nn.Parameter(torch.FloatTensor(hidden_len, 1))

        self.i_Wx = nn.Parameter(torch.FloatTensor(hidden_len, input_len))
        self.i_Wh = nn.Parameter(torch.FloatTensor(hidden_len, hidden_len))
        self.i_b = nn.Parameter(torch.FloatTensor(hidden_len, 1))

        self.o_Wx = nn.Parameter(torch.FloatTensor(hidden_len, input_len))
        self.o_Wh = nn.Parameter(torch.FloatTensor(hidden_len, hidden_len))
        self.o_b = nn.Parameter(torch.FloatTensor(hidden_len, 1))
        
        # cell values
        self.Ax = nn.Parameter(torch.FloatTensor(hidden_len, input_len))
        self.Ah = nn.Parameter(torch.FloatTensor(hidden_len, hidden_len))
        self.e = nn.Parameter(torch.FloatTensor(hidden_len, 1))

        # normal values
        # self.Wx = nn.Parameter(torch.FloatTensor(hidden_len, input_len))
        # self.Wh = nn.Parameter(torch.FloatTensor(hidden_len, hidden_len))
        # self.b = nn.Parameter(torch.FloatTensor(hidden_len, 1))
        self.c = nn.Parameter(torch.FloatTensor(output_len, 1))
        self.V = nn.Parameter(torch.FloatTensor(output_len, hidden_len))
        self.TEST = nn.Parameter(torch.FloatTensor(hidden_len, output_len)) # ??? - is this correct?
        init_weights(self)


    def forward(self, sequence: list[float], horizon=20) -> torch.FloatTensor:
        hidden_state = torch.zeros(self.hidden_len)
        cell_state = torch.zeros(self.hidden_len)

        outputs = torch.FloatTensor([[]])
        for x in sequence:
            output, hidden_state, cell_state = self._calculate_forward(x, hidden_state, cell_state)
            outputs = torch.hstack([outputs, output])
        
        for _ in range(horizon):
            x = outputs[0][-1]
            output, hidden_state, cell_state = self._calculate_forward(x, hidden_state, cell_state)
            outputs = torch.hstack([outputs, output])
        
        return outputs.view(-1)[-horizon:]


    def _calculate_forward(self, x, h, c):
        x = torch.tensor([[x]], dtype=torch.float32)

        # calculate gates
        f = torch.sigmoid(torch.matmul(self.f_Wx, x) + torch.matmul(self.f_Wh, h) + self.f_b)
        i = torch.sigmoid(torch.matmul(self.i_Wx, x) + torch.matmul(self.i_Wh, h) + self.i_b)
        o = torch.sigmoid(torch.matmul(self.o_Wx, x) + torch.matmul(self.o_Wh, h) + self.o_b)

        # cell state
        c_hat = torch.tanh(torch.matmul(self.Ax, x) + torch.matmul(self.Ah, h) + self.e)
        c = f * c + i * c_hat

        # hidden state & output
        h = o * torch.tanh(c)
        output = torch.matmul(torch.matmul(self.V, h), self.TEST) # not sure about this line

        return output, h, c


class GRU_RNN(nn.Module):
    @property
    def name(self):
        return f"{self._name if self._name is not None else 'GRU RNN'}"

    def __init__(self, input_len, hidden_len, output_len):
        super(GRU_RNN, self).__init__()
        self.hidden_len = hidden_len
        # gates
        self.u_Wx = nn.Parameter(torch.FloatTensor(hidden_len, input_len)) # update gate weights that will be multiplied with input
        self.u_Wh = nn.Parameter(torch.FloatTensor(hidden_len, hidden_len)) # update gate weights that will be multiplied with hidden state of t-1
        self.u_b = nn.Parameter(torch.FloatTensor(hidden_len, 1)) # update gate bias
        
        self.r_Wx = nn.Parameter(torch.FloatTensor(hidden_len, input_len)) # reset gate weights that will be multiplied with input
        self.r_Wh = nn.Parameter(torch.FloatTensor(hidden_len, hidden_len)) # reset gate weights that will be multiplied with hidden state of t-1
        self.r_b = nn.Parameter(torch.FloatTensor(hidden_len, 1)) # reset gate bias

        # 'standard' rnn
        self.Wx = nn.Parameter(torch.FloatTensor(hidden_len, input_len)) # weight matrix for inputs
        self.Wh = nn.Parameter(torch.FloatTensor(hidden_len, hidden_len)) # weight matrix for hidden state
        self.b = nn.Parameter(torch.FloatTensor(hidden_len, 1))
        self.V = nn.Parameter(torch.FloatTensor(output_len, hidden_len))
        self.c = nn.Parameter(torch.FloatTensor(output_len, 1))
        init_weights(self)
    
    def forward(self, input_sequence: list[torch.FloatTensor], horizon=20, device=torch.device("cpu")):
        hidden = torch.zeros((self.hidden_len,1)) # initialize hidden state with all zeros
        outputs = torch.FloatTensor([])
        
        # start on existing data points
        for x in input_sequence:
            x = torch.tensor([[x]], dtype=torch.float32)
            output, hidden = self._calculate_forward(x, hidden)
            outputs = torch.hstack((outputs, output))

        # continue sequence
        for _ in range(horizon):
            x = outputs[0][-1].unsqueeze(dim=0).unsqueeze(dim=1)
            output, hidden = self._calculate_forward(x, hidden)
            outputs = torch.hstack((outputs, output))

        return outputs.view(-1)[-horizon:]

    def _calculate_forward(self, x, h):
        update_gate = torch.sigmoid(torch.matmul(self.u_Wx, x)+ torch.matmul(self.u_Wh, h) + self.u_b)
        reset_gate = torch.sigmoid(torch.matmul(self.r_Wx, x) + torch.matmul(self.r_Wh, h) + self.r_b)

        h_hat = torch.tanh(torch.matmul(self.Wx, x) + reset_gate * torch.matmul(self.Wh, h) + self.b)
        h = ((1 - update_gate) * h) + (update_gate * h_hat)
        output = torch.matmul(self.V, h) + self.c
        return output, h

class GRU_RNN_Torch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU_RNN_Torch, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0):
        out, hn = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def move_to_mac_gpu(model: nn.Module) -> (nn.Module, torch.device):
    """If exists, move the model to Apple GPU (sort of like cuda, but for apple devices)"""
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print(type(mps_device))
        print(f"Moved {model.__class__.__name__} to MPS Device")
        return model.to(mps_device), mps_device
    else:
        print("MPS device not found.")
        return model, None