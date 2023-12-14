import torch
import torch.nn as nn
import numpy as np
from typing import List

def init_weights(module: nn.Module):
    """Initializes a single module depending on its type"""
    for param in module.parameters():
        torch.nn.init.xavier_uniform_(param.data)

class Base_RNN(nn.Module):
    """ A 'standard' Recurrent Neural Network that continues an input sequence.

    Attributes:
        V (FloatTensor): weight matrix for concatenated input and hidden state
        b (FloatTensor): bias vector for first layer (W)
        W (FloatTensor): weight matrix for hidden state to output
        c (FloatTensor): bias vector for second layer (V)
    """
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
            hidden = torch.tanh(torch.mm(self.V, xh) + self.b)
            outputs = torch.hstack([outputs, torch.mm(self.W, hidden) + self.c])

        # continue sequence
        for _ in range(horizon):
            xh = torch.vstack([outputs[0][-1], hidden]).to(device)
            hidden = torch.tanh(torch.matmul(self.V, xh) + self.b)
            outputs = torch.hstack([outputs, torch.matmul(self.W, hidden) + self.c])
        return outputs.view(-1)[-horizon:] # only return last n elements -> n..horizon
    

class LSTM_RNN(nn.Module):
    """A Recurrent Neural Network that implements long short term memory (LSTM) to continue an input sequence.

    Attributes:
        f_Wx (FloatTensor): forget gate weight matrix that will be multiplied with the input 
        f_Wh (FloatTensor): forget gate weight matrix that will be multiplied with the (previous) hidden state
        f_b (FloatTensor): forget gate bias vector
        i_Wx (FloatTensor): input gate weight matrix that will be multiplied with the input 
        i_Wh (FloatTensor): input gate weight matrix that will be multiplied with the (previous) hidden state
        i_b (FloatTensor): input gate bias vector
        o_Wx (FloatTensor): output gate weight matrix that will be multiplied with the input 
        o_Wh (FloatTensor): output gate weight matrix that will be multiplied with the (previous) hidden state
        o_b (FloatTensor): output gate bias vector
        Ax (FloatTensor): weight matrix that is multiplied with the inout to generate c_hat 
        Ah (FloatTensor): weight matrix that is multiplied with the previous hidden state to generate c_hat 
        e (FloatTensor): bias vector for c_hat
        V (FloatTensor): weight matrix that is multiplied to create the output
        d (FloatTensor): weight matrix that is multiplied to create the output
    """
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
        self.V = nn.Parameter(torch.FloatTensor(output_len, hidden_len))
        self.D = nn.Parameter(torch.FloatTensor(hidden_len, output_len))
        init_weights(self)


    def forward(self, sequence: list[float], horizon=20, device=torch.device("cpu")) -> torch.FloatTensor:
        # I tried implementing cuda/mps devices in Base_RNN, but it was not faster - probably messed something up there.
        # But since I want to use the train method (and keep passing the device), I will keep device in this method signature for now. (Same thing for GRU and attention)
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
        output = torch.mm(torch.matmul(self.V, h), self.D)

        return output, h, c


class GRU_RNN(nn.Module):
    """ A Recurrent Neural Network that implements gated recurrent units and continues an input sequence.

    Attributes:
        u_Wx (FloatTensor): update gate weight matrix that will be multiplied with the input 
        u_Wh (FloatTensor): update gate weight matrix that will be multiplied with the (previous) hidden state
        u_b (FloatTensor): update gate bias vector
        r_Wx (FloatTensor): reset gate weight matrix that will be multiplied with the input
        r_Wh (FloatTensor): reset gate weight matrix that will be multiplied with the (previous) hidden state
        r_b (FloatTensor): reset gate bias vector
        Wx (FloatTensor): weight matrix that will be multiplied with the input, to generate h_hat
        Wh (FloatTensor): weight matrix that will be multiplied with the (previous) hidden state, to generate h_hat
        b (FloatTensor): bias vector used to generate h_hat
        V (FloatTensor): weight matrix that is multiplied with h_t to generate the output
        c (FloatTensor): bias vector that is added to generate the output
    """
    def __init__(self, input_len, hidden_len, output_len):
        super(GRU_RNN, self).__init__()
        self.hidden_len = hidden_len
        # gates
        self.u_Wx = nn.Parameter(torch.FloatTensor(hidden_len, input_len)) # in the whitepaper as U
        self.u_Wh = nn.Parameter(torch.FloatTensor(hidden_len, hidden_len)) # in the whitepaper as U_wave
        self.u_b = nn.Parameter(torch.FloatTensor(hidden_len, 1)) # in the whitepaper as u
        
        self.r_Wx = nn.Parameter(torch.FloatTensor(hidden_len, input_len)) # in the whitepaper as R
        self.r_Wh = nn.Parameter(torch.FloatTensor(hidden_len, hidden_len)) # in the whitepaper as R_wave
        self.r_b = nn.Parameter(torch.FloatTensor(hidden_len, 1)) # in the whitepaper as r

        # 'standard' rnn
        self.Wx = nn.Parameter(torch.FloatTensor(hidden_len, input_len)) # in the whitepaper as W
        self.Wh = nn.Parameter(torch.FloatTensor(hidden_len, hidden_len)) # in the whitepaper as W_wave
        self.b = nn.Parameter(torch.FloatTensor(hidden_len, 1)) # in the whitepaper as b
        self.V = nn.Parameter(torch.FloatTensor(output_len, hidden_len)) # in the whitepaper as V
        self.c = nn.Parameter(torch.FloatTensor(output_len, 1)) # in the whitepaper as c
        init_weights(self)
    
    def forward(self, input_sequence: list[torch.FloatTensor], horizon=20):
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
        update_gate = torch.sigmoid(torch.matmul(self.u_Wx, x) + torch.matmul(self.u_Wh, h) + self.u_b)
        reset_gate = torch.sigmoid(torch.matmul(self.r_Wx, x) + torch.matmul(self.r_Wh, h) + self.r_b)

        h_hat = torch.tanh(torch.matmul(self.Wx, x) + reset_gate * torch.matmul(self.Wh, h) + self.b)
        h = ((1 - update_gate) * h) + (update_gate * h_hat)
        output = torch.matmul(self.V, h) + self.c
        return output, h

def move_to_mac_gpu(model: nn.Module) -> (nn.Module, torch.device):
    """If exists, move the model to Apple GPU (sort of like cuda, but for apple devices)"""
    # for me, this (and also cuda) did not make the training faster.
    # I assume, it is because the .to(device) method is expensive, and in my implementation this method is called dozens of times during the forward pass. 
    # However, I did not have the time to dive deeper and try to solve this issue.
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print(type(mps_device))
        print(f"Moved {model.__class__.__name__} to MPS Device")
        return model.to(mps_device), mps_device
    else:
        print("MPS device not found.")
        return model, None