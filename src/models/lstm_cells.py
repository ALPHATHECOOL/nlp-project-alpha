

import torch
import torch.nn as nn

def _init_linear(layer, in_features, out_features):
    layer.weight = nn.Parameter(torch.Tensor(out_features, in_features))
    nn.init.xavier_uniform_(layer.weight)
    layer.bias = nn.Parameter(torch.zeros(out_features))
    return layer

class VanillaLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VanillaLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wxi = _init_linear(nn.Linear(input_size, hidden_size), input_size, hidden_size)
        self.Whi = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)

        self.Wxf = _init_linear(nn.Linear(input_size, hidden_size), input_size, hidden_size)
        self.Whf = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)

        self.Wxo = _init_linear(nn.Linear(input_size, hidden_size), input_size, hidden_size)
        self.Who = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)

        self.Wxg = _init_linear(nn.Linear(input_size, hidden_size), input_size, hidden_size)
        self.Whg = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h, c):
        it = self.sigmoid(self.Wxi(x) + self.Whi(h))
        ft = self.sigmoid(self.Wxf(x) + self.Whf(h))
        ot = self.sigmoid(self.Wxo(x) + self.Who(h))
        gt = self.tanh(self.Wxg(x) + self.Whg(h))
        c_next = ft * c + it * gt
        h_next = ot * self.tanh(c_next)
        return h_next, c_next

class PeepholeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PeepholeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wxi = _init_linear(nn.Linear(input_size, hidden_size), input_size, hidden_size)
        self.Whi = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)
        self.Wci = nn.Parameter(torch.zeros(hidden_size))

        self.Wxf = _init_linear(nn.Linear(input_size, hidden_size), input_size, hidden_size)
        self.Whf = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)
        self.Wcf = nn.Parameter(torch.zeros(hidden_size))

        self.Wxo = _init_linear(nn.Linear(input_size, hidden_size), input_size, hidden_size)
        self.Who = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)
        self.Wco = nn.Parameter(torch.zeros(hidden_size))

        self.Wxg = _init_linear(nn.Linear(input_size, hidden_size), input_size, hidden_size)
        self.Whg = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h, c):
        it = self.sigmoid(self.Wxi(x) + self.Whi(h) + self.Wci * c)
        ft = self.sigmoid(self.Wxf(x) + self.Whf(h) + self.Wcf * c)
        c_new = ft * c + it * self.tanh(self.Wxg(x) + self.Whg(h))
        ot = self.sigmoid(self.Wxo(x) + self.Who(h) + self.Wco * c_new)
        h_new = ot * self.tanh(c_new)
        return h_new, c_new

class WorkingMemoryLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(WorkingMemoryLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wxi = _init_linear(nn.Linear(input_size, hidden_size), input_size, hidden_size)
        self.Whi = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)
        self.Wci = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)

        self.Wxf = _init_linear(nn.Linear(input_size, hidden_size), input_size, hidden_size)
        self.Whf = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)
        self.Wcf = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)

        self.Wxo = _init_linear(nn.Linear(input_size, hidden_size), input_size, hidden_size)
        self.Who = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)
        self.Wco = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)

        self.Wxg = _init_linear(nn.Linear(input_size, hidden_size), input_size, hidden_size)
        self.Whg = _init_linear(nn.Linear(hidden_size, hidden_size), hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h, c):
        tanh_ci = self.tanh(self.Wci(c))
        tanh_cf = self.tanh(self.Wcf(c))
        tanh_co = self.tanh(self.Wco(c))

        it = self.sigmoid(self.Wxi(x) + self.Whi(h) + tanh_ci)
        ft = self.sigmoid(self.Wxf(x) + self.Whf(h) + tanh_cf)
        c_new = ft * c + it * self.tanh(self.Wxg(x) + self.Whg(h))
        ot = self.sigmoid(self.Wxo(x) + self.Who(h) + tanh_co)
        h_new = ot * self.tanh(c_new)
        return h_new, c_new
