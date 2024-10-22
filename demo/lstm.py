"""
目前实现LSTM（不带proj_size和双向）
参考：
1. pytorch 官方文档 https://pytorch.org/docs/stable/index.html
2. LSTM 论文 https://arxiv.org/pdf/1506.04214.pdf
https://www.zhihu.com/column/c_1733174241909047296
"""
# my LSTM
from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.__version__)

# api
rnn = nn.LSTMCell(10, 20)  # (input_size, hidden_size)
input = torch.randn(2, 3, 10)  # (time_steps, batch, input_size)
h0 = torch.randn(3, 20)  # (batch, hidden_size)
c0 = torch.randn(3, 20)
hx, cx = h0, c0
output = []
for i in range(input.size()[0]):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)
output = torch.stack(output, dim=0)


# for name, param in rnn.named_parameters():
#     print(f"{name}: {param.size()}")


class MyLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros((4 * hidden_size,)))
            self.bias_hh = nn.Parameter(torch.zeros((4 * hidden_size,)))
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]):
        # x: [btz, input_size]
        # states: (hidden_state, c)
        # hidden_state, c: [btz, hidden_size]
        if not x.shape[-1] == self.input_size:
            raise ValueError(f"input_size:{x.shape[-1]} cannot match!")
        if not states[0].shape[-1] == self.hidden_size and states[1].shape[-1] == self.hidden_size:
            raise ValueError("hidden_size cannot match!")
        h, c = states
        # print(self.weight_ih.transpose(0, 1))
        i_f_g_o = torch.matmul(x, self.weight_ih.transpose(-1, -2)) + torch.matmul(h, self.weight_hh.transpose(-1, -2))
        if self.bias:
            i_f_g_o = i_f_g_o + self.bias_ih.view(1, 4 * self.hidden_size) + self.bias_hh.view(1, 4 * self.hidden_size)
        i, f, g, o = torch.chunk(i_f_g_o, 4, dim=-1)
        next_c = F.sigmoid(f) * c + F.sigmoid(i) * F.tanh(g)
        next_h = F.sigmoid(o) * F.tanh(next_c)
        return (next_h, next_c)

    def _init_weights(self, module: torch.nn.Module):
        for param in module.parameters():
            nn.init.uniform_(param, -1 / math.sqrt(self.hidden_size), 1 / math.sqrt(self.hidden_size))


my_rnn = MyLSTMCell(10, 20)
my_rnn.weight_ih, my_rnn.weight_hh = rnn.weight_ih, rnn.weight_hh
my_rnn.bias_ih, my_rnn.bias_hh = rnn.bias_ih, rnn.bias_hh

my_output = []
hx, cx = h0, c0
for i in range(input.size()[0]):
    hx, cx = my_rnn(input[i], (hx, cx))
    my_output.append(hx)
my_output = torch.stack(my_output, dim=0)

# print(f"api_output:{output}")
# print(f"my_output:{my_output}")
print(f"torch.allclose : {torch.allclose(output, my_output)}")


# 多层LSTM
class MyLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([MyLSTMCell(input_size, hidden_size)] + \
                                   [MyLSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)])

    def forward(self, x: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]):
        # x:[seq_len, batch_size, input_size]
        # h, c:[num_layers, batch_size, hidden_size]
        seq_len, btz, input_size = x.shape
        if not input_size == self.input_size:
            raise ValueError(f"input_size:{input_size} cannot match the module's input_size:{self.input_size}")
        h, c = states  #
        if not (h.shape[-1] == self.hidden_size and c.shape[-1] == self.hidden_size):
            raise ValueError("hidden_size cannot match!")

        h, c = list(torch.unbind(h)), list(torch.unbind(c))
        output = []  # [seq_len, batch_size, hidden_size]

        for t in range(seq_len):
            inp = x[t]  # [btz, inpit_size]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](inp, (h[layer], c[layer]))
                inp = h[layer]
            output.append(h[-1])
        output = torch.stack(output)
        h = torch.stack(h)
        c = torch.stack(c)
        return output, (h, c)


h0 = torch.randn(3, 3, 20)
c0 = torch.randn(3, 3, 20)

# api
lstm = nn.LSTM(10, 20, num_layers=3)
output, (hn, cn) = lstm(input, (h0, c0))
# print(output.size(), hn.size(), cn.size(), sep='\n')
# print(output, hn, cn, sep='\n')
# for name, param in lstm.named_parameters():
#     print(f"{name}: {param.size()}")

# print("*"*100)

my_lstm = MyLSTM(10, 20, num_layers=3)

# for name, param in my_lstm.named_parameters():
#     print(f"{name}: {param.size()}")

# 配置模型参数

my_lstm.cells[0].weight_ih = lstm.weight_ih_l0
my_lstm.cells[0].weight_hh = lstm.weight_hh_l0
my_lstm.cells[0].bias_ih = lstm.bias_ih_l0
my_lstm.cells[0].bias_hh = lstm.bias_hh_l0

my_lstm.cells[1].weight_ih = lstm.weight_ih_l1
my_lstm.cells[1].weight_hh = lstm.weight_hh_l1
my_lstm.cells[1].bias_ih = lstm.bias_ih_l1
my_lstm.cells[1].bias_hh = lstm.bias_hh_l1

my_lstm.cells[2].weight_ih = lstm.weight_ih_l2
my_lstm.cells[2].weight_hh = lstm.weight_hh_l2
my_lstm.cells[2].bias_ih = lstm.bias_ih_l2
my_lstm.cells[2].bias_hh = lstm.bias_hh_l2

my_output, (my_hn, my_cn) = my_lstm(input, (h0, c0))
print(my_output.size(), my_hn.size(), my_cn.size(), sep='\n')
# print(my_output, my_hn, my_cn, sep='\n')
print(f"torch.allclose(output, my_output): {torch.allclose(output, my_output, atol=1e-5)}")
print(f"torch.allclose(hn, my_hn): {torch.allclose(hn, my_hn, atol=1e-5)}")
print(f"torch.allclose(cn, my_cn): {torch.allclose(cn, my_cn, atol=1e-5)}")

"""
输出：
2.0.1
torch.allclose : True
torch.allclose(output, my_output): True
torch.allclose(hn, my_hn): True
torch.allclose(cn, my_cn): True
"""