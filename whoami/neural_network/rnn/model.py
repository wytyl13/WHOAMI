#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/06 17:56
@Author  : weiyutao
@File    : model.py

RNN
    h_t = tanh(W_ih*x_t + W_hh*h_t-1 + b), h_t is range from -1 to 1, 
    h_t is the hidden state of current step, dimension is hidden_size,
    W_ih and W_hh is weight matrix acting on the current input x_t and the hidden state
    of the previous time step h_t-1, respectively.
    why tanh? tanh normalizes the hidden state to (-1, 1), keeping the values stable.
    tanh is a nonlinear function that intriduces nonlinear properties that allow simple 
    RNNs to learn and represent more complex patterns. The output of tanh is symmetric around 0
    (i.e., the output of tanh(0)=0), a property that helps to balance the gradient update and avoid biases
    that are too large or too small.
    
    Notice, you should distinguish the input_size, layer_input_size, hidden_size and real_hidden_size.
    
    x1 (batch_size, layer_input_size): the current token feature_size what is passed layer1, layer_input_size = input_size if t = 1
    x2+ (batch_size, layer_input_size): 
        the current token feature_size what is passed layer2+, layer_input_size = real_hidden_size*num_directions, 
        real_hidden_size = hidden_size if proj_size=0 else proj_size, num_directions = 1 or 2.
        why layer_input_size = real_hidden_size*num_directions when t is greater than 1?
        because the xt is equal to the concatenate(forward, backward) = concatenate(real_hidden_size, real_hidden_size) = 2*real_hidden_size if num_directions == 2.
    
    x_t(batch_size, layer_input_size),
    w_ih(gate_size, layer_input_size), gate_size = hidden_size if simple RNN else gate_num*hidden_size
    (x_t @ w_ih.T + b_ih)  = (batch_size, layer_input_size) @ (layer_input_size, gate_size) + (gate_size,) = (batch_size, gate_size)
    
    prev_hidden(batch_size, real_hidden_size)
    w_hh(gate_size, real_hidden_size)
    (prev_hidden @ w_hh.T + b_hh) = (batch_size, real_hidden_size) @ (real_hidden_size, gate_size) + (gate_size,) = (batch_size, gate_size)
    
    xt: 当前时间步的输入序列
    combined = (xt @ w_ih.T + b_ih) + (prev_hidden @ w_hh.T + b_hh) = (batch_size, gate_size)
    tanh(combined) = (batch_size, gate_size)
    projection layer: if projection layer, hidden size -> projection size -> (batch_size, real_hidden_size) what is also the dimension of prev_hidden for x_t+1
    
    layer_num what is the layers of RNN
    the hx what is the all hidden state of the current time step, the dimension is hx(layer_nums*num_directions, batch_size, real_hidden_size), real_hidden_size=proj_size if proj_size > 0 else hidden_size
    if num_directions = 2 && layer_nums = 2
        hx[0] = layer_1_forward = (batch_size, real_hidden_size)
        hx[1] = layer_1_backward = (batch_size, real_hidden_size)
        hx[2] = layer_2_forward = (batch_size, real_hidden_size)
        hx[3] = layer_2_backward = (batch_size, real_hidden_size)

LSTM
    forget gate, input gate, cell state, output gate.
    Just like RNN, LSTM add the forget gate, input gate, cell state and output gate based on RNN.
    Just like hidden_size = 128, input_size is 28
    real_hidden_size = proj_size if proj_size > 0 else hidden_size
    layer_input_size = input_size if layer_num == 0 else real_hidden_size*num_directions
    f_t = sigmoid(Wf*[h_t-1, x_t] + b_f), wf is forget gate weights, [h_t-1, x_t] is vector concatenation operation, 
        h_t-1(real_hidden_size, 1), h_t-1(hidden_size, 1) if do not apply projection in forget gate, 
        x_t(real_hidden_size, 1), x_t(hidden_size, 1) if do not apply projection in forget gate. binary directions is not suitable for the gate.
        concatenation(h_t-1, x_t) = [h_t-1, x_t] = 
            (input_size + real_hidden_size, 1) if layer_num == 0 else (real_hidden_size + real_hidden_size, 1)
        b_f(real_hidden_size)
        Wf = (real_hidden_size, input_size + real_hidden_size) if layer_num == 0 else (real_hidden_size, real_hidden_size + real_hidden_size)
        f_t = (real_hidden_size, 1)
    
    i_t = sigmoid(Wi*[h_t-1, x_t] + b_i), 
        ...
        i_t = (real_hidden_size, 1)
        
    C~_t = tanh(Wc*[h_t-1, x_t] + b_c),
        C~_t = (real_hidden_size, 1)
    C_t = f_t*C_t-1 + i_t*C~_t,
        C_t = (real_hidden_size, 1) * (real_hidden_size, 1) + (real_hidden_size, 1) * (real_hidden_size, 1) = (real_hidden_size, 1)
    
    output layer
    o_t = sigmoid(Wo*[h_t-1, x_t] + b_o)
        o_t = (real_hidden_size, 1)
    h_t = o_t * tanh(C_t) = (real_hidden_size, 1) * (real_hidden_size, 1) = (real_hidden_size, 1)
GRU

"""

from torch import nn
from torch import Tensor
import weakref
import torch
import numbers
import warnings
from torch.nn import Parameter
from typing import (
    Tuple,
    Optional,
    List,
    overload
)
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F


class RNNBase(nn.Module):

    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool
    proj_size: int

    def __init__(self,
                mode: str,
                input_size: int,
                hidden_size: int,
                num_layers: int,
                bias: bool = True,
                batch_first: bool = False,
                dropout: float = 0.,
                bidirectional: bool = False,
                proj_size: int = 0, # projection layer
                device=None,
                dtype=None
            ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.num_directions = 2 if bidirectional else 1
        
        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
            isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1]")
        
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          f"num_layers greater than 1, but got dropout={dropout} and "
                          f"num_layers={num_layers}")

        if not isinstance(hidden_size, int):
            raise TypeError(f"hidden_size should be of type int, got: {type(hidden_size).__name__}")
        
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")
        
        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN_TANH':
            gate_size = hidden_size
        elif mode == 'RNN_RELU':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)
        
        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                # why use real_hidden_size? hidden_size is the objective hidden size of neural network.
                # real_hidden_size is the hidden size what consider the projection.
                real_hidden_size = proj_size if proj_size > 0 else hidden_size
                layer_input_size = input_size if layer == 0 else real_hidden_size * self.num_directions
                
                # input_size(batch_size, layer_input_size)
                # w_ih(gate_size, layer_input_size)
                # input_size*w_ih^T(batch_size, gate_size) # 中间结果
                # 如果是简单的rnn，gate_size = hidden_size = real_hidden_size
                # 如果是LSTM，gate_size = hidden_size * gate_num
                # 特别地对于LSTM，对于每个门，这里将所有的门对应的权重初始化进行了一个纵向拼接
                w_ih = Parameter(torch.empty((gate_size, layer_input_size), **factory_kwargs))

                # hidden_state(batch_size, real_hidden_size)
                # w_hh(gate_size, real_hidden_size)
                # hidden_state*w_hh^T(batch_size, gate_size)
                w_hh = Parameter(torch.empty((gate_size, real_hidden_size), **factory_kwargs))
                b_ih = Parameter(torch.empty(gate_size, **factory_kwargs))
                b_hh = Parameter(torch.empty(gate_size, **factory_kwargs))
                layer_params: Tuple[Tensor, ...] = () # [w_ih, w_hh, b_ih, b_hh, w_hr]
                if self.proj_size == 0:
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh)
                    else:
                        layer_params = (w_ih, w_hh)
                else:
                    w_hr = Parameter(torch.empty((proj_size, hidden_size)), **factory_kwargs)
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh, w_hr)
                    else:
                        layer_params = (w_ih, w_hh, w_hr)
                
                suffix = '_reverse' if direction == 1 else '' # direction=1 reverse, direction=0 forward.
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                if self.proj_size > 0:
                    param_names += ['weight_hr_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]
                
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using appropriate strategies for RNN architectures"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights - use orthogonal initialization for recurrent connections
                nn.init.orthogonal_(param)
            elif 'weight_hr' in name:
                # Hidden-to-projection weights (if using projection)
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.zeros_(param)
    
    
    def check_input(self, input: Tensor) -> None:
        expected_input_dim = 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                f"input must have {expected_input_dim} dimensions, got {input.dim()}")
        
        if self.input_size != input.size(-1):
            raise RuntimeError(
                f"input.size(-1) must be equal to input_size. Exepcted {self.input_size}, got {input.size(-1)}")


    def get_expected_hidden_size(self, input: Tensor) -> Tuple[int, int, int]:
        mini_batch = input.size(0) if self.batch_first else input.size(1)
        if self.proj_size > 0:
            expected_hidden_size = (self.num_layers * self.num_directions, 
                                    mini_batch, self.proj_size)
        else:
            expected_hidden_size = (self.num_layers * self.num_directions,
                                    mini_batch, self.hidden_size)
        return expected_hidden_size


    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int]):
        if hx.size() != expected_hidden_size:
            raise RuntimeError(f"Expected hidden size {expected_hidden_size}, got {list(hx.size())}")


    def check_forward_args(self, input: Tensor, hidden: Tensor):
        self.check_input(input)
        expected_hidden_size = self.get_expected_hidden_size(input)
        self.check_hidden_size(hidden, expected_hidden_size)


class RNN(RNNBase):# 

    @overload
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 1, 
        nonlinearity: str = 'tanh',
        bias: bool = True, 
        batch_first: bool = False, 
        dropout: float = 0.,
        bidirectional: bool = False, 
        device=None, 
        dtype=None):
        ...


    @overload
    def __init__(self, *args, **kwargs):
        ...


    def __init__(self, *args, **kwargs):
        
        if 'proj_size' in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        self.nonlinearity = kwargs.pop('nonlinearity', 'tanh')
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif self.nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError(f"Unknown nonlinearity '{self.nonlinearity}'. Select from 'tanh' or 'relu'.")
        super().__init__(mode, *args, **kwargs)


    def _rnn_forward(self, input: Tensor, hx: Tensor, activation) -> Tuple[Tensor, Tensor]:
        """RNN forward"""
    
        # clone the hidden state.
        # real_hidden_size = hidden_size if rnn(proj_size = 0)
        hx_output = hx.clone() # hx_output(layer_nums*num_directions, batch_size, hidden_size)

        if self.batch_first:
            # if batch_first, transpose input from (batch_size, seq_len, feature_size) to (seq_len, batch_size, feature_size)
            input = input.transpose(0, 1) 
        
        seq_len, batch_size, _ = input.size()
        
        # layer_output what store all the sequence output for the all layers. 
        # but we just need to return the final layer output for all the sequences each forward.
        # and we need to use the former layer output in current layer for each sequences for each forward.
        
        # The input of first layer is the feature for each sequences.
        layer_output = input # layer_output(seq_len, batch_size, input_size), input_size = feature_size.
        
        for layer in range(self.num_layers):

            # forward
            h_forward = hx[layer * self.num_directions].clone() # h_forward(batch_size, hidden_size)
            
            # get forward weights.
            w_ih = getattr(self, f'weight_ih_l{layer}') # w_ih(hidden_size, input_size) if layer == 0 else (hidden_size, hidden_size)
            w_hh = getattr(self, f'weight_hh_l{layer}') # w_hh(hidden_size, hidden_size)
            b_ih = getattr(self, f'bias_ih_l{layer}') if self.bias else None # b_ih(hidden_size, )
            b_hh = getattr(self, f'bias_hh_l{layer}') if self.bias else None # b_hh(hidden_size, )
            
            forward_outputs = []
            for t in range(seq_len):
                x = layer_output[t] 
                # x(batch_size, input_size), input_size = feature_size if layer == 0, 
                # x(batch_size, input_size), input_size = feature_size if layer == 0, 
                
                # (batch_size, input_size) @ (input_size, hidden_size) + (hidden_size, ) \
                    # + (batch_size, hidden_size) @ (hidden_size, hidden_size) + (hidden_size, ) \
                    # = (batch_size, hidden_size) if layer == 0 else \
                    # (batch_size, hidden_size) @ (hidden_size, hidden_size) + (hidden_size, ) + \
                    # (batch_size, hidden_size) @ (hidden_size, hidden_size) + (hidden_size) = (batch_size, hidden_size)
                # when the first layer, x is the sequence input(batch_size, input_size)
                # when the other layer, x is the correspond time step sequence hidden state of former layer.
                # h_forward is hx[layer*directions] what can be set used zero in first batch train or inference, 
                # and can be set as the last hidden state of each layer by former training batch if the first time step(the first sequence for each layer)
                # else h_forward is the former time step(former sequence) hidden state of the current layer.
                gates = F.linear(x, w_ih, b_ih) + F.linear(h_forward, w_hh, b_hh) 
                h_forward = activation(gates)
                forward_outputs.append(h_forward)
            
            # update the hidden state of the last sequence for each layer and each direction.
            # h_forward(batch, hidden_size)
            hx_output[layer * self.num_directions] = h_forward
            
            
            # handle the bidirectional.
            if self.bidirectional:
                # because the hx is [layer1_forward, layer1_backward, layer2_forward, layer2_backward].
                h_backward = hx[layer * self.num_directions + 1].clone() 
                
                # get backward weights. dimension is same to forward.
                w_ih_reverse = getattr(self, f'weight_ih_l{layer}_reverse')
                w_hh_reverse = getattr(self, f'weight_hh_l{layer}_reverse')
                b_ih_reverse = getattr(self, f'bias_ih_l{layer}_reverse') if self.bias else None
                b_hh_reverse = getattr(self, f'bias_hh_l{layer}_reverse') if self.bias else None
                
                backward_outputs = []
                for t in range(seq_len - 1, -1, -1):
                    x = layer_output[t]
                    gates = F.linear(x, w_ih_reverse, b_ih_reverse) + F.linear(h_backward, w_hh_reverse, b_hh_reverse)
                    h_backward = activation(gates)
                    backward_outputs.append(h_backward)
                
                # Invert the backward outputs because we will cat the forward and backward.
                backward_outputs = backward_outputs[::-1]
                
                # update the backward hidden state.
                hx_output[layer * self.num_directions + 1] = h_backward
                
                # cat the forward and backward output.
                layer_output = torch.stack([
                    torch.cat([forward_outputs[t], backward_outputs[t]], dim=-1)
                    for t in range(seq_len)
                ], dim=0)
            else:
                # num_directions = 1
                # forward_outputs[(batch_size, hidden_size), ...], len(forward_outputs) = seq_len
                # the dimension of torch.stack(forward_outputs, dim=0) is (seq_len, batch_size, hidden_size*num_directions)
                # what store the hidden state of each sequences of the current layer.
                # why? because we will use the hidden state of each sequences of former layer as the xt.
                layer_output = torch.stack(forward_outputs, dim=0) 
            
            # apply dropout except the last layer.
            if self.dropout > 0 and self.training and layer < self.num_layers - 1:
                layer_output = F.dropout(layer_output, p=self.dropout, training=self.training)
        
        # transpose to (batch_size, seq_len, hidden_size*num_directions) if self.batch_first. 
        if self.batch_first:
            layer_output = layer_output.transpose(0, 1)
        
        return layer_output, hx_output 
        # layer_output(seq_len, batch_size, real_hidden_size*num_directions), hx_output(layer_nums*num_directions, batch_size, real_hidden_size)
        # layer_output 代表每一个批次每一个序列的前向和反向隐藏状态（一般用于命名体识别或注意力计算任务）
        # hx_output 代表每层每个方向每个批次最终的隐藏状态（一般用于文本分类任务）


    def forward(self, input, hx=None):
        """
        Args:
            input (Tensor): input data for x_0(sequence_length, batch_size, feature_size) if t = 0. The original input data.
                            # input data for each layer. x_t(batch_size, layer_input_size), layer_input_size = input_size if t = 1, 
                            # layer_input_size = real_hidden_size*num_directions, real_hidden_size = hidden_size if proj_size=0 else proj_size, num_directions = 1 or 2.
            hx (Tensor, optional): hidden state. what is the all hidden state of the current time step, the dimension is 
                                    hx(layer_nums*num_directions, batch_size, real_hidden_size), real_hidden_size=proj_size if proj_size > 0 else hidden_size
                hx存储每一层初始时刻的隐藏状态
        Raises:
            ValueError: _description_
        Return:
            hidden: hidden存储每一层最终的隐藏状态，可以用来做文本分类，hidden为当前批次的最终隐藏状态输出，可以作为下一个批次的每一层的初始隐藏状态去重新进行训练。
            output: 代表每一个批次每一个序列的前向和反向隐藏状态（一般用于命名体识别或注意力计算任务）
        """
        if input.dim() not in (2, 3):
            raise ValueError(f"RNN Expected input to be 2D or 3D, got {input.dim()}D tensor instead")
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        if not is_batched:
            input = input.unsqueeze(batch_dim) # insert batch dimension 1 if not is_batched.
            # input dimension can be (batch_size, sequence_length, feature_size) or (sequence_length, batch_size, feature_size)
            
            if hx is not None:
                if hx.dim() != 2:
                    raise RuntimeError(
                        f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor")
                hx = hx.unsqueeze(1) # the dimension of hidden state must be (num_layers*num_directions, batch_size, hidden_size)
        else:
            if hx is not None and hx.dim() != 3:
                raise RuntimeError(
                    f"For batched 3-D input, hx should be 3-D but got {hx.dim()}-D tensor")
        max_batch_size = input.size(0) if self.batch_first else input.size(1)

    
        # init the hx, just suitable for the first batch training.
        if hx is None:
            # because RNN is not support projection, so real_hidden_size = hidden_size
            # the dimension of hx is hx(num_layers*num_directions, batch_size, real_hidden_size), real_hidden_size=proj_size if proj_size > 0 else hidden_size
            hx = torch.zeros(
                self.num_layers*self.num_directions,
                max_batch_size, self.hidden_size, 
                dtype=input.dtype, device=input.device)
        
        assert hx is not None
        self.check_forward_args(input, hx)
        assert self.mode == 'RNN_TANH' or self.mode == 'RNN_RELU'
        
        activation = torch.tanh if self.mode == 'RNN_TANH' else torch.relu
        output, hidden = self._rnn_forward(input, hx, activation)
        # 处理非批次输出
        if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = hidden.squeeze(1)
        
        return output, hidden


class LSTM(RNNBase):
    """LSTM"""
    
    @overload
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 1, 
        bias: bool = True, 
        batch_first: bool = False, 
        dropout: float = 0., 
        bidirectional: bool = False, 
        proj_size: int = 0, 
        device=None, 
        dtype=None) -> None:
        ...
    
    
    @overload
    def __init__(self, *args, **kwargs):
        ...
    
    
    def __init__(self, *args, **kwargs):
        super().__init__('LSTM', *args, **kwargs)


    def _lstm_forward(self, 
                      input: Tensor,
                      hx: Tuple[Tensor, Tensor]
                      ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h_0, c_0 = hx
        
        h_n = h_0.clone()
        c_n = c_0.clone()
        if self.batch_first:
            input = input.transpose(0, 1)

        seq_len, batch_size, _ = input.size()
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        
        layer_output = input # (seq_len, batch_size, input_size)
        for layer in range(self.num_layers):
            forward_outputs = []

            h_forward = h_0[layer * self.num_directions].clone() # (batch_size, real_hidden_size)
            c_forward = c_0[layer * self.num_directions].clone() # (batch_size, hidden_size)

            # get forward weights and bias
            w_ih = getattr(self, f'weight_ih_l{layer}') # (4*hidden_size, layer_input_size)
            w_hh = getattr(self, f'weight_hh_l{layer}') # (4*hidden_size, real_hidden_size)
            b_ih = getattr(self, f'bias_ih_l{layer}') if self.bias else None # (4*hidden_size, 1)
            b_hh = getattr(self, f'bias_hh_l{layer}') if self.bias else None # (4*hidden_size, 1)


            if self.proj_size > 0:
                w_hr = getattr(self, f'weight_hr_l{layer}') # (proj_size, hidden_size)

            # process the sequence for forward direction
            for t in range(seq_len):
                x = layer_output[t] # (batch_size, layer_input_size)

                gates = F.linear(x, w_ih, b_ih) + F.linear(h_forward, w_hh, b_hh) # (batch_size, 4*hidden_size)

                chunks = gates.chunk(4, 1)
                i_gate = torch.sigmoid(chunks[0]) # input gate(batch_size, hidden_size)
                f_gate = torch.sigmoid(chunks[1]) # forget gate(batch_size, hidden_size)
                g_gate = torch.sigmoid(chunks[2]) # cell input(batch_size, hidden_size)
                o_gate = torch.sigmoid(chunks[3]) # output gate(batch_size, hidden_size)
                
                # c_t = f_t * c_t-1 + i_t * g_t
                c_forward = f_gate * c_forward + i_gate * g_gate # (batch_size, hidden_size)

                # h_t = o_t * tanh(c_t)
                h_prime = o_gate * torch.tanh(c_forward)

                # apply projection if need.
                if self.proj_size > 0:
                    h_forward = F.linear(h_prime, w_hr) # (batch_size, proj_size)
                else:
                    h_forward = h_prime # (batch_size, hidden_size)
                
                forward_outputs.append(h_forward)
            
            h_n[layer * self.num_directions] = h_forward
            c_n[layer * self.num_directions] = c_forward
            
            # handle bidirectional LSTM
            if self.bidirectional:
                backward_outputs = []
                h_backward = h_0[layer * self.num_directions].clone()
                c_backward = c_0[layer * self.num_directions].clone()
                
                # get backward weights and biases.
                w_ih_reverse = getattr(self, f'weight_ih_l{layer}_reverse') # (4*hidden_size, layer_input_size)
                w_hh_reverse = getattr(self, f'weight_hh_l{layer}_reverse') # (4*hidden_size, layer_input_size) 
                b_ih_reverse = getattr(self, f'bias_ih_l{layer}_reverse') # (4*hidden_size, )
                b_hh_reverse = getattr(self, f'bias_hh_l{layer}_reverse') # (4*hidden_size, )

                if self.proj_size > 0:
                    w_hr_reverse = getattr(self, f'weight_hr_l{layer}_reverse') # (proj_size, hidden_size)
                
                for t in range(seq_len - 1, -1, -1):
                    x = layer_output[t] # (batch_size, layer_input_size)

                    # calculate gates.
                    gates = F.linear(x, w_ih_reverse, b_ih_reverse) + F.linear(h_backward, w_hh_reverse, b_hh_reverse) # (batch_size, 4*hidden_size)
                    chunks = gates.chunk(4, 1)
                    i_gate = torch.sigmoid(chunks[0]) # input gate: (batch_size, hidden_size)
                    f_gate = torch.sigmoid(chunks[1]) # forget gate: (batch_size, hidden_size)
                    g_gate = torch.tanh(chunks[2]) # cell input: (batch_size, hidden_size)
                    o_gate = torch.sigmoid(chunks[3]) # output gate: (batch_size, hidden_size)


                    c_backward = f_gate * c_backward + i_gate * g_gate # (batch_size, hidden_size)
                    h_prime = o_gate * torch.tanh(c_backward)

                    # apply projection if need.
                    if self.proj_size > 0:
                        h_backward = F.linear(h_prime, w_hr_reverse) # (batch_size, hidden_size)
                    else:
                        h_backward = h_prime # (batch_size, hidden_size)
                    backward_outputs.append(h_backward)
                
                # Reverse the backward outputs to match sequence order.
                backward_outputs = backward_outputs[::-1]

                # update the final hidden and cell states for this layer (backward direction)
                h_n[layer * self.num_directions + 1] = h_backward
                c_n[layer * self.num_directions + 1] = c_backward
                
                # concatenate forward and backward outputs for each time step
                layer_output = torch.stack([
                    torch.cat([forward_outputs[t], backward_outputs[t]], dim=-1)
                    for t in range(seq_len)
                ], dim=0)
            else:
                layer_output = torch.stack(forward_outputs, dim=0) # (seq_len, batch_size, real_hidden_size)
            
            # apply dropout except for the last layer.
            if self.dropout > 0 and layer < self.num_layers - 1 and self.training:
                layer_output = F.dropout(layer_output, p=self.dropout, training=True)
            
        if self.batch_first:
            layer_output = layer_output.transpose(0, 1) # (seq_len, batch_size, real_hidden_size*num_directions) -> (batch_size, seq_len, real_hidden_size*num_directions)
        
        return layer_output, (h_n, c_n)


    def forward(self, input, hx=None):
        num_directions = 2 if self.bidirectional else 1
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        
        if input.dim() not in (2, 3):
            raise ValueError(f"LSTM Expected input to be 2D or 3D, got {input.dim()}D tensor instead")
        
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if hx is not None:
                if isinstance(hx, tuple) or isinstance(hx, list):
                    hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))
                else:
                    raise TypeError("For LSTM, expected hx to be a tuple (h_0, c_0)")
        max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            zeros = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size, 
                real_hidden_size,
                dtype=input.dtype,
                device=input.device
            ) # consider projection if need, and the dimension of h_zeros is after projection.
            
            # for LSTM, we need both h_0 and c_0
            h_zeros = zeros
            c_zeros = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size, # cell state is always hidden_size, not real_hidden_size. because the cell do not apply projection
                dtype=input.dtype,
                device=input.device
            )
            hx = (h_zeros, c_zeros)
        assert hx is not None
        h_0, c_0 = hx
        self.check_forward_args(input, h_0)
        output, (h_n, c_n) = self._lstm_forward(input, (h_0, c_0))

        
        if not is_batched:
            output = output.squeeze(batch_dim)
            h_n = h_n.squeeze(1)
            c_n = c_n.squeeze(1)
        return output, (h_n, c_n)
            
            
            
            
