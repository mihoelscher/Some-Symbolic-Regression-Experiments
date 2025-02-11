from argparse import ArgumentError

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers


class PTABlock(nn.Module):

    def __init__(self, num_inputs, block_number=1, l1_reg=0.0, l2_reg=0.0):
        super(PTABlock, self).__init__()
        # Store parameters for regularization
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.block_number = block_number
        self.losses = []
        self.exponents = nn.Parameter(torch.randn(num_inputs), requires_grad = True)

    def forward(self, inputs_x):
        # inputs_x should be a list or tensor where each element represents x_i
        if not isinstance(inputs_x, torch.Tensor):
            raise ArgumentError(inputs_x, "PTA Input should be a torch.Tensor")

        # Raise each input x_i to its corresponding exponent e_i
        powered_inputs = inputs_x ** self.exponents  # Element-wise exponentiation

        # Take the product of all powered inputs along the feature dimension (dim=-1)
        output = torch.prod(powered_inputs, dim=-1).unsqueeze(1)  # Shape: (batch_size,1)

        # Optional: apply L1 and L2 regularization
        if self.l1_reg > 0 or self.l2_reg > 0:
            l1_penalty = torch.sum(torch.abs(self.exponents)) * self.l1_reg
            l2_penalty = torch.sum(self.exponents ** 2) * self.l2_reg
            output += l1_penalty + l2_penalty
        return output


    def call(self, inputs):
        if not isinstance(inputs, tf.Tensor):
            raise ValueError("PTA Input should be a tf.Tensor")

        # Raise each input x_i to its corresponding exponent e_i
        powered_inputs = tf.pow(inputs, self.exponents)  # Element-wise exponentiation

        # Take the product of all powered inputs along the feature dimension (axis=-1)
        output = tf.reduce_prod(powered_inputs, axis=-1, keepdims=True)  # Shape: (batch_size, 1)

        return output

class PTABlockWithLog(nn.Module):
    def __init__(self, num_inputs, block_number=1, l1_reg=0.0, l2_reg=0.0):
        super(PTABlockWithLog, self).__init__()

        # Store parameters for regularization
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.block_number = block_number
        self.exponents = None
        self.losses = []

        # Log activation and exponential activation
        self.log_activation = lambda x: torch.log(torch.abs(x) + 1e-10)

        # Identity-initialized linear layers without bias
        self.identity_layers = nn.ModuleList([
            nn.Linear(1, 1, bias=False) for _ in range(num_inputs)
        ])
        for layer in self.identity_layers:
            with torch.no_grad():
                layer.weight.copy_(torch.eye(1))  # Identity initialization

        # Final dense layer with exponential activation
        self.ln_dense = nn.Linear(num_inputs, 1, bias=False)

    def forward(self, inputs_x):
        # Apply each identity-initialized layer to corresponding input
        x_tensors = torch.split(inputs_x, 1, dim=1)
        ln_layers = [self.log_activation(layer(x)) for layer, x in zip(self.identity_layers, x_tensors)]

        # Concatenate along the feature dimension if there are multiple ln_layers
        if len(ln_layers) == 1:
            ln_concat = ln_layers[0]
        else:
            ln_concat = torch.cat(ln_layers, dim=-1)

        self.exponents = self.ln_dense.weight


        # Final dense layer with exponential activation
        ln_dense_output = torch.exp(self.ln_dense(ln_concat))

        # L1 and L2 regularization terms
        l1_penalty = sum(torch.sum(torch.abs(param)) for param in self.parameters())
        l2_penalty = sum(torch.sum(param ** 2) for param in self.parameters())

        # Apply regularization
        if self.l1_reg > 0:
            ln_dense_output += self.l1_reg * l1_penalty
        if self.l2_reg > 0:
            ln_dense_output += self.l2_reg * l2_penalty

        return ln_dense_output