from argparse import ArgumentError
import torch
import torch.nn as nn


class PtaBlock(nn.Module):

    def __init__(self, num_inputs):
        super(PtaBlock, self).__init__()
        self.losses = []
        self.exponents = nn.Parameter(torch.randn(num_inputs), requires_grad=True)
        self.active_weight = nn.Parameter(torch.randn(1))

    def forward(self, inputs_x):
        # inputs_x should be a list or tensor where each element represents x_i
        if not isinstance(inputs_x, torch.Tensor):
            raise ArgumentError(inputs_x, "PTA Input should be a torch.Tensor")

        # Raise each input x_i to its corresponding exponent e_i
        powered_inputs = (inputs_x ** self.exponents)  # Element-wise exponentiation

        active_gate = torch.sigmoid(self.active_weight)

        # Take the product of all powered inputs along the feature dimension (dim=-1)
        return active_gate * torch.prod(powered_inputs, dim=-1).unsqueeze(1)  # Shape: (batch_size,1)

    def to_formula(self, input_names = None):
        if self.active_weight == 0:
            return '0'
        if not input_names:
            terms = [f"x_{i + 1}**{self.exponents[i].item():.4f}" for i in range(len(self.exponents))]
        else:
            terms = [f"{input_names[i]}**{self.exponents[i].item():.4f}" for i in range(len(self.exponents))]
        return " * ".join(terms)

    def prune_exponents(self):
        with torch.no_grad():
            self.exponents[torch.abs(self.exponents) < 0.001] = 0.0
            self.active_weight[torch.abs(self.active_weight) < 0.01] = 0.0
            if self.active_weight == 0.0:
                self.exponents[True] = 0.0



if __name__ == '__main__':
    block = PtaBlock(2)
    block.exponents = nn.Parameter(torch.tensor([1, 2]), requires_grad=False)
    print(block(torch.tensor([[3,2]])))
