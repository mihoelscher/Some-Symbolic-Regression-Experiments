import torch
import torch.nn as nn

from PtaBlock import PtaBlock


class PtaGrowth(nn.Module):

    def __init__(self, input_dim, max_number_of_blocks=3):
        super(PtaGrowth, self).__init__()
        self.input_dim = input_dim
        self.max_number_of_blocks = max_number_of_blocks
        self.pta_blocks = nn.ModuleList([PtaBlock(num_inputs=self.input_dim) for _ in range(self.max_number_of_blocks)])

    def forward(self, inputs):
        result = 0
        for pta_block in self.pta_blocks:
            result += pta_block(inputs)
        return result

    def to_formula(self, input_names = None):
        terms = [block.to_formula(input_names) for block in self.pta_blocks]
        return " + ".join(terms)

    def prune_exponents(self):
        for pta_block in self.pta_blocks:
            pta_block.prune_exponents()


if __name__ == '__main__':
    model = PtaGrowth(3)
    print(model.to_formula())