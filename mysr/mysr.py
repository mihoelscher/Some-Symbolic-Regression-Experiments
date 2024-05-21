import numpy as np
import sympy
from sympy import symbols
import matplotlib.pyplot as plt
import torch

_base_functions = [
    lambda x: x,
    lambda x: torch.sqrt(torch.abs(x)),
    lambda x: torch.minimum(torch.exp(x), np.exp(10) + torch.abs(x)),
    lambda x: torch.log(torch.abs(x) + 0.000001),
    torch.cos,
    torch.sin]
_base_function_names = [
    'id',
    'sqrt',
    'exp',
    'cos',
    'sin',
    'log']


class MySR:
    def __init__(self, functions=None, function_names=None, processing_unit='cpu'):
        if len(function_names) != len(functions):
            raise ValueError(f'Length of function_names {function_names} and functions {functions} is not the'
                             f' same, please fix this to compute the formula')

        self.processing_unit = processing_unit
        self.functions = functions
        self.function_names = function_names
        if function_names is None:
            self.function_names = _base_function_names
        if functions is None:
            self.functions = _base_functions
