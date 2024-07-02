import sympy
from parfam.parfamwarpper import ParFamWrapper
from mysr.RationalSR import RationalFunction
from ginnlp.ginnlp import GINNLP
from data_utility import process_tensor_with_function
import torch

if __name__ == '__main__':
    usage = ['ginnlp']
    device = 'cpu'
    function = '2*x_0**3+3*x_1+2'  # parfam fail,
    # ginnlp close: 0.064/(X_0**0.088*X_1**0.09) + 2.001*X_0**3 + 3.014*X_1**0.998
    x_train = torch.rand((1000, 2))*5
    y_train = process_tensor_with_function(x_train, function)
    y_train = y_train.squeeze()

    # ----------- PARFAM ------------- #
    if 'parfam' in usage:
        parfam = ParFamWrapper(iterate=True, functions=[lambda x: x], function_names=[sympy.Id])
        parfam.fit(x_train, y_train)
        print(parfam.formula.simplify())

    # ------------ MySR -------------- #
    if 'mysr' in usage:
        mySR = RationalFunction(3, 0)
        mySR.fit(x_train.squeeze(), y_train)
        mySr_function = mySR.get_function()
        print(mySr_function)

    # ----------- GINN-LP ------------ #
    if 'ginnlp' in usage:
        ginnLP = GINNLP()
        ginnLP.fit(x_train.squeeze(), y_train)
        print(ginnLP.recovered_eq)
