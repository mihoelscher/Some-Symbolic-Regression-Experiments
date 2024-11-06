import os
import time

import sympy
import torch

from data_utility import process_tensor_with_function, create_input_tensor
import pandas as pd

if __name__ == '__main__':
    usage = ['ginnlp', "parfam", 'mysr']
    # usage = ['ginnlp']
    device = 'cpu'
    function = 'x_0'
    override_results = True
    # parfam fail,
    # ginnlp close: 0.064/(X_0**0.088*X_1**0.09) + 2.001*X_0**3 + 3.014*X_1**0.998
    file = "comparison_table.csv"
    RESULT_COLUMNS = ['target', 'parfam', 'parfam_time', 'ginnLP', 'ginnLP_time', 'mySR', 'mySR_time']
    results = pd.Series([function, 'not yet computed', 'not yet computed', 'not yet computed', 'not yet computed', 'not yet computed', 'not yet computed'],
                        index=RESULT_COLUMNS)
    if os.path.exists(file):
        comparison_table = pd.read_csv(file)
    else:
        comparison_table = pd.DataFrame(columns=RESULT_COLUMNS)

    x_train = torch.linspace(1, 5, 101).to('cpu').unsqueeze(0).T
    y_train = process_tensor_with_function(x_train, function)
    y_train = y_train.squeeze()

    # ----------- PARFAM ------------- #
    if 'parfam' in usage:
        from parfam.parfamwarpper import ParFamWrapper
        parfam = ParFamWrapper(iterate=True, functions=[lambda x: x], function_names=[sympy.Id])
        start_time = time.time()
        parfam.fit(x_train, y_train, time_limit=100)
        end_time = time.time()
        results["parfam"] = parfam.formula.simplify()
        results["parfam_time"] = end_time - start_time
        print(parfam.formula.simplify())

    # ------------ MySR -------------- #
    if 'mysr' in usage:
        from mysr.RationalSR import RationalFunction
        mySR = RationalFunction(3)
        start_time = time.time()
        mySR.fit(x_train.squeeze(), y_train)
        end_time = time.time()
        mySr_function = mySR.get_function()
        results["mySR"] = mySr_function
        results["mySR_time"] = end_time - start_time
        print(mySr_function)

    # ----------- GINN-LP ------------ #
    if 'ginnlp' in usage:
        from ginnlp import GINNLP
        ginnLP = GINNLP(num_epochs=500, round_digits=3, start_ln_blocks=1, growth_steps=3,
                        l1_reg=1e-4, l2_reg=1e-4, init_lr=0.01, decay_steps=1000, reg_change=0.5)
        start_time = time.time()
        ginnLP.fit(x_train, y_train)
        end_time = time.time()
        results["ginnLP"] = ginnLP.recovered_eq
        results["ginnLP_time"] = end_time - start_time
        print(ginnLP.recovered_eq)

    # save results
    if override_results:
        if results.iloc[0] in comparison_table['target'].values:
            comparison_table.loc[comparison_table['target'] == results.iloc[0]] = results.values
        else:
            comparison_table = pd.concat([comparison_table, results.to_frame().T], axis=0)
        comparison_table.to_csv(file, index=False)
