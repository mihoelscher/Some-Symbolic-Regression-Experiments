import time
from contextlib import contextmanager
import sys, os
from typing import final

import pandas as pd
from ginnlp import GINNLP
from pathlib import Path

# Set up default values for GINNLP
num_epochs = 500
round_digits = 3
start_ln_blocks = 1
growth_steps = 3
l1_reg = 1e-4
l2_reg = 1e-4
init_lr = 0.01
decay_steps = 1000
reg_change = 0.5
train_iter = 4

# Relative Paths
dataset_dir = Path.cwd().parent.parent.joinpath('Datasets')
data_dir = dataset_dir.joinpath('Feynman_with_units')


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_test_results_for_file(filename: str):
    sep = ' '

    df = pd.read_csv(data_dir.joinpath(filename), sep=sep, header=None).iloc[:,:-1]
    print(df.shape)
    df = df.sample(10000)
    train_x = df.iloc[:, :-1].values.astype(float)
    train_y = df.iloc[:, -1].values

    model = GINNLP(num_epochs=num_epochs, round_digits=round_digits, start_ln_blocks=start_ln_blocks,
                   growth_steps=growth_steps, l1_reg=l1_reg, l2_reg=l2_reg,
                   init_lr=init_lr, decay_steps=decay_steps, reg_change=reg_change, train_iter=train_iter)
    start_time = time.time()
    with suppress_stdout():
        model.fit(train_x, train_y)
    run_time = time.time() - start_time
    final_error = model.train_history[-1].history['loss'][-1]
    print(f'Finished training for {filename}')
    return [model.recovered_eq, final_error, run_time]


if __name__ == '__main__':
    function_infos = pd.read_csv(dataset_dir.joinpath('Feynmann_laurent_polynomials.csv'), sep=';')
    data_dict = dict(zip(function_infos['Filename'], function_infos['Formula']))

    columns = ['Filename', 'Target Formula', 'Recovered Formula', 'Last Error', 'Run Time']
    result_table = pd.DataFrame(columns=columns)

    for name, target_formula in data_dict.items():
        test_result = [name, target_formula] + get_test_results_for_file(name)
        result_table.loc[len(result_table)] = test_result

    result_table.to_csv('ginn_lp_feynmann_results.csv', index=False)
