import time
from contextlib import contextmanager
import sys, os
from parfam.parfamwarpper import ParFamWrapper
import pandas as pd
from pathlib import Path

# Set up default values for Parfam
max_deg_input = 2
max_deg_output = 4
max_deg_input_denominator = 2
max_deg_output_denominator = 3
max_deg_output_polynomials_specific = [1, 1, 1]
max_deg_output_polynomials_denominator_specific = [1, 1, 1]
width = 1
functions = []
function_names = []
maximal_potence = 3
maximal_n_functions = 1

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

    df = pd.read_csv(data_dir.joinpath(filename), sep=sep, header=None).iloc[:, :-1]
    print(df.shape)
    df = df.sample(10000)
    train_x = df.iloc[:, :-1].values.astype(float)
    train_y = df.iloc[:, -1].values

    model = ParFamWrapper(iterate=True, functions=functions, function_names=function_names)
    start_time = time.time()
    with suppress_stdout():
        model.fit(train_x, train_y)
    run_time = time.time() - start_time
    final_error = model.r_squared_val.item()
    print(f'Finished training for {filename}')
    return [model.formula, final_error, run_time]


if __name__ == '__main__':
    function_infos = pd.read_csv(dataset_dir.joinpath('Feynmann_laurent_polynomials.csv'), sep=';')
    data_dict = dict(zip(function_infos['Filename'], function_infos['Formula']))

    columns = ['Filename', 'Target Formula', 'Recovered Formula', 'Last Error', 'Run Time']
    result_table = pd.DataFrame(columns=columns)

    for name, target_formula in data_dict.items():
        test_result = [name, target_formula] + get_test_results_for_file(name)
        result_table.loc[len(result_table)] = test_result

    result_table.to_csv('parfam_feynmann_results.csv', index=False)
