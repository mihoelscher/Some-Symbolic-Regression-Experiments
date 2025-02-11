from pathlib import Path
import pandas as pd

from tests.tests_on_feynman_datset.parfam_on_feynmann import get_test_results_for_file as runParfam

# Relative Paths
dataset_dir = Path.cwd().parent.parent.joinpath('Datasets')
data_dir = dataset_dir.joinpath('Feynman_with_units')
name = 'II.38.3'


if __name__ == '__main__':
    function_infos = pd.read_csv(dataset_dir.joinpath('Feynmann_laurent_polynomials.csv'), sep=';')
    data_dict = dict(zip(function_infos['Filename'], function_infos['Formula']))
    target_formula = data_dict.get(name, "Formula not found")

    columns = ['Filename', 'Target Formula', 'Recovered Formula', 'Last Error', 'Run Time']
    result_table = pd.DataFrame(columns=columns)

    test_result = [name, target_formula] + runParfam(name)
    result_table.loc[len(result_table)] = test_result

    result_table.to_csv('single_results.csv', index=False)
