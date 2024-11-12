import os

import pandas as pd
import numpy as np
import time
from pysr import PySRRegressor
import sympy as sp
import torch
from sklearn.metrics import mean_squared_error
from sympy import false
from torch.backends.mkl import verbose


def save_to_csv(df, file_path):
    """Save the comparison DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")


class ModelComparison:
    def __init__(self, ginnLP, pysr: PySRRegressor):
        self.ginnLP = ginnLP
        self.pysr = pysr

    def train_and_compare(self, X, y, target):
        results = {
            'target': target
        }

        # Train and evaluate GINNLP model
        start_time = time.time()
        self.ginnLP.fit(X, y)
        ginnLP_time = time.time() - start_time
        ginnLP_predictions = self.ginnLP.predict(X)
        ginnLP_mse = mean_squared_error(y, ginnLP_predictions)

        # Store GINNLP results
        results['ginnLP_output'] = self.ginnLP.recovered_eq
        results['ginnLP_time'] = ginnLP_time
        results['ginnLP_mse'] = ginnLP_mse

        # Train and evaluate PySR model
        start_time = time.time()
        self.pysr.fit(X, y)
        pysr_time = time.time() - start_time
        pysr_predictions = self.pysr.predict(X)
        pysr_mse = mean_squared_error(y, pysr_predictions)

        # Store PySR results
        results['pysr_output'] = self.pysr.sympy()
        results['pysr_time'] = pysr_time
        results['pysr_mse'] = pysr_mse

        # Convert results to DataFrame
        comparison_df = pd.DataFrame(results)
        return comparison_df


if __name__ == '__main__':

    file = "comparison_table.csv"
    if os.path.exists(file):
        comparison_table = pd.read_csv(file)
    else:
        comparison_table = pd.DataFrame()


    x_train = torch.linspace(1, 5, 101).to('cpu').unsqueeze(0).T
    function = '2*x**2 - 5 * x'
    target_function = sp.lambdify('x', function)
    y_train = target_function(x_train)

    from ginnlp import GINNLP
    ginnLP = GINNLP(num_epochs=500, round_digits=3, start_ln_blocks=1, growth_steps=3,
                    l1_reg=1e-4, l2_reg=1e-4, init_lr=0.01, decay_steps=1000, reg_change=0.5)


    model = PySRRegressor(
        niterations=40,  # < Increase me for better results
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
    )

    comparison = ModelComparison(ginnLP, pysr=model)
    comparison_df = comparison.train_and_compare(x_train, y_train, target_function)
    pd.concat([comparison_table, comparison_df]).to_csv(file, index=False)




# Usage example:
# Initialize the models as per the provided setup
# ginnLP = GINNLP(num_epochs=500, ...)
# pysr = PySRRegressor(niterations=40, ...)

# Instantiate the comparison class with the models
# comparison = ModelComparison(ginnLP, pysr)

# Define your input data X and target y
# X = np.array(...)  # Input features
# y = np.array(...)  # Target values

# Run the comparison
# comparison_df = comparison.train_and_compare(X, y)
# comparison.save_to_csv(comparison_df, 'model_comparison_results.csv')
