import re
import sympy
from sympy import simplify, symbols
import torch
import numpy as np
import matplotlib.pyplot as plt


def dataset_from_formula_string(formula: str, number_of_evals: int, low: float = 0.0, high: float = 1.0):
    try:
        formula = simplify(formula)
    except SyntaxError:
        print(f'The formula is not valid')
    parameters = formula.free_symbols
    # create data
    data = []
    for _ in range(number_of_evals):
        random_values = {symbol: np.random.uniform(low, high) for symbol in parameters}
        evaluation = formula.evalf(subs=random_values)
        row = [random_values[symbol] for symbol in parameters] + [evaluation]
        data.append(row)

    return data


def plot_data(inputs, outputs, color='blue'):
    input_dim = len(inputs[0])
    if input_dim == 1:
        # For 1D inputs, create a simple scatter plot
        inputs = [_input[0] for _input in inputs]  # Unpack the single-element lists
        plt.scatter(inputs, outputs, c=color, marker='o')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Input vs. Output')
    elif input_dim == 2:
        # For 2D inputs, create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = [_input[0] for _input in inputs]
        ys = [_input[1] for _input in inputs]
        ax.scatter(xs, ys, outputs, c=color, marker='o')
        ax.set_xlabel('Input 1')
        ax.set_ylabel('Input 2')
        ax.set_zlabel('Output')
        plt.title('Input 1 and Input 2 vs. Output')
    else:
        print("Plotting only works for 1D or 2D inputs")
    plt.show()


def tensor_to_rational_function(coefficients: torch.Tensor):
    coefficients = coefficients.cpu().numpy()
    numerator_coeffs, denominator_coeffs = coefficients[:coefficients.size//2], coefficients[coefficients.size//2:]

    _x = symbols('x')

    numerator = sum(coeff.round(decimals=5) * _x ** i for i, coeff in enumerate(reversed(numerator_coeffs)))
    denominator = sum(coeff.round(decimals=5) * _x ** i for i, coeff in enumerate(reversed(denominator_coeffs)))
    return simplify(numerator/denominator)


def rational_function_to_tensor(rational_func):
    try:
        numerator_coeffs, denominator_coeffs = [sympy.Poly(poly).all_coeffs() for poly in sympy.fraction(rational_func)]
    except sympy.polys.polyerrors.PolynomialError:
        print(f'Sympy Error: Please only use integers as exponents for your rational function')
        return
    len1, len2 = len(numerator_coeffs), len(denominator_coeffs)
    if len1 > len2:
        denominator_coeffs = [0] * (len1 - len2) + denominator_coeffs
    elif len2 > len1:
        numerator_coeffs = [0] * (len2 - len1) + numerator_coeffs

    return torch.Tensor(numerator_coeffs + denominator_coeffs)


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


def plot_predictions_2d(train_data, train_labels, test_data, test_labels, predictions=None):
    """
    Plots training data, test data, and compares predictions for 2D input.

    Args:
        train_data (array-like): Training data points with 2D input.
        train_labels (array-like): Labels for the training data.
        test_data (array-like): Test data points with 2D input.
        test_labels (array-like): Labels for the test data.
        predictions (array-like, optional): Predicted labels for the test data.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot training data in blue
    ax.scatter(train_data[:, 0], train_data[:, 1], train_labels, c="b", s=20, label="Training data")

    # Plot test data in green
    ax.scatter(test_data[:, 0], test_data[:, 1], test_labels, c="g", s=20, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        ax.scatter(test_data[:, 0], test_data[:, 1], predictions, c="r", s=20, label="Predictions")

    # Set labels
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Labels')

    # Show the legend
    ax.legend(prop={"size": 14})

    # Show the plot
    plt.show()


def plot_losses(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(losses)), losses, label="Loss", color="r")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.legend()
    # plt.savefig('test.svg', format='svg')
    plt.show()


def function_to_plot(target_function, predicted_function=None, x_min=-10, x_max=10, num_points=400):
    # Generate x values
    x_values = np.linspace(x_min, x_max, num_points)

    # Compute y values for the main function
    y_values = target_function(x_values)

    # Plot the main function
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label='Target Function', color='blue')

    # Optionally, plot the predicted_function if provided
    if predicted_function:
        y_predicted_values = predicted_function(x_values)
        plt.plot(x_values, y_predicted_values, label='Predicted Function', linestyle='--', color='red')

    # Add labels, title, grid, and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of Functions')
    plt.grid(True)
    plt.legend()
    plt.show()


def string_to_function(expression_str):
    variable_names = sorted(set(re.findall(r'x_\d+', expression_str)))
    variables = sympy.symbols(variable_names)
    replacements = {name: var for name, var in zip(variable_names, variables)}
    for name, var in replacements.items():
        expression_str = expression_str.replace(name, str(var))
    sympy_expr = sympy.sympify(expression_str)
    func_numeric = sympy.lambdify(variables, sympy_expr, modules=['numpy'])

    return func_numeric


def process_tensor_with_function(tensor, expression_str):
    function = string_to_function(expression_str)
    tensor_np = tensor.cpu().numpy()
    input_components = [tensor_np[:, i] for i in range(tensor_np.shape[1])]
    result_np = function(*input_components)
    result_tensor = torch.tensor(result_np, dtype=tensor.dtype)
    return result_tensor


def create_input_tensor(expression_str, input_dim, scalar=1):
    variable_names = sorted(set(re.findall(r'x_\d+', expression_str)))
    return torch.randn(input_dim, len(variable_names))*scalar


if __name__ == '__main__':
    my_function_str = '(x_0**3+x_1**2)/(x_1 + 1)'
    my_function = string_to_function(my_function_str)
    x = torch.ones((10, 2))
    print(process_tensor_with_function(x, '(x_0**3+x_1**2)/(x_1 + 1)'))
