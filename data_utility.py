from sympy import simplify
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


def plot_data(data_array):
    inputs = [pair[:-1] for pair in data_array]
    outputs = [pair[-1] for pair in data_array]
    input_dim = len(inputs[0])
    if input_dim == 1:
        # For 1D inputs, create a simple scatter plot
        inputs = [x[0] for x in inputs]  # Unpack the single-element lists
        plt.scatter(inputs, outputs, c='blue', marker='o')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Input vs. Output')
    elif input_dim == 2:
        # For 2D inputs, create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = [x[0] for x in inputs]
        ys = [x[1] for x in inputs]
        ax.scatter(xs, ys, outputs, c='blue', marker='o')
        ax.set_xlabel('Input 1')
        ax.set_ylabel('Input 2')
        ax.set_zlabel('Output')
        plt.title('Input 1 and Input 2 vs. Output')
    else:
        print("Plotting only works for 1D or 2D inputs")
    plt.show()


if __name__ == '__main__':
    function = 'x_0**3'
    data = dataset_from_formula_string(function, 100, low=-3.0, high=3.0)
    print(data)
    plot_data(data)
