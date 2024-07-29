import concurrent.futures
import sympy
import torch
from matplotlib import pyplot as plt
from torch import optim
from RationalSR import RationalFunction


def train_multiple_models_parallel(function_string, regularization_orders):
    def train_model(r):
        torch.manual_seed(1337)
        _model = RationalFunction(2, 0)
        _loss = _model.fit(x_train, y_train, num_epochs=1000,
                           regularization_parameter=0.1,
                           regularization_order=r,
                           optimizer=optim.Adam(_model.parameters(), lr=0.01))
        return _model, _loss

    x_train = torch.linspace(-10, 10, 1000).to('cpu')
    function = sympy.lambdify('x', sympy.sympify(function_string))
    y_train = function(x_train)

    _models, _losses = [], []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_model, reg) for reg in regularization_orders]

        for future in concurrent.futures.as_completed(futures):
            model, loss = future.result()
            _models.append(model)
            _losses.append(loss)

    return _models, _losses


def plot_multiple_losses(_models, _losses):
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.plot(range(len(_losses)), _losses, label="Losses of different regularizations", color="r")
    _ax.set_xlabel('Epoch')
    _ax.set_ylabel('Loss')
    _ax.set_title('Training loss')
    _ax.legend()
    print("Recovered function: ", _models[0].get_function())
    labels = ["l0", "l0.5", "l1", "l2"]

    for i, model in enumerate(_models[1:]):
        with torch.no_grad():
            print("Learned coefficients for P(x): ", model.coeffs_p, "Learned coefficients for Q(x): ",
                  model.coeffs_q)
            print("Recovered function: ", model.get_function())
        _ax.plot(_losses[i + 1], _losses, label=labels[i + 1])
    return _fig, _ax


if __name__ == '__main__':
    models, losses = train_multiple_models_parallel(f'(2*x**2 + 3.141 * x + 3)', [0, 0.5, 1, 2])
    fig, ax = plot_multiple_losses(models, losses)
    plt.show()
