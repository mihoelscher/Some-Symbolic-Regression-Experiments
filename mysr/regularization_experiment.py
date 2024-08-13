import concurrent.futures
import sympy
import torch
from matplotlib import pyplot as plt
from torch import optim
from RationalSR import RationalFunction


def train_multiple_models_parallel(function_string, regularization_orders):
    def train_model(r):
        torch.manual_seed(1337)
        _model = RationalFunction(3, 1)
        _loss = _model.fit(x_train, y_train, num_epochs=1000,
                           regularization_parameter=1,
                           regularization_order=r,
                           optimizer=optim.Adam(_model.parameters(), lr=0.01),
                           verbose=0)
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
    _fig, axes = plt.subplots(figsize=(16, 12))
    # _ax.plot(range(len(_losses)), _losses[0], label="Losses of different regularizations", color="r")
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.set_title('Training loss')
    print("Recovered function: ", _models[0].get_function())
    labels = ["l0", "l0.5", "l1", "l2"]
    colors = ["b", "g", "r", "black"]

    for i, model in enumerate(_models):
        with torch.no_grad():
            print("Learned coefficients for P(x): ", model.coeffs_p, "Learned coefficients for Q(x): ",
                  model.coeffs_q)
            print("Recovered function: ", model.get_function())
        axes.plot(range(len(_losses[i])), _losses[i], label=labels[i], color=colors[i])

    axes.legend()
    return _fig, axes


if __name__ == '__main__':
    models, losses = train_multiple_models_parallel(f'(2*x**2)/(x+3)', [0, 0.5, 1, 2])
    fig, ax = plot_multiple_losses(models, losses)
    # fig.savefig('regularization_result.svg', format='svg')
    plt.show()
