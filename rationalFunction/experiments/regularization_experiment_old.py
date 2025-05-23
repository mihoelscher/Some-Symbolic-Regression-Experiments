import concurrent.futures
import sympy
import torch
from matplotlib import pyplot as plt
from rationalFunction.RationalSR import RationalFunction


def train_multiple_models_parallel(function_string, regularization_orders):
    def train_model(r):
        torch.manual_seed(12)
        _model = RationalFunction(2, 1)
        print(_model.coeffs_p, _model.coeffs_q)
        _model.fit(x_train, y_train, num_epochs=1000,
                           regularization_parameter=0.1,
                           regularization_order=r,
                           verbose=0)
        with torch.no_grad():
            scale_factor = 1/_model.coeffs_q[0].item()
            _model.coeffs_q.data *= scale_factor
            _model.coeffs_p.data *= scale_factor

        return _model, _model.losses

    x_train = torch.linspace(-0.8, 5, 1000).to('cpu')
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
    labels = ["No regularization", "l1", "l2"]
    colors = ["black", "r", "b"]

    for i, model in enumerate(_models):
        with torch.no_grad():
            print(f'Reg: {labels[i]} Recovered function: , {model.get_function()}')
        axes.plot(range(len(_losses[i])), _losses[i], label=labels[i], color=colors[i])

    axes.legend()
    return _fig, axes


if __name__ == '__main__':
    models, losses = train_multiple_models_parallel(f'(2*x**2 + x + 1)/(x + 1)', [None, 1, 2])
    fig, ax = plot_multiple_losses(models, losses)
    # fig.savefig('regularization_result.svg', format='svg')
    plt.show()
