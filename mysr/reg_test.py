import sympy
import torch
import matplotlib.pyplot as plt
from torch import optim

from mysr.RationalSR import RationalFunction

if __name__ == '__main__':
    # We see that regularization has no impact at this point, since in comparison with the standard loss,
    # it is negligible. (loss
    function_string = f'(2*x**2 + 3.141 * x + 3)'
    x_train = torch.linspace(-10, 10, 1000).to('cpu')
    function = sympy.lambdify('x', sympy.sympify(function_string))
    y_train = function(x_train)

    torch.manual_seed(1337)
    model_none = RationalFunction(2, 0)
    loss_none = model_none.fit(x_train, y_train, num_epochs=500,
                               regularization_parameter=0.1,
                               regularization_order=None,
                               optimizer=optim.Adam(model_none.parameters(), lr=0.01))

    torch.manual_seed(1337)
    model_l1 = RationalFunction(2, 0)
    loss_l_1 = model_l1.fit(x_train, y_train, num_epochs=500,
                            regularization_parameter=0.1,
                            regularization_order=1,
                            optimizer=optim.Adam(model_l1.parameters(), lr=0.01))

    _fig, axes = plt.subplots(figsize=(16, 12))
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.set_title('Training loss')
    axes.plot(range(len(loss_none)), loss_none, label="no regularization", color="b")
    axes.plot(range(len(loss_l_1)), loss_l_1, label="l1 regularization", color="r")
    axes.legend()
    plt.show()
