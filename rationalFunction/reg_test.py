import sympy
import torch
import matplotlib.pyplot as plt
from torch import optim

from rationalFunction.RationalSR import RationalFunction

if __name__ == '__main__':
    # We see that regularization has no impact at this point, since in comparison with the standard loss,
    # it is negligible. (loss
    seed = 1337
    function_string = f'(2*x**2)/(x+3)'
    x_train = torch.linspace(-2, 3, 1000).to('cpu')
    function = sympy.lambdify('x', sympy.sympify(function_string))
    y_train = function(x_train)

    torch.manual_seed(seed)
    model_none = RationalFunction(2, 1)
    loss_none = model_none.fit(x_train, y_train, num_epochs=1000,
                               regularization_parameter=0.1,
                               regularization_order=None,
                               optimizer=optim.Adam(model_none.parameters(), lr=0.01),
                               verbose=0)

    torch.manual_seed(seed)
    model_l1 = RationalFunction(2, 1)
    loss_l_1 = model_l1.fit(x_train, y_train, num_epochs=1000,
                            regularization_parameter=0.1,
                            regularization_order=1,
                            optimizer=optim.Adam(model_l1.parameters(), lr=0.01),
                            verbose=0)

    model_l2 = RationalFunction(2, 1)
    loss_l_2 = model_l2.fit(x_train, y_train, num_epochs=1000,
                            regularization_parameter=0.1,
                            regularization_order=2,
                            optimizer=optim.Adam(model_l2.parameters(), lr=0.01),
                            verbose=0)

    _fig, axes = plt.subplots(figsize=(16, 12))
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.set_title('Training loss')
    axes.plot(range(len(loss_none)), loss_none, label="no regularization", color="b")
    axes.plot(range(len(loss_l_1)), loss_l_1, label="l1 regularization", color="r")
    # axes.plot(range(len(loss_l_2)), loss_l_2, label="l1 regularization", color="black")
    axes.legend()
    print("No Reg: ", model_none.get_function())
    print("L1 Reg: ", model_l1.get_function())
    print("L2 Reg: ", model_l2.get_function())
    plt.show()
