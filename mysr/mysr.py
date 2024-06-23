import numpy as np
import sympy
from sympy import symbols
import torch
from torch import nn
from torch.nn.modules.module import T


def evaluate_polynomial(coeffs, x):
    # Evaluate polynomial given coefficients and input x
    degree = len(coeffs)-1
    powers_of_x = torch.pow(x, torch.arange(degree, -1, -1))
    return torch.sum(torch.mul(coeffs, powers_of_x))


class MySR(torch.nn.Module):
    def __init__(self, nominator_degree=5, denominator_degree=5):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.numerator_coeffs = nn.Parameter(torch.randn(nominator_degree+1))
        self.denominator_coeffs = nn.Parameter(torch.randn(denominator_degree+1))

    def forward(self, x):
        numerator_value = evaluate_polynomial(self.numerator_coeffs, x)
        denominator_value = evaluate_polynomial(self.denominator_coeffs, x)
        return numerator_value / denominator_value

    def prune_coefficients(self):
        with torch.no_grad():
            self.linear.weight[torch.abs(self.linear.weight) < 0.1] = 0.0

    def fit(self,
            training_input,
            training_target,
            num_epochs=100,
            batch_size=1
            ):
        loss_fn = torch.nn.L1Loss(reduction='mean')
        optimizer = torch.optim.SGD(params=self.parameters(), lr=0.1)
        self.train()
        for epoch in range(num_epochs):
            prediction = self.forward(training_input).squeeze()
            loss = loss_fn(prediction, training_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
                print(self.state_dict())


def generate_dummy_data(num_samples):
    x = torch.randn(num_samples, 1)
    y = 2 * x + 1  # Example ground truth output (for demonstration)
    return x, y


if __name__ == '__main__':
    model = MySR(nominator_degree=2, denominator_degree=2)
    train_X, train_y = generate_dummy_data(10000)
    model.fit(train_X, train_y)
    print(model.state_dict())
