import concurrent.futures

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
import sympy as sp

from pta_module import PTABlock, PTABlockWithLog

def train_model(model, inputs_x, y, num_epochs=5000, lr=0.01):
    # Define a mean squared error loss function
    criterion = torch.nn.MSELoss()

    # Use an optimizer (e.g., Adam) to update model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_threshold = 1e-5  # You can set a threshold close to zero
    first_zero_epoch = None  # Initialize to None to indicate it hasn't been reached

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        # Forward pass: Compute the model's prediction
        predictions = model(inputs_x)

        # Compute the loss
        loss = criterion(predictions, y)
        model.losses.append(loss.item())

        # Check if the loss has gone below the threshold for the first time
        if first_zero_epoch is None and loss.item() <= loss_threshold:
            first_zero_epoch = epoch + 1  # Store the epoch (1-indexed)

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero out the gradients from the previous step
        loss.backward()  # Backpropagate to compute gradients
        optimizer.step()  # Update the parameters

        # Print the loss every 100 epochs for monitoring
        if (epoch + 1) % 100 == 0 or epoch == 0:
            if epoch == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Print final results after training
    print(f"Learned exponents for {model.__class__.__name__}:", model.exponents.detach().numpy())
    print(f"Final loss for {model.__class__.__name__}: {loss.item():.4f}")

    if first_zero_epoch is not None:
        print(
            f"Loss reached zero (or below threshold {loss_threshold}) for the first time at epoch {first_zero_epoch}.")
        return first_zero_epoch
    else:
        print("Loss did not reach zero during training.")
        return 10e10


def train_and_evaluate(formula_str, num_samples, seed):
    torch.manual_seed(seed)

    # Define symbolic variables
    x1, x2 = sp.symbols('x1 x2')
    formula = sp.sympify(formula_str)  # Convert string to sympy expression

    # Generate random input values
    x1_vals = torch.rand(num_samples, 1) * 5
    x2_vals = torch.rand(num_samples, 1) * 5

    # Convert sympy expression to lambda function
    formula_func = sp.lambdify((x1, x2), formula, 'numpy')
    y_vals = torch.tensor(formula_func(x1_vals.numpy(), x2_vals.numpy()), dtype=torch.float32)

    inputs_x = torch.cat([x1_vals, x2_vals], dim=1)

    # Initialize models
    model_pta = PTABlock(num_inputs=2, block_number=1)
    model_pta_log = PTABlockWithLog(num_inputs=2, block_number=1)

    with torch.no_grad():
        learned_exponents = model_pta.exponents.clone()
        model_pta_log.ln_dense.weight = torch.nn.Parameter(learned_exponents.view_as(model_pta_log.ln_dense.weight))

    torch.manual_seed(seed)
    print(f"Training PTABlock for formula: {formula_str}, seed: {seed}")
    res1 = train_model(model_pta, inputs_x, y_vals)

    torch.manual_seed(seed)
    print(f"\nTraining PTABlockWithLog for formula: {formula_str}, seed: {seed}")
    res2 = train_model(model_pta_log, inputs_x, y_vals)

    if res1 < res2:
        winner = "PTA"
    elif res1 > res2:
        winner = "PTA with log"
    else:
        winner = "Fail"

    return [winner, res1, res2]

def train_and_evaluate_parallel(formula, seed, num_samples):
    return [formula, seed] + train_and_evaluate(formula, num_samples, seed)

if __name__ == '__main__':
    formulas = ["x1**2 * x2", "x1 / x2", "x1**3/x2"]  # Example formulas
    seeds = [6, 42, 99]  # Example seeds [random.randint(0, 1000) for _ in range(100)]
    num_samples = 1000
    columns = ['Target Formula', 'Seed', 'Winner', 'Last Epoch PTA', 'Last Epoch PTA with Log']
    result_table = pd.DataFrame(columns=columns)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda args: train_and_evaluate_parallel(*args),
                                    [(formula, seed, num_samples) for formula in formulas for seed in seeds]))

    result_table = pd.DataFrame(results, columns=columns)
    print(result_table)