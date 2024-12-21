import random

import numpy as np
import torch
import tensorflow as tf
import torch.optim.lr_scheduler as lr_scheduler
from matplotlib import pyplot as plt

from de_learn_network import take_weights
from ginnlp import GINNLP
from pta_module import PTABlock, PTABlockWithLog


def train_model(model, inputs_x, y, num_epochs=500, lr=0.1):
    # Define a mean squared error loss function
    criterion = torch.nn.MSELoss()

    # Use an optimizer (e.g., Adam) to update model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define an exponential learning rate scheduler
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

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
        # Step the learning rate scheduler
        # scheduler.step()

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
    else:
        print("Loss did not reach zero during training.")


if __name__ == '__main__':
    # Generate evenly spaced training data
    num_samples = 1000
    seed = 1253
    torch.manual_seed(seed)
    x1 = torch.rand(num_samples, 1) * 5  # Random values for x1
    x2 = torch.rand(num_samples, 1) * 5  # Random values for x2

    # The target output is x1 * x2^2
    y = x1 * x2

    # Combine x1 and x2 into a single input tensor
    inputs_x = torch.cat([x1, x2], dim=1)  # Shape: (num_samples, 2)

    # Initialize the models
    model_pta = PTABlock(num_inputs=2, block_number=1)
    model_pta_log = PTABlockWithLog(num_inputs=2, block_number=1)
    ginnLP = GINNLP(num_epochs=500, round_digits=3, start_ln_blocks=1, growth_steps=0,
                    l1_reg=0, l2_reg=0, init_lr=0.01, decay_steps=1000, reg_change=0.5, train_iter=1)
    with torch.no_grad():
        learned_exponents = model_pta.exponents.clone()
        take_weights(np.array([learned_exponents.numpy()]).reshape(2,1))
        model_pta_log.ln_dense.weight = torch.nn.Parameter(learned_exponents.view_as(model_pta_log.ln_dense.weight))

    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print("\nTraining GinnLp")
    ginnLP.fit(inputs_x, y)
    ginnLP_losses = ginnLP.train_history[0].history['loss'] + ginnLP.train_history[1].history['loss']

    torch.manual_seed(seed)
    print("Training PTABlock...")
    train_model(model_pta, inputs_x, y)

    torch.manual_seed(seed)
    print("\nTraining PTABlockWithLog...")
    train_model(model_pta_log, inputs_x, y)

    # Plot the losses for both models
    plt.figure(figsize=(10, 6))
    plt.plot(model_pta.losses, label='PTABlock Loss')
    plt.plot(model_pta_log.losses, label='PTABlockWithLog Loss')
    plt.plot(ginnLP_losses, label='GINNLP Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Comparison seed: {seed} function: x1 * x2')
    plt.legend()
    plt.show()
