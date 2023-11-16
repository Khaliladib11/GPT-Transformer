import matplotlib.pyplot as plt

import torch


def read_data(data_path):
    with open(data_path, 'r') as file:
        data = file.read()

    return data


def create_dictionaries(chars):
    # Create two ways mapping
    c2i = {char: idx for idx, char in enumerate(chars)}
    i2c = {idx: char for idx, char in enumerate(chars)}

    return c2i, i2c


def split_data(data, train_percent, eval_percent, test_percent):
    assert train_percent + eval_percent + test_percent == 1.0, f"The summation of all percentags must be 1.0, we got {train_percent + eval_percent + test_percent}"

    train_range = [0, int(len(data) * train_percent)]
    eval_range = [train_range[1], int(len(data) * eval_percent) + train_range[1]]
    test_range = [eval_range[1], -1]

    return data[:train_range[1]], data[eval_range[0]:eval_range[1]], data[test_range[0]:]


def get_batch(data, batch_size, context_length):
    indexes = torch.randint(len(data) - batch_size, (batch_size,))  # get BATCH_SIZE random indexes within the dataset

    # loop through each index and get the context data and finally stack them together to get torch.tensor of shape (
    # BATCH_SIZE, CONTEXT_LENGTH)
    x = torch.stack([data[idx: idx + context_length] for idx in indexes])
    # loop through each index and get the target data and finally stack them together to get torch.tensor of shape (
    # BATCH_SIZE, CONTEXT_LENGTH)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in indexes])

    return x, y


def training_loop(model, train_data, eval_data, epochs, eval_interval, optimizer, batch_size, context_length, device):
    
    model.to(device)
    model.train()
    training_loss = []
    eval_loss = []
    
    for epoch in range(epochs):
        if epoch % eval_interval == 0:
            losses = validation(model, train_data, eval_data, 100, device)
            training_loss.append(losses['train'])
            eval_loss.append(losses['eval'])
            print(f"Epoch: {epoch}, Training Loss: {losses['train']}, Eval Loss: {losses['eval']}")
    
        x, y = get_batch(train_data, batch_size, context_length)
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return training_loss, eval_loss

@torch.no_grad()
def validation(model, train_data, eval_data, eval_epoch, device="mps"):
    output = {}
    model.eval()
    model.to(device)

    # train data
    losses = torch.zeros(eval_epoch)
    for k in range(eval_epoch):
        X, y = get_batch(train_data)
        X, y = X.to(device), y.to(device)
        logits, loss = model(X, y)
        losses[k] = loss.item()

    output['train'] = losses.mean()

    # eval data
    losses = torch.zeros(eval_epoch)
    for k in range(eval_epoch):
        X, y = get_batch(eval_data)
        X, y = X.to(device), y.to(device)
        logits, loss = model(X, y)
        losses[k] = loss.item()

    output['eval'] = losses.mean()

    model.train()
    return output

    
def plot_loss(train_loss_values, val_loss_values):
    """
    Plot the training and validation loss values over training epochs with a line plot.

    Parameters:
    - train_loss_values (list): List of training loss values at each epoch.
    - val_loss_values (list): List of validation loss values at each epoch.
    """
    epochs = range(1, len(train_loss_values) + 1)

    # Plotting the training loss values with a line plot
    plt.plot(epochs, train_loss_values, label='Training Loss', marker='o', linestyle='-')

    # Plotting the validation loss values with a line plot
    plt.plot(epochs, val_loss_values, label='Validation Loss', marker='o', linestyle='-')

    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Display the plot
    plt.show()
    """
    Plot the loss values over training epochs with a line plot.

    Parameters:
    - loss_values (list): List of loss values at each training epoch.
    """
    epochs = range(1, len(loss_values) + 1)

    # Plotting the loss values with a line plot
    plt.plot(loss_values, label='Training Loss')

    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Display the plot
    plt.show()