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


def training_loop(model, data, epochs, optimizer, batch_size, context_length, device):
    training_loss = []
    model.to(device)
    for epoch in range(epochs):
        x, y = get_batch(data, batch_size, context_length)
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())

    return training_loss

