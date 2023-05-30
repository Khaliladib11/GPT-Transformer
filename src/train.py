import argparse
from utils import *
from bigram import BigramLanguageModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=32, help="Total batch size for all GPUs")
    parser.add_argument("--data-path", type=str, default='../input.txt', help="The path to the data.")
    parser.add_argument("--lr", type=float, default=1e-2, help="The learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="total number of epochs")
    parser.add_argument("--context-length", type=int, default=8, help="The context window")
    parser.add_argument("--train-percentage", type=float, default=0.8, help="The split percentage for training data")
    parser.add_argument("--eval-percentage", type=float, default=0.1, help="The split percentage for evaluation data")
    parser.add_argument("--test-percentage", type=float, default=0.1, help="The split percentage for testing data")
    parser.add_argument("--device", type=str, default="cpu", help="The device where we want to train the model")

    # Fetch the params from the parser
    args = parser.parse_args()

    data_path = args.data_path
    batch_size = args.batch_size  # Batch Size
    context_length = args.context_length  # Context length
    lr = args.lr  # Learning Rate
    epochs = args.epochs  # number of epochs

    train_percentage = args.train_percentage
    eval_percentage = args.eval_percentage
    test_percentage = args.test_percentage

    device = args.device

    text = read_data(data_path)

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    c2i, i2c = create_dictionaries(chars)

    encoder = lambda string: [c2i[char] for char in string]
    decoder = lambda indexes: "".join([i2c[idx] for idx in indexes])

    data = torch.tensor(encoder(text), dtype=torch.long)

    train_set, eval_set, test_set = split_data(data, train_percentage, eval_percentage, test_percentage)

    bigram_model = BigramLanguageModel(vocab_size)
    bigram_model = bigram_model.to(device)
    # define an optimizer
    optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=lr)

    training_loss = training_loop(bigram_model, train_set, epochs, optimizer, batch_size, context_length, device)

    # Test
    test_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_results = bigram_model.generate(test_idx, max_tokens=50)[0].tolist()
    print(generated_results)
    print(decoder(generated_results))

