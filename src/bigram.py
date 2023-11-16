# Import libraries
import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, context, targets=None):

        # this returns a torch.tensor with shape of (BATCH_SIZE, CONTEXT_LENGTH, VOCAB_SIZE)
        # e.g. (4, 8, 65)
        logits = self.embedding_table(context)

        if targets == None:
            loss = None

        else:
            B, C, V = logits.shape

            # we're going to strech out the array, new shape: (BATCH_SIZE * CONTEXT_LENGTH, VOCAB_SIZE)
            # e.g. (4*8, 65) == (32, 65)
            logits = logits.view(B * C, V)

            # and for the targets as well, we're going to change it's shape to be one dim
            # e.g. (32)
            targets = targets.view(B * C)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context, max_tokens, device="mps"):
        # Fist of all the context is with (B, C) shape
        for _ in range(max_tokens):
            # we get the prediction, the logits will be in (B, C, V) shape and the loss will be None
            logits, loss = self(context)
            # Focus only on the last character, this will change later
            logits = logits[:, -1, :]
            # get the probability distribution where the sum of probabilities are equal to 1
            probs = F.softmax(logits, dim=1)
            # get random sample distribution from the probability
            if device == "mps":
                next_token = torch.multinomial(probs.to("cpu"), num_samples=1).to(device)
            else:
                next_token = torch.multinomial(probs.to("cpu"), num_samples=1).to(device)
            # concatenate the generated token with the previous set of tokens
            context = torch.cat((context, next_token), dim=1)

        return context
