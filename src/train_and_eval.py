from tokenizer import CharacterTokenizer
from model import LanguageModel
from utils import get_batch
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)

tokenizer = CharacterTokenizer(vocab)

data = torch.tensor(data=tokenizer.encode(text), dtype=torch.int64)
train_data = data[:int(len(data) * 0.9)]
valid_data = data[int(len(data) * 0.9):]

model = LanguageModel(
    vocab_size=vocab_size,
    d_emb=64,
    d_head=16,
    d_mlp=256,
    max_context_length=64,
    num_heads=4,
    num_blocks=3,
)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

for i in range(10000):

    x, y = get_batch(
        data=train_data,
        num_batches=16,
        context_length=64
    )
    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if (i + 1) % 500 == 0:
        print(f'Iteration {i + 1}, Loss = {loss}')

print('TRAINING FINISHED!')


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(context, max_tokens_generated=5000)[0].tolist()))


# @torch.no_grad()
# def estimate_loss(model, num_iters, data, num_batches, context_length, device='cpu'):
#     model.eval()
#     losses = torch.zeros(num_iters)
#     for i in range(num_iters):
#         train_x, train_y = get_batch(
#             data=train_data, 
#             num_batches=16,

#         )
