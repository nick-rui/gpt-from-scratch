import torch

def get_batch(data, num_batches, context_length, device='cpu'):
    '''
    Returns a batch of training/testing examples:
    
    x: (num_batches, context_length)
    y: (num_batches, context_length)

    For the i-th training example in batch b,
        the input (context) is x[b, :i]
        the output (label) is y[b, i]
    
    :param data: Large sequence of tokens to sample from for training
    :param num_batches: Number of batches to sample
    :param context_length: Number of tokens per sample
    '''
    n = len(data)
    random_offsets = torch.randint(low=0, high=n - context_length, size=(num_batches,))
    x = torch.stack([data[i:i + context_length] for i in random_offsets])
    y = torch.stack([data[i + 1:i + context_length + 1] for i in random_offsets])
    x, y = x.to(device), y.to(device)
    return x, y



