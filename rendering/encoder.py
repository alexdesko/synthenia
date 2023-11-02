import torch

class PositionalEncoder():
    """
    Positional encoder for the input tensor
    """
    def __init__(self, n_encoding_functions):
        self.n_encoding_functions = n_encoding_functions

    def __call__(self, x):
        encoded_tensor = [x]
        for i in range(self.n_encoding_functions):
            for func in [torch.sin, torch.cos]:
                encoded_tensor.append(func(x * torch.pi * (2 ** i)))
        return torch.cat(encoded_tensor, dim = -1)    
    

if __name__ == "__main__":
    x = torch.ones(size=(10,3))
    print(x)
    encoder = PositionalEncoder(6)
    print(encoder(x))