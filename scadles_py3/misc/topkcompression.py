import torch

from scadles_py3.misc import Compressor, Memory

class ResidualMemory(Memory):
    def __init__(self, beta=1.0):
        self.residuals = {}
        self.beta = beta
        #self.gamma = gamma
        self.layer_decompress = {}

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if name in self.residuals:
            tensor = self.beta * self.residuals[name] + tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        # Aug 10, 2022: @styagi removing residuals from adaptive compression
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        self.layer_decompress[name] = tensor_decompressed
        residual = tensor - tensor_decompressed
        self.residuals[name] = residual

def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    return values, indices

    # k = max(1, int(tensor.numel() * compress_ratio))
    # values, indexes = tensor.abs().sort(descending=True)
    # return values[:k], indexes[:k]

def desparsify(tensors, numel, device):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed


class TopKCompressor(Compressor):

    def __init__(self, device):
        super().__init__()
        self.residual = ResidualMemory()
        self.device = device

    def compress(self, tensor, name, compress_ratio):
        tensor = tensor.to(self.device)
        
        tensor = self.residual.compensate(tensor, name)
        numel = tensor.numel()
        shape = tensor.size()
        tensors = sparsify(tensor, compress_ratio)
        ctx = numel, shape
        self.residual.update(tensor, name, self, tensors, ctx)
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = desparsify(tensors, numel, self.device)
        return tensor_decompressed.view(shape)