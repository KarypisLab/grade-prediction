"""
Sparsemax activation function.
Pytorch implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
"""

from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None, temp=1.0):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim
        self.temp = temp
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, input, mask=None):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        dim = self.dim
        if mask is not None:
            input = mask*input # (b, n)
            one_minus_mask = 1.0-mask
            max_mask = one_minus_mask*-9999999.9
            input = input+max_mask

        input = input.view(-1, input.size(self.dim))

        # multiply by the temperature parameter
        if self.temp != 0.0:
            input = input / (1.0-self.temp)

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        zs = torch.sort(input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits+1).view(1, -1).type(input.data.type()) # .to(self.device)
        range = Variable(range, requires_grad=False).expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.data.type()) # .to(self.device)
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)
        self.output = self.output*mask

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = self.dim

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input
