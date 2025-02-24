"""
Adapted from https://github.com/learningmatter-mit/NeuralForceField/blob/master/nff/utils/scatter.py
"""

from itertools import repeat
from torch.autograd import grad
import torch


def compute_grad(
    inputs,
    output,
    grad_outputs=None,
    create_graph=True,
    retain_graph=True,
    allow_unused=False,
):
    """Compute gradient of the scalar output with respect to inputs.

    Args:
        inputs (torch.Tensor): torch tensor, requires_grad=True
        output (torch.Tensor): scalar output

    Returns:
        torch.Tensor: gradients with respect to each input component
    """
    if not isinstance(inputs, torch.Tensor):
        for inp in inputs:
            assert inp.requires_grad
    else:
        assert inputs.requires_grad

    if grad_outputs is None:
        grad_outputs = output.data.new(output.shape).fill_(1)
    try:
        (gradspred,) = grad(
            output,
            inputs,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=allow_unused,
        )
    except:
        gradspred = grad(
            output,
            inputs,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=allow_unused,
        )

    return gradspred


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        dim_size = index.max().item() + 1 if dim_size is None else dim_size
        out_size = list(src.size())
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    return out.scatter_add_(dim, index, src)
