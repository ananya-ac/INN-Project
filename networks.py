"""
Centralized network architecture definitions for Invertible Neural Networks.
This module contains reusable network components to eliminate code duplication.
"""

import torch.nn as nn
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, SequenceINN
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom, IResNetLayer


def subnet_fc(c_in, c_out, hidden_units=512):
    """
    Create a fully connected subnet for use in coupling blocks.

    Args:
        c_in (int): Input dimension
        c_out (int): Output dimension
        hidden_units (int): Number of hidden units (default: 512)

    Returns:
        nn.Sequential: Sequential network with GELU activations
    """
    net = nn.Sequential(
        nn.Linear(c_in, hidden_units),
        nn.GELU(),
        nn.Linear(hidden_units, hidden_units),
        nn.GELU(),
        nn.Linear(hidden_units, c_out)
    )
    return net


def get_inn(dim_tot, hidden_units=512, num_layers=8):
    """
    Build an Invertible Neural Network using GLOW-style coupling blocks.

    Args:
        dim_tot (int): Total dimension of the data
        hidden_units (int): Number of hidden units in coupling blocks (default: 512)
        num_layers (int): Number of coupling layers (default: 8)

    Returns:
        ReversibleGraphNet: Configured invertible network
    """
    nodes = [InputNode(dim_tot, name='input')]
    for k in range(num_layers):
        nodes.append(Node(nodes[-1],
                        GLOWCouplingBlock,
                        {'subnet_constructor': lambda c_in, c_out: subnet_fc(c_in, c_out, hidden_units), 'clamp': 2.0},
                        name=f'coupling_{k}'))
        nodes.append(Node(nodes[-1],
                        PermuteRandom,
                        {'seed': k},
                        name=f'permute_{k}'))
    nodes.append(OutputNode(nodes[-1], name='output'))
    model = ReversibleGraphNet(nodes, verbose=False)

    # Initialize parameters
    for param in model.parameters():
        if param.requires_grad:
            if param.dim() > 1:
                nn.init.orthogonal_(param, gain=0.8)
            else:
                nn.init.constant_(param, 0.0)
    return model


def get_iresnet(input_dim, num_blocks=8, hidden_size=128):
    """
    Set up an Invertible Residual Network (iResNet) using FrEIA framework.
    Uses actual residual blocks (i-ResNet) instead of coupling layers.

    Args:
        input_dim (int): Dimension of input data
        num_blocks (int): Number of invertible residual blocks (default: 8)
        hidden_size (int): Hidden layer size for residual blocks (default: 128)

    Returns:
        SequenceINN: Configured invertible residual network
    """
    # Initialize the sequential INN
    inn = SequenceINN(input_dim)

    # Add invertible residual blocks
    for i in range(num_blocks):
        inn.append(IResNetLayer, n_internal_layers=4, internal_size=hidden_size)

    return inn
