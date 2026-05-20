from typing import Tuple, Union, Callable
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
# Common functions for all neural networks.                                    #
################################################################################

def get_act_fn(activation_name, **kwargs):
    if not isinstance(activation_name, str) and hasattr(activation_name, "__call__"):
        return activation_name
    if activation_name == "relu":
        return nn.ReLU
    elif activation_name == "elu":
        return nn.ELU
    elif activation_name == "lrelu":
        return functools.partial(nn.LeakyReLU, negative_slope=0.1)  # nn.LeakyReLU
    elif activation_name == "prelu":
        return nn.PReLU
    elif activation_name in ("swish", "silu"):
        return nn.SiLU
    elif activation_name == "gelu":
        return nn.GELU
    else:
        raise KeyError(activation_name)


def get_num_params(net):
    num_params = 0
    if hasattr(net, "parameters"):
        for parameter in net.parameters():
            if hasattr(parameter, "numel"):
                num_params += parameter.numel()
    return num_params


def get_device(name, load=False) -> (torch.device, dict[str,torch.device]):

    if "cuda" in name:
        if torch.cuda.is_available():
            device = torch.device(name)
            map_location_args = {"map_location" : device}
        else:
            raise OSError(f"No available device for device spec '{name}'.")

    elif name == "cpu":
        device = torch.device("cpu")
        map_location_args = {"map_location": device}

    else:
        raise ValueError(f"No device for spec '{name}'")


    if load:
        return device, map_location_args
    else:
        return device, {}


def scatter(src, index, dim_size=None, reduce="sum"):
    """
    A simple Python implementation of 1D scatter operations.
    
    Args:
    src (list or 1D array): Source data.
    index (list or 1D array): Index positions for scattering the src data.
    dim_size (int): Size of the output tensor.
    
    Returns:
    list: Result of the scatter operation.
    """
    # Check the input.
    if not reduce in ("sum", "mean"):
        raise KeyError(reduce)
    if src.shape[0] != index.shape[0]:
        raise ValueError(
            "Dimensions are not matching: src={}, index={}".format(
            src.shape[0], index.shape[0]))
    i_max = index.max()
    if dim_size is None:
        dim_size = i_max+1
    else:
        if dim_size < i_max+1:
            raise IndexError(
                f"Maximum index is {i_max} but a dim_size of {dim_size} was"
                " provided"
            )
    # Initialize the output tensor with zeros
    if len(src.shape) > 1:
        extra_dim = src.shape[1:]
    else:
        extra_dim = ()
    output = torch.zeros(dim_size, *extra_dim, dtype=src.dtype)
    counts = torch.zeros(dim_size, dtype=src.dtype)
    # Iterate over all the indices.
    use_counts = reduce == "mean"
    for i in range(dim_size):
        mask_i = index == i
        output[i] = src[mask_i].sum(dim=0)
        if use_counts:
            counts[i] = mask_i.sum()
    # Return the output.
    if reduce == "sum":
        return output
    elif reduce == "mean":
        return output/counts
    else:
        raise KeyError(reduce)


################################################################################
# Common modules.                                                              #
################################################################################

class MLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = None,
                 activation: Callable = nn.ReLU,
                 n_hidden_layers: int = 1,
                 # final_activation: bool = False,
                 final_norm: bool = False):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else out_dim
        layers = [nn.Linear(in_dim, hidden_dim),
                  activation()]
        for l in range(n_hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim),
                           activation()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        # if final_activation:
        #     layers.append(activation)
        if final_norm:
            layers.append(nn.LayerNorm(out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class FeedForward(nn.Module):
    """New module documentation: TODO."""

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int = None,
            out_dim: int = None,
            activation: Union[Callable, str] = "silu",
            linear_bias: bool = True
        ):
        """Arguments: TODO."""
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        hidden_dim = hidden_dim if hidden_dim is not None else in_dim
        if activation == "swiglu":
            self.main = SwiGLU(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                linear_bias=linear_bias
            )
        else:
            act_cls = get_act_fn(activation)
            self.main = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=linear_bias),
                act_cls(),
                nn.Linear(hidden_dim, out_dim, bias=linear_bias)
            )

    def forward(self, x):
        return self.main(x)


class SwiGLU(nn.Module):
    """New module documentation: TODO."""

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int = None,
            linear_bias: bool = True
        ):
        """Arguments: TODO."""
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        self.a_linear = nn.Linear(in_dim, hidden_dim, bias=linear_bias)
        self.a_act = nn.SiLU()
        self.b_linear = nn.Linear(in_dim, hidden_dim, bias=linear_bias)
        self.o_linear = nn.Linear(hidden_dim, out_dim, bias=linear_bias)

    def forward(self, x):
        a = self.a_act(self.a_linear(x))
        b = self.b_linear(x)
        x = self.o_linear(a * b)
        return x


# class SmoothConv(nn.Module):
    
#     def __init__(self, probs, n_channels=1):
#         """
#         `probs`: example [0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]
#         """
#         super().__init__()
#         # Create kernels.
#         kernel = torch.FloatTensor([[probs]])
#         kernel = kernel.repeat(n_channels, 1, 1)
#         self.register_buffer('kernel', kernel)
#         self.n_channels = n_channels
        
#     def forward(self, x):
#         # Apply smoothing.
#         x = F.conv1d(x, self.kernel, padding="same", groups=self.n_channels)
#         return x

def get_layer_operation_arch(num_layers, operation_freq, operation):
    operation_arch = [None for l in range(num_layers)]
    if operation_freq is None:
        return operation_arch
    if operation is not None:
        if isinstance(operation_freq, int):
            for l in range(num_layers):
                if l % operation_freq == 0:
                    operation_arch[l] = operation
        elif isinstance(operation_freq, (tuple, list)):
            for l in operation_freq:
                operation_arch[l] = operation
        else:
            raise TypeError(type(operation_freq))
    return operation_arch


# ################################################################################
# # GNN classes.                                                                 #
# ################################################################################

# class UpdaterModule(nn.Module):

#     def __init__(self, gnn_layer, in_dim,
#                  dim_feedforward, activation,
#                  dropout=0.0,
#                  layer_norm_eps=1e-5):
#         super(UpdaterModule, self).__init__()
#         self.gnn_layer = gnn_layer
#         self.linear1 = nn.Linear(in_dim, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, in_dim)
#         self.use_norm = layer_norm_eps is not None
#         if self.use_norm:
#             self.norm1 = nn.LayerNorm(in_dim, eps=layer_norm_eps)
#             self.norm2 = nn.LayerNorm(in_dim, eps=layer_norm_eps)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = activation
#         self.update_module = nn.Sequential(self.linear1,
#                                            self.activation,
#                                            self.linear2)

#     def forward(self, h, edge_index, edge_attr=None):
#         h2 = self.gnn_layer(h, edge_index, edge_attr=edge_attr)
#         h = h + self.dropout1(h2)
#         if self.use_norm:
#             h = self.norm1(h)
#         # s2 = self.linear2(self.dropout(self.activation(self.linear1(s))))
#         h2 = self.update_module(h)
#         h = h + self.dropout2(h2)
#         if self.use_norm:
#             h = self.norm2(h)
#         return h


# class EdgeUpdaterModule(nn.Module):

#     def __init__(self, gnn_layer, in_edge_dim, out_edge_dim, activation):
#         super(EdgeUpdaterModule, self).__init__()
#         self.gnn_layer = gnn_layer
#         self.in_edge_dim = in_edge_dim
#         self.out_edge_dim = out_edge_dim
#         self.activation = activation

#         self.edge_mlp = nn.Sequential(nn.Linear(in_edge_dim, out_edge_dim),
#                                       activation,
#                                       nn.Linear(out_edge_dim, out_edge_dim))

#     def forward(self, h, edge_index, edge_attr):
#         edge_attr = self.edge_mlp(edge_attr)
#         return self.gnn_layer(h, edge_index, edge_attr=edge_attr)


################################################################################
# Positional embeddings.                                                       #
################################################################################

class AF2_PositionalEmbedding(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
                 pos_embed_dim: int,
                 pos_embed_r: int = 32,
                 dim_order: str = "transformer"
                ):
        super().__init__()
        self.embed = nn.Embedding(pos_embed_r*2+1, pos_embed_dim)
        self.pos_embed_r = pos_embed_r
        self.set_dim_order(dim_order)

    def set_dim_order(self, dim_order):
        self.dim_order = dim_order
        if self.dim_order == "transformer":
            self.l_idx = 0  # Token (residue) index.
            self.b_idx = 1  # Batch index.
        elif self.dim_order == "trajectory":
            self.l_idx = 1
            self.b_idx = 0
        else:
            raise KeyError(dim_order)

    def forward(self, x, r=None):
        """
        x: xyz coordinate tensor of shape (L, B, *) if `dim_order` is set to
            'transformer'.
        r: optional, residue indices tensor of shape (B, L).

        returns:
        p: 2d positional embedding of shape (B, L, L, `pos_embed_dim`).
        """
        if r is None:
            prot_l = x.shape[self.l_idx]
            p = torch.arange(0, prot_l, device=x.device)
            p = p[None,:] - p[:,None]
            bins = torch.linspace(-self.pos_embed_r, self.pos_embed_r,
                                  self.pos_embed_r*2+1, device=x.device)
            b = torch.argmin(
                torch.abs(bins.view(1, 1, -1) - p.view(p.shape[0], p.shape[1], 1)),
                axis=-1)
            p = self.embed(b)
            p = p.repeat(x.shape[self.b_idx], 1, 1, 1)
        else:
            b = r[:,None,:] - r[:,:,None]
            b = torch.clip(b, min=-self.pos_embed_r, max=self.pos_embed_r)
            b = b + self.pos_embed_r
            p = self.embed(b)
        return p


################################################################################
# Attention-based.                                                             #
################################################################################

class TransformerBlock_v01(nn.Module):
    """Transformer layer block for the AA AE."""

    def __init__(
            self,
            embed_dim: int,
            edge_dim: int,
            mlp_dim: int,
            num_heads: int,
            activation: Union[Callable, str] = "silu",
            d_model: int = None,
            linear_bias: bool = True,
            add_bias_2d: bool = True,
            # add_bias_kv: bool = True,
            # attention_type: str = "transformer"
        ):

        ### Initialize and store the attributes.
        super().__init__()

        if d_model is None:
            d_model = embed_dim
        else:
            raise NotImplementedError()

        ### Transformer layer.
        self.attn_norm = nn.LayerNorm(
            embed_dim,
            elementwise_affine=True
        )
        # Actual transformer layer.
        self.self_attn = PyTorchAttentionLayer_v01(
            embed_dim=embed_dim,
            num_heads=num_heads,
            edge_dim=edge_dim,
            add_bias_kv=False,
            add_bias_2d=add_bias_2d,
            dropout=0.0
        )
        
        ### MLP.
        self.final_norm = nn.LayerNorm(
            embed_dim,
            elementwise_affine=True
        )
        self.feedforward = FeedForward(
            in_dim=embed_dim,
            hidden_dim=mlp_dim,
            activation=activation,
            linear_bias=linear_bias
        )

    def forward(self, x, p):
        # Attention mechanism.
        residual = x
        x = self.attn_norm(x)
        x = self.self_attn(x, x, x, p=p)[0]
        x = residual + x
        # MLP update.
        residual = x
        x = self.final_norm(x)
        x = self.feedforward(x)
        x = residual + x
        return x


class PyTorchAttentionLayer_v01(nn.Module):
    """New module documentation: TODO."""

    def __init__(
            self,
            embed_dim,
            num_heads,
            edge_dim,
            add_bias_kv=False,
            add_bias_2d=True,
            dropout=0.0
        ):
        """Arguments: TODO."""

        super().__init__()

        # Attributes to store.
        self.num_heads = num_heads

        # Multihead attention.
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=True
        )

        # Project edge features.
        self.edge_to_bias = nn.Linear(edge_dim, num_heads, bias=add_bias_2d)

    def forward(self, q, k, v, p):
        b_size = q.shape[0]
        seq_l = q.shape[1]
        p = self.edge_to_bias(p)
        p = p.transpose(1, 3).transpose(2, 3)
        p = p.contiguous().view(b_size*self.num_heads, seq_l, seq_l)
        out = self.mha(q, k, v, attn_mask=p, need_weights=False)
        return out
