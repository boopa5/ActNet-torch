import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import random

from typing import Any, Callable, Sequence, Tuple, Union, Literal
from functools import partial
import math

from tensor_layers_CoMERA.layers import wrapped_linear_layers, config_class

from .utils import *

identity = lambda x : x

###############################################################
######################## Architectures ########################
###############################################################


################
#### ActNet ####
################

# from https://www.wolframalpha.com/input?i=E%5B%28sin%28wx%2Bp%29%29%5D+where+x+is+normally+distributed
def _mean_transf(mu, sigma, w, p):
    return torch.exp(-0.5* (sigma*w)**2) * torch.sin(p + mu*w)

# from https://www.wolframalpha.com/input?i=E%5Bsin%28wx%2Bp%29%5E2%5D+where+x+is+normally+distributed
def _var_transf(mu, sigma, w, p):
    return 0.5 - 0.5*torch.exp(-2 * ((sigma*w)**2))*torch.cos(2*(p+mu*w)) - _mean_transf(mu, sigma, w, p)**2


class ActLayer(nn.Module):
    def __init__(
        self,
        in_dim : int,
        out_dim : int,
        num_freqs : int,
        use_bias : bool=True,
        # parameter initializers
        freqs_init : Callable=nn.init.normal_,  # normal entries w/ mean zero
        phases_init : Callable=nn.init.zeros_,
        beta_init : Callable=partial(nn.init.variance_scaling_, scale=1., mode='fan_in', distribution='uniform'),
        lamb_init : Callable=partial(nn.init.variance_scaling_, scale=1., mode='fan_in', distribution='uniform'),
        bias_init : Callable=nn.init.zeros_,
        # other configurations
        freeze_basis : bool=False,
        freq_scaling : bool=True,
        freq_scaling_eps : float=1e-3, # used for numerical stability of gradients
    ):
        super(ActLayer, self).__init__()

        self.out_dim = out_dim
        self.num_freqs = num_freqs
        self.use_bias = use_bias
        self.freqs_init = freqs_init
        self.phases_init = phases_init
        self.beta_init = beta_init
        self.lamb_init = lamb_init
        self.bias_init = bias_init
        self.freeze_basis = freeze_basis
        self.freq_scaling = freq_scaling
        self.freq_scaling_eps = freq_scaling_eps

        self.freqs = nn.Parameter(torch.zeros(1,1,num_freqs), requires_grad=not freeze_basis)
        freqs_init(self.freqs)

        self.phases = nn.Parameter(torch.zeros(1,1,num_freqs), requires_grad=not freeze_basis)
        phases_init(self.phases)

        self.beta = nn.Parameter(torch.zeros(num_freqs, out_dim))
        beta_init(self.beta)

        self.lamb = nn.Parameter(torch.zeros(in_dim, out_dim))
        lamb_init(self.lamb)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
            bias_init(self.bias)

    
    def forward(self, x):
        # x should initially be shape (batch, d)

        # perform basis expansion
        x = x.unsqueeze(2) # shape (batch, d, 1)
        x = torch.sin(self.freqs*x + self.phases) # shape (batch_dim, d, num_freqs)
        if self.freq_scaling:
            x = (x - _mean_transf(0., 1., self.freqs, self.phases)) / (torch.sqrt(self.freq_scaling_eps + _var_transf(0., 1., self.freqs, self.phases)))


        # combines lamb and beta into a single matrix 'aux'
        # 'aux' encodes out_dim outter products between rows of beta and columns of lamb
        # this is a batch-efficient way of carrying out the forward pass prescibed by the Kolmogorov representation
        # otherwise, for each element of the batch it would implicitly repeat several computations
        # (there are likely more elegant ways of doing this)
        # this whole block can also be implemented as x=torch.einsum('bij, jk, ik->bk', x, beta, lamb), but runs slower on my computer (maybe a JAX bug?)
        aux = torch.matmul(self.lamb.T[...,None], self.beta.T[:,None,:]) # shape (out_dim, d, num_freqs)
        aux = aux.reshape((self.out_dim,-1)) # shape (out_dim, d*num_freqs)
        aux = aux.T # shape (d*num_freqs, out_dim)
        x = x.reshape((x.shape[0], -1)) # shape (batch, d*num_freqs)
        x = torch.matmul(x, aux) # Shape (batch_size, out_dim)

        # optionally add bias
        if self.use_bias:
            x = x + self.bias # Shape (batch_size, out_dim)

        return x # Shape (batch_size, out_dim)
    

class ActNet(nn.Module):
    def __init__(self,
        embed_dim : int,
        num_layers : int,
        in_dim : int,
        out_dim : int,
        num_freqs : int,
        output_activation : Callable = identity,
        op_order : str='A', # string containing only 'A' (ActLayer), 'S' (Skip connection) or 'L' (LayerNorm) characters
        # op_order was used for development/debugging, but is not used in any experiment

        # parameter initializers
        freqs_init : Callable=nn.init.normal_,  # normal entries w/ mean zero
        phases_init : Callable=nn.init.zeros_,
        beta_init : Callable=partial(nn.init.variance_scaling_, scale=1., mode='fan_in', distribution='uniform'),
        lamb_init : Callable=partial(nn.init.variance_scaling_, scale=1., mode='fan_in', distribution='uniform'),
        act_bias_init : Callable=nn.init.zeros_,
        # proj_bias_init : Callable=lambda key, shape, dtype : random.uniform(key, shape, dtype, minval=-jnp.sqrt(3), maxval=jnp.sqrt(3)),
        proj_bias_init : Callable=partial(nn.init.uniform_, a=-math.sqrt(3), b=math.sqrt(3)),
        
        w0_init : Callable=partial(nn.init.constant_, val=30.), # following SIREN strategy
        w0_fixed : Union[Literal[False], float]=False,

        # other ActLayer configurations
        use_act_bias : bool=True,
        freeze_basis : bool=False,
        freq_scaling : bool=True,
        freq_scaling_eps : float=1e-3, # used for numerical stability of gradients
        tensorized : bool=False,
    ):
        super(ActNet, self).__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.num_freqs = num_freqs
        self.output_activation = output_activation
        self.op_order = op_order
        self.freqs_init = freqs_init
        self.phases_init = phases_init
        self.beta_init = beta_init
        self.lamb_init = lamb_init
        self.act_bias_init = act_bias_init
        self.proj_bias_init = proj_bias_init
        self.w0_init = w0_init
        self.w0_fixed = w0_fixed
        self.use_act_bias = use_act_bias
        self.freeze_basis = freeze_basis
        self.freq_scaling = freq_scaling
        self.freq_scaling_eps = freq_scaling_eps

        # initialize w0 parameter
        if self.w0_fixed is False:
            # trainable scalar parameter
            self.w0 = nn.Parameter(torch.zeros(1))
            w0_init(self.w0)
            # use softplus to ensure w0 is positive and does not decay to zero too fast (used only while debugging)

        else: # use user-specified value for w0
            self.w0 = self.w0_fixed

        # projection layers
        if not tensorized:
            self.proj_in = nn.Linear(in_dim, embed_dim, bias=True)
            proj_bias_init(self.proj_in.bias.data)
            self.proj_out = nn.Linear(embed_dim, out_dim, bias=True)
            proj_bias_init(self.proj_out.bias.data)
        else:
            r = 30
            rank_down, shape_down = rank_shape_lookup(in_dim, embed_dim, r)
            rank_up, shape_up = rank_shape_lookup(embed_dim, out_dim, r)

            build_rank_parameters = True
            set_scale_factors = False
            config_down = config_class(shape=shape_down,ranks=rank_down,set_scale_factors=set_scale_factors,build_rank_parameters=build_rank_parameters)
            config_up = config_class(shape=shape_up,ranks=rank_up,set_scale_factors=set_scale_factors,build_rank_parameters=build_rank_parameters)

            self.proj_in = wrapped_linear_layers(in_dim, embed_dim, tensorized=True, config=config_down)
            self.proj_out = wrapped_linear_layers(embed_dim, out_dim, tensorized=True, config=config_up)

        self.proj_out = nn.Linear(embed_dim, out_dim, bias=True)

        # initialize ActLayers
        act_layers = []
        for _ in range(self.num_layers):
            act_layers.append(ActLayer(
                in_dim = self.embed_dim,
                out_dim = self.embed_dim,
                num_freqs = self.num_freqs,
                use_bias = self.use_act_bias,
                freqs_init = self.freqs_init,
                phases_init = self.phases_init,
                beta_init = self.beta_init,
                lamb_init = self.lamb_init,
                bias_init = self.act_bias_init,
                freeze_basis = self.freeze_basis,
                freq_scaling = self.freq_scaling,
                freq_scaling_eps = self.freq_scaling_eps,
            ))
        self.act_layers = nn.ParameterList(act_layers)
        self.output_activation = output_activation

        self.register_forward_pre_hook(reshape_input_hook)
        self.register_forward_hook(reshape_output_hook)

    def forward(self, x):
        x = x * self.w0

        # project to embeded dimension
        x = self.proj_in(x)
        
        for act_layer in self.act_layers:
            x = act_layer(x)
        x = self.proj_out(x)

        x = self.output_activation(x)
        return x
