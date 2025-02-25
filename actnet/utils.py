def rank_shape_lookup(input_dim, output_dim, r=20):
    input_shape_dict = {
        8: [2, 2, 2],
        16: [4, 2, 2],
        32: [4, 4, 2],
        64: [4, 4, 4],
        128: [8, 4, 4],
        256: [8, 8, 4],
        512: [8, 8, 8],
        768: [12, 8, 8],
        1024: [16, 8, 8],
        3072: [16, 16, 12]
    }

    input_shape = input_shape_dict[input_dim]
    output_shape = input_shape_dict[output_dim][::-1]
    shape = input_shape + output_shape

    rank = [1] + [input_shape[0]] + ([r] * (len(shape) - 3)) + [output_shape[-1]] + [1]

    if input_dim == 256 and output_dim == 256:
        rank = [1, 8, r, r, r, 8, 1]
        shape = [8, 8, 4, 4, 8, 8]
    elif input_dim == 256 and output_dim == 1024:
        rank = [1, 8, r, r, r, 16, 1]
        shape = [8, 8, 4, 8, 8, 16]
    elif input_dim == 1024 and output_dim == 256:
        rank = [1, 16, r, r, r, 8, 1]
        shape = [16, 8, 8, 4, 8, 8]

    return rank, shape


def split_tensor_rank_params(model):

    par_tensor = []
    par_origin = []
    par_rank = []

    for name,par in model.named_parameters():
        if 'tensor.factors' in name:
            par_tensor.append(par)
        elif 'tensor.rank_parameters' in name:
            par_rank.append(par)
        else:
            par_origin.append(par)
    
    return par_tensor, par_origin, par_rank


# add hook to reshape batch to first dim
def reshape_input_hook(module, input):
    input_ = input[0]

    if input_.dim() > 2:
        # print("input shape", input_.shape)
        hidden_size = input_.shape[-1]
        module.B = input_.shape[0]
        input_ = input_.reshape(-1, hidden_size)
        # print("new input shape", input_.shape)
        input = (input_,)
    return input


def reshape_output_hook(module, input, output):

    if output.dim() == 2:
        hidden_size = output.shape[-1]
        output = output.reshape(module.B, -1, hidden_size)
    return output