import torch


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts
