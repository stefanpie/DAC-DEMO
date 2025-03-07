import os
from pathlib import Path
from pprint import pp
import random
from typing import Union, Optional

import numpy as np
from rich.pretty import pprint as pp

import torch
import torch.nn as nn

from siren_pytorch import SirenNet


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def serialize_numpy(array: np.ndarray, fp: Path, np_type=np.float32):
    casted_array: np.ndarray = array.astype(np_type)
    casted_array.tofile(fp)

def gen_data_linear_layer(test_data_dir: Path):
    linear_layer = nn.Linear(32, 16)
    
    linear_in = torch.randn(1, 32)
    linear_out = linear_layer(linear_in)

    linear_layer_in_np = linear_in.detach().numpy()
    linear_layer_out_np = linear_out.detach().numpy()

    linear_layer_weight = linear_layer.weight.detach().numpy()
    linear_layer_bias = linear_layer.bias.detach().numpy()

    serialize_numpy(linear_layer_in_np, test_data_dir / "linear_layer_in.bin")
    serialize_numpy(linear_layer_out_np, test_data_dir / "linear_layer_out.bin")
    serialize_numpy(linear_layer_weight, test_data_dir / "linear_layer_weight.bin")
    serialize_numpy(linear_layer_bias, test_data_dir / "linear_layer_bias.bin")

def gen_data_activation(test_data_dir: Path):

    activation_elu = nn.ELU(alpha=1.0)
    activation_hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)
    activation_leakyrelu = nn.LeakyReLU(negative_slope=0.1)
    activation_relu = nn.ReLU()
    activation_gelu = nn.GELU()
    activation_sigmoid = nn.Sigmoid()
    activation_silu = nn.SiLU()
    activation_tanh = nn.Tanh()
    activation_softsign = nn.Softsign()
    activation_sin = torch.sin
    activation_cos = torch.cos

    activation_in = torch.randn(1, 32)
    activation_out_elu = activation_elu(activation_in)
    activation_out_hardtanh = activation_hardtanh(activation_in)
    activation_out_leakyrelu = activation_leakyrelu(activation_in)
    activation_out_relu = activation_relu(activation_in)
    activation_out_gelu = activation_gelu(activation_in)
    activation_out_sigmoid = activation_sigmoid(activation_in)
    activation_out_silu = activation_silu(activation_in)
    activation_out_tanh = activation_tanh(activation_in)
    activation_out_softsign = activation_softsign(activation_in)
    activation_out_sin = activation_sin(activation_in)
    activation_out_cos = activation_cos(activation_in)

    activation_in_np = activation_in.detach().numpy()
    activation_out_elu_np = activation_out_elu.detach().numpy()
    activation_out_hardtanh_np = activation_out_hardtanh.detach().numpy()
    activation_out_leakyrelu_np = activation_out_leakyrelu.detach().numpy()
    activation_out_relu_np = activation_out_relu.detach().numpy()
    activation_out_gelu_np = activation_out_gelu.detach().numpy()
    activation_out_sigmoid_np = activation_out_sigmoid.detach().numpy()
    activation_out_silu_np = activation_out_silu.detach().numpy()
    activation_out_tanh_np = activation_out_tanh.detach().numpy()
    activation_out_softsign_np = activation_out_softsign.detach().numpy()
    activation_out_sin_np = activation_out_sin.detach().numpy()
    activation_out_cos_np = activation_out_cos.detach().numpy()

    serialize_numpy(activation_in_np, test_data_dir / "activation_in.bin")
    serialize_numpy(activation_out_elu_np, test_data_dir / "activation_out_elu.bin")
    serialize_numpy(activation_out_hardtanh_np, test_data_dir / "activation_out_hardtanh.bin")
    serialize_numpy(activation_out_leakyrelu_np, test_data_dir / "activation_out_leakyrelu.bin")
    serialize_numpy(activation_out_relu_np, test_data_dir / "activation_out_relu.bin")
    serialize_numpy(activation_out_gelu_np, test_data_dir / "activation_out_gelu.bin")
    serialize_numpy(activation_out_sigmoid_np, test_data_dir / "activation_out_sigmoid.bin")
    serialize_numpy(activation_out_silu_np, test_data_dir / "activation_out_silu.bin")
    serialize_numpy(activation_out_tanh_np, test_data_dir / "activation_out_tanh.bin")
    serialize_numpy(activation_out_softsign_np, test_data_dir / "activation_out_softsign.bin")
    serialize_numpy(activation_out_sin_np, test_data_dir / "activation_out_sin.bin")
    serialize_numpy(activation_out_cos_np, test_data_dir / "activation_out_cos.bin")

def gen_data_siren(test_data_dir: Path):
    net = SirenNet(
        dim_in = 2,
        dim_hidden = 256,
        dim_out = 3,
        num_layers = 5,
        final_activation = nn.Sigmoid(),
        w0_initial = 30
    )

    input_layer_weight = net.layers[0].weight.detach().numpy()
    input_layer_bias = net.layers[0].bias.detach().numpy()
    input_layer_w0 = np.array([net.layers[0].activation.w0]).astype(np.float32)

    hidden_layers_weight_list = [layer.weight.detach().numpy() for layer in net.layers[1:]]
    hidden_layers_bias_list = [layer.bias.detach().numpy() for layer in net.layers[1:]]
    hidden_layers_w0_list = [np.array(layer.activation.w0).astype(np.float32) for layer in net.layers[1:]]

    hidden_layers_weight = np.array(hidden_layers_weight_list)
    hidden_layers_bias = np.array(hidden_layers_bias_list)
    hidden_layers_w0 = np.array(hidden_layers_w0_list)

    output_layer_weight = net.last_layer.weight.detach().numpy()
    output_layer_bias = net.last_layer.bias.detach().numpy()

    x_in = torch.randn(1, 2)
    x_out = net(x_in)

    x_in_np = x_in.detach().numpy()
    x_out_np = x_out.detach().numpy()

    serialize_numpy(x_in_np, test_data_dir / "siren_x_in.bin")
    serialize_numpy(x_out_np, test_data_dir / "siren_x_out.bin")
    serialize_numpy(input_layer_weight, test_data_dir / "siren_input_layer_weight.bin")
    serialize_numpy(input_layer_bias, test_data_dir / "siren_input_layer_bias.bin")
    serialize_numpy(input_layer_w0, test_data_dir / "siren_input_layer_w0.bin")
    serialize_numpy(hidden_layers_weight, test_data_dir / "siren_hidden_layers_weight.bin")
    serialize_numpy(hidden_layers_bias, test_data_dir / "siren_hidden_layers_bias.bin")
    serialize_numpy(hidden_layers_w0, test_data_dir / "siren_hidden_layers_w0.bin")
    serialize_numpy(output_layer_weight, test_data_dir / "siren_output_layer_weight.bin")
    serialize_numpy(output_layer_bias, test_data_dir / "siren_output_layer_bias.bin")


if __name__ == "__main__":
    TEST_DATA_DIR = Path(os.path.dirname(__file__)) / "test_data"
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    gen_data_linear_layer(TEST_DATA_DIR)
    gen_data_activation(TEST_DATA_DIR)
    gen_data_siren(TEST_DATA_DIR)
