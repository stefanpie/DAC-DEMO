import os
from pathlib import Path
import random
from typing import Union, Optional

import numpy as np
from rich.pretty import pprint as pp

import torch
import torch.nn as nn

from siren_arch import SIREN_IMG_MODEL

import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent
HW_DIR = SCRIPT_DIR.parent/"hw"
TEST_DATA_DIR = HW_DIR/"test_data"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def serialize_numpy(array: np.ndarray, fp: Path, np_type=np.float32):
    casted_array: np.ndarray = array.astype(np_type)
    casted_array.tofile(fp)

def deserialize_numpy(fp: Path, np_type=np.float32):
    array: np.ndarray = np.fromfile(fp, dtype=np_type)
    return array


def gen_data_siren(test_data_dir: Path, model_fp: Path):
    net = SIREN_IMG_MODEL()
    net.load_state_dict(torch.load(model_fp))
    net.eval()

    input_layer_weight = net.layers[0].weight.detach().numpy()
    input_layer_bias = net.layers[0].bias.detach().numpy()
    input_layer_w0 = np.array([net.layers[0].activation.w0]).astype(np.float32)

    print(input_layer_weight)

    hidden_layers_weight_list = [
        layer.weight.detach().numpy() for layer in net.layers[1:]
    ]
    hidden_layers_bias_list = [layer.bias.detach().numpy() for layer in net.layers[1:]]
    hidden_layers_w0_list = [
        np.array(layer.activation.w0).astype(np.float32) for layer in net.layers[1:]
    ]

    hidden_layers_weight = np.array(hidden_layers_weight_list)
    hidden_layers_bias = np.array(hidden_layers_bias_list)
    hidden_layers_w0 = np.array(hidden_layers_w0_list)

    output_layer_weight = net.last_layer.weight.detach().numpy()
    output_layer_bias = net.last_layer.bias.detach().numpy()

    # x_in = torch.randn(1, 2)
    x_in = torch.tensor([[0.0, 0.0]])
    x_out = net(x_in)
    print(x_in)
    print(x_out)

    x_in_np = x_in.detach().numpy()
    x_out_np = x_out.detach().numpy()

    serialize_numpy(x_in_np, test_data_dir / "siren_x_in.bin")
    serialize_numpy(x_out_np, test_data_dir / "siren_x_out.bin")
    serialize_numpy(input_layer_weight, test_data_dir / "siren_input_layer_weight.bin")
    serialize_numpy(input_layer_bias, test_data_dir / "siren_input_layer_bias.bin")
    serialize_numpy(input_layer_w0, test_data_dir / "siren_input_layer_w0.bin")
    serialize_numpy(
        hidden_layers_weight, test_data_dir / "siren_hidden_layers_weight.bin"
    )
    serialize_numpy(hidden_layers_bias, test_data_dir / "siren_hidden_layers_bias.bin")
    serialize_numpy(hidden_layers_w0, test_data_dir / "siren_hidden_layers_w0.bin")
    serialize_numpy(
        output_layer_weight, test_data_dir / "siren_output_layer_weight.bin"
    )
    serialize_numpy(output_layer_bias, test_data_dir / "siren_output_layer_bias.bin")


if __name__ == "__main__":
    TEST_DATA_DIR.mkdir(exist_ok=True)
    gen_data_siren(TEST_DATA_DIR, SCRIPT_DIR/"demo_models/demo_image_00.pt")
