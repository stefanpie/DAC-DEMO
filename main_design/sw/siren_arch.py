from siren_pytorch import SirenNet
from torch import nn

from functools import partial

SIREN_IMG_MODEL = partial(
    SirenNet,
    dim_in = 2,
    dim_hidden = 256,
    dim_out = 3,
    num_layers = 5,
    final_activation = None,
    w0_initial = 100
)
