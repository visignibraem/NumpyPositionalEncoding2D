import math
import numpy as np


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = np.zeros((length, d_model))
    position = np.expand_dims(np.arange(0, length), axis=1)
    div_term = np.exp((np.arange(0, d_model, 2, dtype=np.float) *
                       -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = np.zeros((d_model, height, width))
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = np.exp(np.arange(0., d_model, 2) *
                      -(math.log(10000.0) / d_model))
    pos_w = np.expand_dims(np.arange(0., width), axis=1)
    pos_h = np.expand_dims(np.arange(0., height), axis=1)
    pe[0:d_model:2, :, :] = np.tile(np.expand_dims(np.sin(pos_w * div_term).transpose(1, 0), 1), (1, height, 1))
    pe[1:d_model:2, :, :] = np.tile(np.expand_dims(np.cos(pos_w * div_term).transpose(1, 0), 1), (1, height, 1))
    pe[d_model::2, :, :] = np.tile(np.expand_dims(np.sin(pos_h * div_term).transpose(1, 0), 2), (1, 1, width))
    pe[d_model + 1::2, :, :] = np.tile(np.expand_dims(np.cos(pos_h * div_term).transpose(1, 0), 2), (1, 1, width))

    return pe
