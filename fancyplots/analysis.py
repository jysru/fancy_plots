import matplotlib.pyplot as plt
import numpy as np


def plot_features(layers_dict, layer_idx: int = -1, img_idx: int = -1, channel_idx: int = -1):
    plt.imshow(layers_dict[layer_idx]['data'][img_idx, channel_idx, ...])
    plt.title(f"Layer {layer_idx}, {layers_dict[layer_idx]['name']}")
    plt.colorbar()

def plot_layer_channels(layers_dict, layer_idx: int = -1, img_idx: int = -1, transpose: bool = False, cmap = 'viridis'):
    chans = layers_dict[layer_idx]['data'].shape[1]
    img_shape = layers_dict[layer_idx]['data'].shape[-2:]
    img_type = layers_dict[layer_idx]['data'].dtype
    divs = squarest_divisors(chans)

    stitched_image = np.zeros(shape=(*divs, *img_shape), dtype=img_type)
    for chan in range(chans):
        idxs = np.unravel_index(chan, shape=divs)
        stitched_image[idxs[0], idxs[1], ...] = layers_dict[layer_idx]['data'][img_idx, chan, ...]

    stitched_image = np.moveaxis(stitched_image, 2, 1)
    stitched_image = np.reshape(stitched_image, (divs[0] * img_shape[0], divs[1] * img_shape[1]))
    plt.imshow(np.transpose(stitched_image) if transpose else stitched_image, cmap=cmap)
    plt.title(f"Layer {layer_idx}, {layers_dict[layer_idx]['name']}, {(chans, *img_shape)}")
    plt.colorbar()

def squarest_divisors(n):
    divs = np.array(list(get_divisors(n)))
    best_div = np.max(divs[divs <= np.sqrt(n)])
    other_div = n // best_div
    return best_div, other_div

def get_divisors(n) -> list[int]:
    for i in range(1, int(n / 2) + 1):
        if n % i == 0:
            yield i
    yield n
