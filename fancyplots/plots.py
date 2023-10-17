import numpy as np
import matplotlib.pyplot as plt
from fancyplots.utils import complex_to_hsv


def complex_imshow(field, figsize: tuple[int,int] = (15,5), remove_ticks: bool = False, ref: tuple[int,int] = None):
    if ref is None:
        ref = np.argmax(np.abs(field))
        ref = np.unravel_index(ref, field.shape)

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    pl0 = axs[0].imshow(np.abs(field), cmap='gray')
    pl1 = axs[1].imshow(np.angle(field * np.exp(-1j * np.angle(field[ref[0], ref[1]]))), cmap='hsv')
    pl2 = axs[2].imshow(complex_to_hsv(field * np.exp(-1j * np.angle(field[ref[0], ref[1]])), rmin=0, rmax=np.max(np.abs(field))))
    pls = [pl0, pl1, pl2]

    _ = axs[0].set_title("Amplitude")
    _ = axs[1].set_title("Phase")
    _ = axs[2].set_title("Complex field")

    if remove_ticks:
        _ = [axs[i].set_xticks([]) for i in range(len(axs))]
        _ = [axs[i].set_yticks([]) for i in range(len(axs))]
    
    return (fig, axs, pls)