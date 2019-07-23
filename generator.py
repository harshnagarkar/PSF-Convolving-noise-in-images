import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
import psf 
from scipy.signal import fftconvolve as conv2


def generate_psf(x, y, cmap='hot', savebin=False, savetif=True, savevol=False,
                plot=True, **kwargs):
    """Calculate, save, and plot various point spread functions."""

    args = {
        'shape': (x, y),  # number of samples in z and r direction
        'dims': (5.0, 5.0),   # size in z and r direction in micrometers
        'ex_wavelen': 488.0,  # excitation wavelength in nanometers
        'em_wavelen': 520.0,  # emission wavelength in nanometers
        'num_aperture': 1.2,
        'refr_index': 1.333,
        'magnification': 1.0,
        'pinhole_radius': 0.05,  # in micrometers
        'pinhole_shape': 'square',
    }
    args.update(kwargs)

    obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
    expsf = obsvol.expsf
    empsf = obsvol.empsf
    gauss = gauss2 = psf.PSF(psf.GAUSSIAN | psf.EXCITATION, **args)

    # print(expsf)
    # print(empsf)
    # print(obsvol)
    print(gauss)
    print(gauss2)
    return (obsvol.data)


img = Image.open('elepant.png')
img = img.convert('1', dither=Image.NONE)
img = np.float32(img)
img = conv2(img,generate_psf(img.shape[0],img.shape[1]),mode='same')

# print(img.dtype)
# img += np.random.poisson(img,img.shape)
# img = gaussian_filter(img,sigma=2)


plt.imshow(img)
plt.show()
