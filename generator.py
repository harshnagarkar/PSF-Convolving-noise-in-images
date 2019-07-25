from scipy import fftpack
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
import psf 
from scipy.signal import convolve as conv2
# from scipy.signal import deconvolve

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

    print(obsvol)
    return (obsvol.data)


# custom convolve and deconvolve function
# def convolve(star, psf):
#     star_fft = fftpack.fftshift(fftpack.fftn(star))
#     psf_fft = fftpack.fftshift(fftpack.fftn(psf))
#     return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft*psf_fft)))


# def deconvolve(star, psf):
#     star_fft = fftpack.fftshift(fftpack.fftn(star))
#     psf_fft = fftpack.fftshift(fftpack.fftn(psf))
#     return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))


img = Image.open('elepant.png')
img = img.convert('L', dither=Image.NONE)
img = np.float32(img)
psf = generate_psf(img.shape[0],img.shape[1])
convimg = conv2(img,psf)
img = convimg
# recover,reminder = deconvolve(convimg,psf)
# if recover == img:
#     print("hell yeah")
img += np.random.poisson(img,img.shape)
img = gaussian_filter(img,sigma=2)


plt.imshow(img)
plt.show()
