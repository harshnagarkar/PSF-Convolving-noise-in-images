import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('elepant.png')
img = img.convert('1', dither=Image.NONE)
img = np.float32(img)
print(img.dtype)
img += np.random.poisson(img,img.shape)
img = gaussian_filter(img,sigma=2)

plt.imshow(img)
plt.show()
