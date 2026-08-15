import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# 1. Generate & filter image
im = np.zeros((256, 256))
im[64:-64, 64:-64] = 1
im = ndimage.gaussian_filter(ndimage.rotate(im, 15, mode="constant"), 8)

# 2. Extract Sobel features (Clean vs Noisy)
def get_sobel(image):
    return np.hypot(ndimage.sobel(image, 0), ndimage.sobel(image, 1))

plots = [
    ("Square", im),
    ("Sobel (X)", ndimage.sobel(im, 0)),
    ("Sobel Filter", get_sobel(im)),
    ("Sobel (Noisy)", get_sobel(im + 0.07 * np.random.random(im.shape))),
]

# 3. Plot
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, (title, img) in zip(axes, plots):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()