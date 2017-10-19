import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog

cars = glob.glob('./vehicles/*/image0000.png')
file = cars[0]
image = mpimg.imread(file)
feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        


pix_per_cell = 8
cell_per_block = 2
orient = 9

features, hog_image = hog(feature_image[:,:,0], orientations=orient,
                          pixels_per_cell=(pix_per_cell, pix_per_cell), 
                          cells_per_block=(cell_per_block, cell_per_block), 
                          visualise=True, feature_vector=False,
                          block_norm="L2-Hys")

plt.imshow(hog_image)

mpimg.imsave('examples/hog_example.png', hog_image)