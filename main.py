## todo :
# SGD visualization
# weights visualization

import matplotlib.image as mpimg
import numpy as np
img = mpimg.imread("mnist_png/training/0/1.png")

print(img[1][0])

for i in range(28):
    for j in range(28):
        if(img[i][j]>0):
            print('@', end='')
        else:
            print('-', end='')
    print('')