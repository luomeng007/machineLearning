# -*- coding:utf-8 -*-
"""
author: 15025
age: 26
e-mail: 1502506285@qq.com
time: 2020/11/30 16:05
software: PyCharm

Description:
"""

import numpy as np
from sklearn import datasets
import matplotlib.pylab as plt

# read data from imported datesets
digits = datasets.load_digits()
# digits has two, one is actual image data and the other is image labels
# print(digits.data.shape)
# print(digits.images.shape)
"""
(1797, 64)
(1797, 8, 8)
1797: the number of images in the dataset
64 means all data arranged in big line vector, this data could not be used to draw figure
(8, 8) means 2D spatial arrangement, we use this data to draw figure
"""
# if we want to draw single image, first grab image from dateset using Numpy's array
# grab first image in digits
# img = digits.images[0, :, :]
# plot image, default colormap is jet
# plt.imshow(img, cmap="gray")
# plt.show()

# we use a iteration to draw first 10 figure in subplots
for image_index in range(10):
    subplot_index = image_index + 1
    plt.subplot(2, 5, subplot_index)
    # plt.imshow(digits.images[image_index, :, :], cmap="gray")

plt.show()
