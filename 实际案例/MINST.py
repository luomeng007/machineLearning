# -*- coding:utf-8 -*-
"""
author: 15025
age: 26
e-mail: 1502506285@qq.com
time: 2020/12/12 20:00
software: PyCharm

Description:
    MNIST 手写字数据集
    SVC 分类器
English:
    估计器（estimator）

"""
from sklearn import svm
from sklearn import datasets
import matplotlib.pyplot as plt

# initialize the estimator
svc = svm.SVC(gamma=0.001, C=100)
# from datasets import Digits image
digits = datasets.load_digits()
# print relative information about datasets
# DESCR for describe
# print(digits.DESCR)

# 0: black, 15:white
# this is a number 0
# print(digits.images[0])

# show image, get grey image
# plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()

# the data of image is saved in target
# print(digits.target)
# print(digits.target.size)   # number of images: 1797

# we use 1791 to train machine, and use rest 6 to verify
#  first, we could have a look on rest 6 image
plt.subplot(321)
plt.imshow(digits.images[1791], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(322)
plt.imshow(digits.images[1792], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(323)
plt.imshow(digits.images[1793], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(324)
plt.imshow(digits.images[1794], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(325)
plt.imshow(digits.images[1795], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(326)
plt.imshow(digits.images[1796], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# then, we train estimator
# we have 1797 images, so the subscribe ends with 1796
svc.fit(digits.data[1:1790], digits.target[1:1790])
result = svc.predict(digits.data[1791:1797])

print(result)

print(digits.target[1791:1797])

# perfect predict the number in image
# [4 9 0 8 9 8]
# [4 9 0 8 9 8]
