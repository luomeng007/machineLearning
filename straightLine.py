# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:35:44 2020

@author: 15025
"""

import numpy as np
import tensorflow as tf


model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

model.fit(xs, ys, epochs=500)

# debug
print(model.predict([10.0]))    # [[18.999876]]