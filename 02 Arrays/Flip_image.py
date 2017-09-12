# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

image = mpimg.imread("MarshOrchid.jpg")
x = tf.Variable(image,name="x")

    
model = tf.global_variables_initializer()

with tf.Session() as sess:
   x = tf.image.flip_up_down(x)
   sess.run(model)
   res = sess.run(x)  
   # for i in range(2):


plt.imshow(res)
plt.show()