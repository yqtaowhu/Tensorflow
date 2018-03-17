from __future__ import print_function
import tensorflow as tf
import numpy as np

x1 = tf.placeholder(tf.float32,shape=(3,2))
y1 = tf.placeholder(tf.float32,shape=(2,3))
z1 = tf.matmul(x1,y1)

x2 = tf.placeholder(tf.float32,shape=None)
y2 = tf.placeholder(tf.float32,shape=None)
z2 = x2 + y2

# using feed_dict when placehoder
with tf.Session() as sess:
	z2_value = sess.run(z2,feed_dict={x2:1,y2:2}) 
	print(z2_value)
	rand_x = np.random.rand(3,2)
	rand_y = np.random.rand(2,3)
	z1_value,z2_value = sess.run(
		[z1,z2],                   # run together
		feed_dict={
			x1:rand_x,y1:rand_y,
			x2:1,y2:2
		}
	)
	print(z1_value,z2_value)
