from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(-1,1,(1000,1)) 
y = np.power(x,2) + np.random.normal(0,0.1,size=x.shape)
x_train,x_test = np.split(x,[800])
y_train,y_test = np.split(y,[800])
print(
	'\nx_train shape',x_train.shape,
	'\ny_train shape',y_train.shape,
)
"""
plt.scatter(x_train,y_train)
plt.show()
"""

tfx = tf.placeholder(x_train.dtype,x_train.shape)
tfy = tf.placeholder(y_train.dtype,y_train.shape)

# create dataloader
dataset = tf.data.Dataset.from_tensor_slices((tfx,tfy)) #reference tf_dataset_basic.py
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.repeat(5)
iterator = dataset.make_initializable_iterator()

# built network
batch_x,batch_y = iterator.get_next()  # batch_x:(32,1)
h1 = tf.layers.dense(batch_x,10,tf.nn.relu) # batch_x:(32,10)
out = tf.layers.dense(h1,1) # 32*1
loss = tf.losses.mean_squared_error(batch_y,out)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
	#initializable
	sess.run([iterator.initializer,tf.global_variables_initializer()],
			feed_dict={tfx:x_train,tfy:y_train})
	for step in range(301):
		try:
			_,train_loss = sess.run([train,loss])
			if step % 10 == 0:
				test_loss = sess.run(loss,{batch_x:x_test,batch_y:y_test})
				print('\nsetp:',step,
					'\ntrain loss:',train_loss,
					'\ntest loss:',test_loss,
				)
		except tf.errors.OutOfRangeError:
			print("finish!")
			break
