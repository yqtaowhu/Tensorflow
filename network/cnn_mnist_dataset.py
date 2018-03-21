from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets import mnist

# path is ~/.keras/datasets/mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data(path='./mnist')

print(
	'\nx_train shape',x_train.shape,  #(60000,28,28)
	'\ny_train shape',y_train.shape,  #(60000,)
	'\nx_test shape',x_test.shape,
	'\ny_test shape',y_test.shape,
)
"""
# show the first digit
plt.imshow(x_train[0],cmap='gray')
plt.title(y_train[0])  # int number
plt.show()
"""
x_train = x_train.reshape(60000,28,28,1)
y_train = y_train.reshape(60000,1)
x_test = x_test[:1000].reshape(1000,28,28,1)
y_test = y_test[:1000].reshape(1000,1)

tfx = tf.placeholder(tf.float32,shape=x_train.shape)/255 #(batch,height,width,channel)
tfy = tf.placeholder(tf.int32,shape=y_train.shape)

dataset = tf.data.Dataset.from_tensor_slices((tfx,tfy))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.repeat(5)
iterator = dataset.make_initializable_iterator()

batch_x,batch_y = iterator.get_next()

# cnn
conv1 = tf.layers.conv2d(
	inputs = batch_x,
	filters = 16,
	kernel_size = 5,
	strides = 1,
	padding = 'same',
	activation = tf.nn.relu
)        # image -> (28,28,16)
pool1 = tf.layers.max_pooling2d(
	inputs = conv1,
	pool_size =2,
	strides = 2
)        # shape -> (14,14,16)
conv2 = tf.layers.conv2d(pool1,32,5,1,'same',activation=tf.nn.relu)#(14,14,32)
pool2 = tf.layers.max_pooling2d(conv2,2,2) # (7,7,32)
flat = tf.reshape(pool2,[-1,7*7*32])
output = tf.layers.dense(flat,10)

loss = tf.losses.sparse_softmax_cross_entropy(
	labels = batch_y,
	logits = output,
)
train = tf.train.AdamOptimizer(0.01).minimize(loss)



with tf.Session() as sess:
	# initiliazble
	sess.run([iterator.initializer,tf.global_variables_initializer()],
		feed_dict={tfx:x_train,tfy:y_train}
	)
	# a batch size (32,28,28,1) ,(32,1)
	bat_x,bat_y = sess.run([batch_x,batch_y])
	print(
		'\nbatch_x shape:',bat_x.shape,
		'\nbatch_y shape:',bat_y.shape
	)
	for step in range(2001):
		try:
			_,train_loss = sess.run([train,loss])
			if step % 100 == 0:
				predict = sess.run(output,{batch_x:x_test,batch_y:y_test}) #(1000,10)
				pre = np.argmax(predict,axis=1)
				#print(pre[:10]),print(y_test[:10])
				print("accuracy:",sum(pre == np.array(y_test).reshape(1000,))/1000.0)
				test_loss = sess.run(loss,{batch_x:x_test,batch_y:y_test})
				print('\nstep:',step,
					'\ntrain loss:',train_loss,
					'\ntest loss:',test_loss,
				)

		except tf.errors.OutOfRangeError:
			print("finish!")
			break
