from __future__ import print_function
import tensorflow as tf
import numpy as np

##### create dataset class  from_tensor_slices
with tf.Graph().as_default(),tf.Session() as sess:
	# instance a dataset,np.array() => tf.constant => tensorflow
	dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5]))
	# we can also use tf.data.TextLineDataset because this inherit tf.data.Dataset
	# dataset = tf.data.TextLineDataset.from_tensor_slices(np.array([1,2,3,4,5]))
	# return a Iterator over the element of this dataset 
	iterator = dataset.make_one_shot_iterator()
	element = iterator.get_next() # every element is a number
	for i in range(5):
		print(sess.run(element))  # 1,2,3,4,5


##### read data from file
"""
we have a file test.csv:
1,2,0
4,5,1
7,8,2
"""
with tf.Graph().as_default(),tf.Session() as sess:
	dataset = tf.data.TextLineDataset("test.csv")
	iterator = dataset.make_one_shot_iterator()
	element = iterator.get_next() # every element is a vector
	try:
		while True:
			print(sess.run(element))
	except tf.errors.OutOfRangeError:
		print("end!")

##### more complex dataset
"""
1,2,0
4,5,1
7,8,2
the last column is label we create => batch of feature,label
"""

with tf.Graph().as_default(),tf.Session() as sess:
	def to_tensor(line):
		parsed_line = tf.decode_csv(line,[[0.],[0.],[0]]) # => tensor
		#label = parsed_line[-1]
		label =  parsed_line[-1]
		del parsed_line[-1]
		features = parsed_line
		features_names = ['feature_1','feature_2']
		d = dict(zip(features_names,features)),label
		return d

	dataset = tf.data.TextLineDataset("test.csv").map(to_tensor).batch(2)
	iterator = dataset.make_one_shot_iterator()
 	batch_features,batch_labels = iterator.get_next()
	try:
		while True:
			batch_fea,batch_lab = sess.run([batch_features,batch_labels])			
			print(batch_fea,batch_lab)
	except tf.errors.OutOfRangeError:
		print("end!")
	

