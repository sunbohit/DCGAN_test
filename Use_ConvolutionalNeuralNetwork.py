import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pickle
import gc

model_path = 'cnn_model/muti_perception_model.ckpt'

#test_path = "./data/lovelive_192_192"
test_path = "./data/lovelive_192_192_2"
output_path = "./lovelive_cnn_result"

files= os.listdir(test_path)
count = 0
for file in files:
	count += 1 

print(count)

kernel_size = 5
stride_size = 1
pool_size = 2

input_feature = tf.placeholder(tf.float32,[None,192,192,3])
input_label = tf.placeholder(tf.int64, [None])
keep_prob = tf.placeholder(tf.float32)

def conv_layer(input, weights, biases):
	conv_result = tf.nn.bias_add(tf.nn.conv2d(input = input, filter = weights, strides=[1,stride_size,stride_size,1], padding='SAME'),biases)
	relu_result = tf.nn.relu(conv_result)
	output = tf.nn.max_pool(value=relu_result, ksize=[1,pool_size,pool_size,1], strides=[1,pool_size,pool_size,1], padding='SAME')
	return output

def full_layer(input, weights, biases):
	linear_result = tf.add(tf.matmul(input, weights),biases)
	output = tf.nn.relu(linear_result)
	output_d = tf.nn.dropout(output,keep_prob=keep_prob)
	return output_d
def out_layer(input, weights, biases):
	linear_result = tf.add(tf.matmul(input, weights),biases)
	return linear_result

weights = {
	'conv_1' : tf.Variable(tf.random_normal([kernel_size,kernel_size,3,16])),
	'conv_2' : tf.Variable(tf.random_normal([kernel_size,kernel_size,16,64])),
	'full_1' : tf.Variable(tf.random_normal([48*48*64,1000])),
	'out' : tf.Variable(tf.random_normal([1000,10])),
}
biases = {
	'conv_1' : tf.Variable(tf.random_normal([16])),
	'conv_2' : tf.Variable(tf.random_normal([64])),
	'full_1' : tf.Variable(tf.random_normal([1000])),
	'out' : tf.Variable(tf.random_normal([10]))
}

reshape_feature = tf.reshape(input_feature,[-1,192,192,3]) 
conv_1 = conv_layer(reshape_feature, weights['conv_1'], biases['conv_1'])
conv_2 = conv_layer(conv_1, weights['conv_2'], biases['conv_2'])
reshape_conv = tf.reshape(conv_2, [-1,48*48*64])
full_1 = full_layer(reshape_conv, weights['full_1'], biases['full_1'])
full_2 = out_layer(full_1,weights['out'],biases['out'])

predict_label = tf.argmax(full_2,1)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(full_2, input_label))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#correct = tf.cast(tf.equal(tf.argmax(full_2,1),tf.argmax(input_label,1)),tf.float32)
correct = tf.cast(tf.equal(tf.argmax(full_2,1),input_label),tf.float32)
correct_rate = tf.reduce_mean(correct)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print("Model Restore ...")
saver.restore(sess, model_path)
print('Model restore from %s', model_path)

files= os.listdir(test_path)
for file in files:
	picture = mpimg.imread(os.path.join(test_path,file))
	input_test_feature = np.reshape(picture,[-1,192,192,3])
	pred_label = sess.run(predict_label,feed_dict={input_feature:np.asarray( input_test_feature ), keep_prob:1.0})
	print(file," : ", pred_label[0])
	#outputfile = os.path.join(output_path,str(pred_label[0]),file)
	outputfile = os.path.join(output_path,str(pred_label[0]),"S2_"+file)
	mpimg.imsave(outputfile, picture)

sess.close()

print("Done")