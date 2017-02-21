import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import gc

model_path = 'cnn_model/muti_perception_model.ckpt'

train_path = "./train"
train_feature = []
train_label = []
filename = []
for folder in range(1,10):
	train_folder = os.path.join(train_path,str(folder))
	files= os.listdir(train_folder)
	for file in files: 
		picture = mpimg.imread(os.path.join(train_folder,file)) 
		feature = np.reshape(picture,[192,192,3])
		train_feature.append(feature)
		train_label.append(int(folder))
		filename.append(file)
train_label = np.asarray(train_label)
train_feature = np.asarray(train_feature)
filename = np.asarray(filename)
print(len(train_label)) # 900
print(len(train_feature)) # 900
print(len(filename)) # 900

index = [i for i in range(len(train_feature))]    
random.shuffle(index)   
train_feature = train_feature[index]  
train_label = train_label[index]
filename = filename[index]

epochs = 100
batch_size = 10
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

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(full_2, input_label))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#correct = tf.cast(tf.equal(tf.argmax(full_2,1),tf.argmax(input_label,1)),tf.float32)
correct = tf.cast(tf.equal(tf.argmax(full_2,1),input_label),tf.float32)
correct_rate = tf.reduce_mean(correct)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.initialize_all_variables())

batch_total = len(train_feature)//batch_size
for epoch in range(epochs):
	index = [i for i in range(len(train_feature))]    
	random.shuffle(index)   
	train_feature = train_feature[index]  
	train_label = train_label[index]
	filename = filename[index]
	total_cost = 0.0
	for batch in range(batch_total):
		batch_feature = np.asarray( train_feature[batch:batch+batch_size] )
		batch_label = np.asarray( train_label[batch:batch+batch_size] )
		_ , batch_cost,cor_rate = sess.run([optimizer,cost,correct_rate],feed_dict={input_feature:batch_feature, input_label:batch_label,keep_prob:0.6})
		total_cost += batch_cost
	avg_cost = total_cost/batch_total
	print("Epoch ",epoch," : Cost- ",avg_cost, "Correct- ",cor_rate)
print("Training Done!")

#test_rate = sess.run(correct_rate,feed_dict={input_feature:np.asarray( train_feature ), input_label:np.asarray( train_label ), keep_prob:1.0})
#print("Test correct rate : ", test_rate)

del train_feature
del train_label
del filename
gc.collect()

save_path = saver.save(sess, model_path, write_meta_graph=False)
print("Save path : ", save_path)

sess.close()

print("Done")