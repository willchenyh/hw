


import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import get_mnist

sess = tf.InteractiveSession()
mnist = get_mnist.read_data_sets("MNIST-data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

# parameters
lr = 0.001
batch_size = 100
train_error_list = []
test_error_list = []

# from here continued

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# First Convolutional Layer in 5x5x1 out 32

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# reshape input from 764 to 28x28

x_image = tf.reshape(x, [-1,28,28,1])

#rectified linear output & max pooling

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2nd layer 5x5x32 out 64

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#rectified linear output & max pooling

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# full connected layer

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# flatten last convolutional output,  pass it to full connected Layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#add dropout after full connected layer

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

total_batch = int(mnist.train.num_examples/batch_size)
print (total_batch)
counter = 0
for epoch in range(1):
  for i in range(100):
    counter += 1
    batch = mnist.train.next_batch(batch_size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    train_error = 1 - train_accuracy
    print("step %d, training error %g"%(counter, train_error))
    test_accuracy = accuracy.eval(feed_dict={
	x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    test_error = 1 - test_accuracy
    print("test error %g"%(test_error))
    train_error_list.append(train_error)
    test_error_list.append(test_error)

# plot error

xcoords = range(len(train_error_list))
plt.plot(xcoords, train_error_list)
plt.plot(xcoords, test_error_list)
axes = plt.gca()
plt.grid()
plt.title('Training and testing errors')
plt.xlabel('batches')
plt.ylabel('error')
plt.savefig('plot_conv_all.png')




