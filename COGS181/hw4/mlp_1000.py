'''
Modified from the script created by Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

Author: Yuhan Chen

Number of hidden layer: 1
Number of units in hidden layer: 300
Optimizer: stochastic gradient descent
Training data: first 1000 from each class. I use the first 10,000 assuming the data is perfectly shuffled.
'''

from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import get_mnist

# Import MNIST data
mnist = get_mnist.read_data_sets('MNIST-data/', one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.1
training_epochs = 10000
batch_size = 100
max_batch = 100
display_step = 1
min_diff = 0.00001

# Network Parameters
n_hidden_1 = 300 # 1st layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    prev_error = 1
    train_error_list = []
    test_error_list = []
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = max_batch
	print (total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
	# Compute training error
	if epoch > 0:
	    prev_error = train_error
        # Test model
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	train_error = 1 - accuracy.eval({x: mnist.train.images, y: mnist.train.labels})
	print("Training Error:", train_error)
	# Compute testing error
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	error = 1 - accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
        print("Testing Error:", error)

	# Save errors in lists
	train_error_list.append(train_error)
	test_error_list.append(error)
	
	# Check error difference
	error_diff = abs(prev_error - train_error)
	if error_diff < min_diff:
	    break

    # plot error
    xcoords = range(len(train_error_list))
    plt.plot(xcoords, train_error_list)
    plt.plot(xcoords, test_error_list)
    axes = plt.gca()
    plt.grid()
    plt.title('Training and testing errors')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.savefig('plot1000.png')

    print("Optimization Finished!")




