from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

# Set this flag to True to retrain model, otherwise set to False to load from file
################################
#TRAIN_MODEL = True
TRAIN_MODEL = False
################################

# import dataset with one-hot encoding
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Define data dimension size, and number of classes to be predicted
data_dim = 784
classes = 10

# Placeholder for the input samples
input_placeholder = tf.placeholder(dtype=tf.float32,shape=[None, data_dim],name='Inputs') # Actual Size = 55000x784
gold_output_placeholder = tf.placeholder(dtype=tf.float32,shape=[None, 10],name='Gold') # Actual Size = 55000x10 (one-hot encoding)

#Reshape input as 28x28 images
input_placeholder_reshaped = tf.reshape(input_placeholder, [-1, 28, 28, 1])

rand_initializer = tf.random_uniform_initializer(-1, 1)

# Define First Convolutional Layer
initializer = tf.truncated_normal([3, 3, 1, 16], stddev=0.1)
weights_conv1 = tf.get_variable("Wc1", initializer=initializer)
bias_conv1 = tf.get_variable("bc1", [16], initializer=rand_initializer)
conv1 = tf.nn.conv2d(input_placeholder_reshaped, weights_conv1, strides=[1,1,1,1],padding='SAME')
conv1_relu = tf.nn.relu(conv1 + bias_conv1)
pool1 = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

# Define Second Convolutional Layer
initializer = tf.truncated_normal([3, 3, 16, 16], stddev=0.1)
weights_conv2 = tf.get_variable("Wc2", initializer=initializer)
bias_conv2 = tf.get_variable("bc2", [16], initializer=rand_initializer)
conv2 = tf.nn.conv2d(pool1, weights_conv2, strides=[1, 1, 1, 1],padding='SAME')
conv2_relu = tf.nn.relu(conv2 + bias_conv2)
pool2 = tf.nn.max_pool(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
pool2_flattened = tf.reshape(pool2, [-1, 7*7*16])

# Define Hidden Layer
initializer = tf.truncated_normal([7*7*16, 256], stddev=0.1)
weights_hidden = tf.get_variable("Wh2", initializer=initializer)
bias_hidden = tf.get_variable("bh2", [256], initializer=rand_initializer)
hidden_layer_output = tf.matmul(pool2_flattened, weights_hidden) + bias_hidden
hidden_layer_relu = tf.nn.relu(hidden_layer_output)

# Define Linear Layer
initializer = tf.truncated_normal([256, classes], stddev=0.1)
weights_linear = tf.get_variable("Wl", initializer=initializer)
bias_linear = tf.get_variable("bl", [classes], initializer=rand_initializer)
linear_layer_output = tf.matmul(hidden_layer_relu, weights_linear) + bias_linear

# Define the Model's Structure:

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(linear_layer_output, gold_output_placeholder))

optimization_operation = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

softmaxed_test = tf.nn.softmax(linear_layer_output)
prediction_test = tf.argmax(softmaxed_test,1)
gold_test = tf.argmax(gold_output_placeholder,1)

correct_predictions = tf.equal(gold_test, prediction_test)
prediction_accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))

EPOCHS = 20
train_errors = []
test_errors = []

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if TRAIN_MODEL:
        for epoch in range(EPOCHS):
            for step in range(1000):
                inputs_batch , gold_batch = mnist.train.next_batch(55)
                the_dict = { input_placeholder: inputs_batch , gold_output_placeholder: gold_batch}
                sess.run([optimization_operation, loss], the_dict)

            # Calculation of train set error rate is handled by using batches in order to combat
            # dimensionality explosion caused by convolutional layers
            train_errors.append(0)
            for step in range(1000):
                inputs_batch , gold_batch = mnist.train.next_batch(55)
                the_dict = { input_placeholder: inputs_batch , gold_output_placeholder: gold_batch}
                train_errors[epoch] += (1 - (sess.run(prediction_accuracy, feed_dict=the_dict)))
            train_errors[epoch] = train_errors[epoch] / 1000

            # Calculation of test set error done directly
            the_dict = { input_placeholder: mnist.test.images , gold_output_placeholder: mnist.test.labels}
            test_errors.append(1 - (sess.run(prediction_accuracy, feed_dict=the_dict)))
            print("Epoch %d done" % (epoch))
    else:
        saver.restore(sess, "./models/conv_model")

    # Calculation of train set error rate is handled by using batches in order to combat
    # dimensionality explosion caused by convolutional layers
    train_err = 0
    for step in range(1000):
                inputs_batch , gold_batch = mnist.train.next_batch(55)
                the_dict = { input_placeholder: inputs_batch , gold_output_placeholder: gold_batch}
                train_err += (1 - (sess.run(prediction_accuracy, feed_dict=the_dict)))
    train_err = train_err / 1000
    print(train_err)

    # Calculation of test set error done directly
    the_dict = { input_placeholder: mnist.test.images , gold_output_placeholder: mnist.test.labels}
    print(1-sess.run(prediction_accuracy, feed_dict=the_dict))

    y_hat = gold_test.eval(feed_dict=the_dict)
    y_pred = prediction_test.eval(feed_dict=the_dict)
    if TRAIN_MODEL:
        plt.plot(range(EPOCHS), train_errors)
        plt.plot(range(EPOCHS), test_errors)
        saver.save(sess, "./models/conv_model")

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_hat, y_pred)
np.set_printoptions(precision=2)
class_names = [0,1,2,3,4,5,6,7,8,9]
