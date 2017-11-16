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
input_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,data_dim],name='Inputs') # Actual Size = 55000x784
gold_output_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,10],name='Gold') # Actual Size = 55000x10 (one-hot encoding)

rand_initializer = tf.random_uniform_initializer(-1, 1)

# Define variables for weights and bias of FIRST LAYER
initializer = tf.truncated_normal([data_dim, 256], stddev=0.1)
weights_hidden1 = tf.get_variable("Wh1", initializer=initializer)
bias_hidden1 = tf.get_variable("bh1", [256], initializer=rand_initializer)

# Define variables for weights and bias of SECOND LAYER
initializer = tf.truncated_normal([256, 256], stddev=0.1)
weights_hidden2 = tf.get_variable("Wh2", initializer=initializer)
bias_hidden2 = tf.get_variable("bh2", [256], initializer=rand_initializer)

# Define variables for weights and bias of LINEAR LAYER
initializer = tf.truncated_normal([256, classes], stddev=0.1)
weights_linear = tf.get_variable("Wl", initializer=initializer)
bias_linear = tf.get_variable("bl", [classes], initializer=rand_initializer)

# Define the Model's Structure:

hidden_layer1_output = tf.matmul(input_placeholder, weights_hidden1) + bias_hidden1
hidden_layer1_relu = tf.nn.relu(hidden_layer1_output)

hidden_layer2_output = tf.matmul(hidden_layer1_relu, weights_hidden2) + bias_hidden2
hidden_layer2_relu = tf.nn.relu(hidden_layer2_output)

linear_layer_output = tf.matmul(hidden_layer2_relu, weights_linear) + bias_linear

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

            # Predictions for both train and test sets, for error rate calculation
            the_dict = { input_placeholder: mnist.train.images , gold_output_placeholder: mnist.train.labels}
            train_errors.append(1 - (sess.run(prediction_accuracy, feed_dict=the_dict)))

            the_dict = { input_placeholder: mnist.test.images , gold_output_placeholder: mnist.test.labels}
            test_errors.append(1 - (sess.run(prediction_accuracy, feed_dict=the_dict)))
    else:
        saver.restore(sess, "./models/two_hidden_layer_model")

    # Calculation of final Error Rates
    the_dict = { input_placeholder: mnist.train.images , gold_output_placeholder: mnist.train.labels}
    print(1-sess.run(prediction_accuracy, feed_dict=the_dict))

    the_dict = { input_placeholder: mnist.test.images , gold_output_placeholder: mnist.test.labels}
    print(1-sess.run(prediction_accuracy, feed_dict=the_dict))

    y_hat = gold_test.eval(feed_dict=the_dict)
    y_pred = prediction_test.eval(feed_dict=the_dict)
    if TRAIN_MODEL:
        plt.plot(range(EPOCHS), train_errors)
        plt.plot(range(EPOCHS), test_errors)
        saver.save(sess, "./models/two_hidden_layer_model")

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_hat, y_pred)
np.set_printoptions(precision=2)
class_names = [0,1,2,3,4,5,6,7,8,9]
