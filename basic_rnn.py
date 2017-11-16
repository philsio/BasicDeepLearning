from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

TRAIN_MODEL = False

# import dataset with one-hot class encoding
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def binarize(images, threshold=0.1):
    return (threshold < images).astype('float32')

# Organise the data into arrays:
train_images = binarize(mnist.train.images)
train_labels = mnist.train.labels
test_images = binarize(mnist.test.images)
test_labels = mnist.test.labels

# Define data dimension size, and number of classes to be predicted
data_dim = 784
classes = 10

# Placeholder for the input samples
input_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,data_dim],name='Inputs') # Actual Size = 55000x784
gold_output_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,classes],name='Gold') # Actual Size = 55000x10 (one-hot encoding)


# Define the Stacked - GRU Cell

cell_size = 32
num_layers = 3
cell = tf.nn.rnn_cell.GRUCell(cell_size)
cell = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers)
    # ...and transform the input to match
#input_transformed = tf.reshape(input_placeholder, [-1, 1, data_dim])
input_transformed = tf.reshape(input_placeholder, [-1, data_dim, 1])

# Define the RNN

outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=input_transformed, dtype=tf.float32)
    #... and extract the last output of each sample to feed to the next layer
last_rnn_output = outputs[:, -1, :]

# Define the First Linear Layer
initializer = tf.truncated_normal([cell_size, 100], stddev=0.1)
weights_linear_1 = tf.get_variable("Wl1", initializer=initializer)
rand_initializer = tf.random_uniform_initializer(-1, 1)
bias_linear_1 = tf.get_variable("bl1", [100], initializer=rand_initializer)

linear_layer_1_output = tf.matmul(last_rnn_output,weights_linear_1) + bias_linear_1
# Batch Normalisation for First Linear Layer
epsilon = 1e-3
gamma = tf.Variable(tf.ones([linear_layer_1_output.get_shape()[-1]]))
beta = tf.Variable(tf.zeros([linear_layer_1_output.get_shape()[-1]]))
batch_mean, batch_var = tf.nn.moments(linear_layer_1_output,[0])
linear_layer_1_output = tf.nn.batch_normalization(linear_layer_1_output, batch_mean, batch_var, beta, gamma, epsilon)

linear_layer_relu = tf.nn.relu(linear_layer_1_output)

# Define the Second Linear Layer
initializer = tf.truncated_normal([100, 10], stddev=0.1)
weights_linear_2 = tf.get_variable("Wl2", initializer=initializer)
rand_initializer = tf.random_uniform_initializer(-1, 1)
bias_linear_2 = tf.get_variable("bl2", [10], initializer=rand_initializer)

linear_layer_2_output = tf.matmul(linear_layer_relu,weights_linear_2) + bias_linear_2
# Batch Normalisation for Second Linear Layer
gamma = tf.Variable(tf.ones([linear_layer_2_output.get_shape()[-1]]))
beta = tf.Variable(tf.zeros([linear_layer_2_output.get_shape()[-1]]))
batch_mean, batch_var = tf.nn.moments(linear_layer_2_output,[0])
linear_layer_2_output = tf.nn.batch_normalization(linear_layer_2_output, batch_mean, batch_var, beta, gamma, epsilon)

logits = linear_layer_2_output

# And define the loss based on the logits extracted
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, gold_output_placeholder))
# Optimisation function
learning_rate = 0.001
opt_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

## Prediction Function definition
prediction_softmax = tf.nn.softmax(logits)
prediction_classes = tf.argmax(prediction_softmax,1)
prediction_gold = tf.argmax(gold_output_placeholder,1)

correct_predictions = tf.equal(prediction_classes, prediction_gold)
prediction_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

############## MODEL DEFINED, TRAINING AND TESTING FOLLOWS ################

EPOCHS = 30
BATCH_SIZE = 512
train_errors = []
test_errors = []
n = 55000
ntest = 10000

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if TRAIN_MODEL:
        for epoch in range(EPOCHS):
            print('Starting epoch', epoch)
            curr_loss = 0
            for step in range(n // BATCH_SIZE):
                #print('     Step:', step)
                inputs_batch = train_images[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
                gold_batch = train_labels[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
                the_dict = { input_placeholder: inputs_batch , gold_output_placeholder: gold_batch}
                _, curr_loss_save = sess.run([opt_op, loss], the_dict)
                curr_loss += curr_loss_save
            # Predictions for both train and test sets, for error rate calculation
            print('Finished epoch', epoch)
            print('Training Loss = ', curr_loss)

            print('Training Set Accuracy :')
            aggregator = 0
            for step in range(n // BATCH_SIZE):
            # Calculation of final Error Rates
                the_dict = { input_placeholder: train_images[step * BATCH_SIZE: (step + 1) * BATCH_SIZE] ,
                             gold_output_placeholder: train_labels[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]}
                #print(1-sess.run(prediction_accuracy, feed_dict=the_dict))
                aggregator += (sess.run(prediction_accuracy, feed_dict=the_dict))
            print(aggregator / (n // BATCH_SIZE))

            print('Test Set Accuracy :')
            aggregator = 0
            for step in range(ntest // BATCH_SIZE):
                the_dict = { input_placeholder: test_images[step * BATCH_SIZE: (step + 1) * BATCH_SIZE] ,
                             gold_output_placeholder: test_labels[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]}
                #print(1-sess.run(prediction_accuracy, feed_dict=the_dict))
                aggregator += (sess.run(prediction_accuracy, feed_dict=the_dict))
            print(aggregator / (ntest // BATCH_SIZE))
    else:
        saver.restore(sess, "./models/rnn_model")

    aggregator = 0
    for step in range(n // BATCH_SIZE):
    # Calculation of final Error Rates
        the_dict = { input_placeholder: train_images[step * BATCH_SIZE: (step + 1) * BATCH_SIZE] ,
                     gold_output_placeholder: train_labels[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]}
        #print(1-sess.run(prediction_accuracy, feed_dict=the_dict))
        aggregator += (sess.run(prediction_accuracy, feed_dict=the_dict))
    print(aggregator / (n // BATCH_SIZE))

    aggregator = 0
    theLoss = 0
    for step in range(ntest // BATCH_SIZE):
        the_dict = {input_placeholder: test_images[step * BATCH_SIZE: (step + 1) * BATCH_SIZE],
                    gold_output_placeholder: test_labels[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]}
        aggregator += (sess.run(prediction_accuracy, feed_dict=the_dict))
        theLoss += (sess.run(loss, feed_dict=the_dict))
    print(aggregator / (ntest // BATCH_SIZE))

    print('Testing Loss = ', theLoss / (ntest // BATCH_SIZE))


    #y_hat = gold_test.eval(feed_dict=the_dict)
    #y_pred = prediction_test.eval(feed_dict=the_dict)
    if TRAIN_MODEL:
        #plt.plot(range(EPOCHS), train_errors)
        #plt.plot(range(EPOCHS), test_errors)
        saver.save(sess, "./models/rnn_model")
