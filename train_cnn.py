import time

import tensorflow as tf


########### Convolutional neural network class ############
class ConvNet(object):
    def __init__(self, mode):
        self.mode = mode

    # Read train, valid and test data.
    def read_data(self, train_set, test_set):
        # Load train set.
        trainX = train_set.images
        trainY = train_set.labels

        # Load test set.
        testX = test_set.images
        testY = test_set.labels

        return trainX, trainY, testX, testY

    # Baseline model. step 1
    def model_1(self, X, hidden_size):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.


        W_fc1 = weight_variable([28 * 28, hidden_size])
        b_fc1 = bias_variable([hidden_size])

        h_pool2_flat = tf.reshape(X, [-1, 28 * 28])
        h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #fully connected layer with input of 28*28 and output of hidden size
        return h_fc1

    # Use two convolutional layers.
    def model_2(self, X, hidden_size):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.

        # first convolution with kernel size 40
        W_conv1 = weight_variable([5, 5, 1, 40])
        b_conv1 = bias_variable([40])
        x_image = tf.reshape(X, [-1, 28, 28, 1])
        h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1)

        #max pool layer
        h_pool1 = max_pool_2x2(h_conv1)

       # second convolution payer
        W_conv2 = weight_variable([5, 5, 40, 40])
        b_conv2 = bias_variable([40])
        h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)

        # max pool layer
        h_pool2 = max_pool_2x2(h_conv2)


        W_fc1 = weight_variable([7 * 7 * 40, hidden_size])
        b_fc1 = bias_variable([hidden_size])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])
        h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # fully connected layer
        return h_fc1

    # Replace sigmoid with ReLU.
    def model_3(self, X, hidden_size):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        W_conv1 = weight_variable([5, 5, 1, 40])
        b_conv1 = bias_variable([40])
        x_image = tf.reshape(X, [-1, 28, 28, 1])
        # replaced sigmoid with relu
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 40, 40])
        b_conv2 = bias_variable([40])
        # replaced sigmoid with relu
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 40, hidden_size])
        b_fc1 = bias_variable([hidden_size])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])
        # replaced sigmoid with relu
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        return h_fc1

    # Add one extra fully connected layer.
    def model_4(self, X, hidden_size, decay):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        W_conv1 = weight_variable([5, 5, 1, 40])
        b_conv1 = bias_variable([40])
        x_image = tf.reshape(X, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 40, 40])
        b_conv2 = bias_variable([40])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 40, hidden_size])
        b_fc1 = bias_variable([hidden_size])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        regularizer1 = tf.nn.l2_loss(W_fc1)# Applying l2 regularization
        W_fc2 = weight_variable([hidden_size, hidden_size])
        b_fc2 = bias_variable([hidden_size])

        regularizer2 = tf.nn.l2_loss(W_fc2)# Applying l2 regularization

        h_fc12 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2) # second fully connected layer
        finalRegularizer = regularizer1 + regularizer2
        return h_fc12, finalRegularizer

    # Use Dropout now.
    def model_5(self, X, hidden_size, is_train):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        W_conv1 = weight_variable([5, 5, 1, 40])
        b_conv1 = bias_variable([40])
        W_conv2 = weight_variable([5, 5, 40, 40])
        b_conv2 = bias_variable([40])
        W_fc1 = weight_variable([7 * 7 * 40, hidden_size])
        b_fc1 = bias_variable([hidden_size])
        W_fc2 = weight_variable([hidden_size, hidden_size])
        b_fc2 = bias_variable([hidden_size])

        x_image = tf.reshape(X, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])

        h_fc11 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc11_drop = tf.nn.dropout(h_fc11, keep_prob=0.5)

        h_fc12 = tf.nn.relu(tf.matmul(h_fc11_drop, W_fc2) + b_fc2)


        h_fc1_drop = tf.nn.dropout(h_fc12, keep_prob=0.5) # Added dropout layer

        return h_fc1_drop

    # Entry point for training and evaluation.
    def train_and_evaluate(self, FLAGS, train_set, test_set):
        class_num = 10
        num_epochs = FLAGS.num_epochs
        batch_size = FLAGS.batch_size
        learning_rate = FLAGS.learning_rate
        if self.mode == 2:
            learning_rate = 0.001
        hidden_size = FLAGS.hiddenSize

        decay = FLAGS.decay

        trainX, trainY, testX, testY = self.read_data(train_set, test_set)

        input_size = trainX.shape[1]
        train_size = trainX.shape[0]
        test_size = testX.shape[0]

        trainX = trainX.reshape((-1, 28, 28, 1))
        testX = testX.reshape((-1, 28, 28, 1))

        with tf.Graph().as_default():
            # Input data
            X = tf.placeholder(tf.float32, [None, 28, 28, 1])
            Y = tf.placeholder(tf.int32, [None, 10])
            is_train = tf.placeholder(tf.bool)

            # model 1: base line
            if self.mode == 1:
                features = self.model_1(X, hidden_size)



            # model 2: use two convolutional layer
            elif self.mode == 2:
                features = self.model_2(X, hidden_size)

            # model 3: replace sigmoid with relu
            elif self.mode == 3:
                features = self.model_3(X, hidden_size)


            # model 4: add one extral fully connected layer
            elif self.mode == 4:
                features, regularizer = self.model_4(X, hidden_size, decay)

            # model 5: utilize dropout
            elif self.mode == 5:
                features = self.model_5(X, hidden_size, is_train)

            # ======================================================================
            # Define softmax layer, use the features.
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign calculated value to logits after code implementation.
            W_fc2 = weight_variable([hidden_size, class_num])
            b_fc2 = bias_variable([class_num])

            if self.mode == 1:
                logits = tf.matmul(features, W_fc2) + b_fc2
            else:
                logits = tf.nn.softmax(tf.matmul(features, W_fc2) + b_fc2)

            # ======================================================================
            # Define loss function, use the logits.
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign calculated value to loss after code implementation.

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))  # check here@@@@@
            if self.mode == 4:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits) + 0.01 * regularizer)# Applying regularizer here for loss
            # ======================================================================
            # Define training op, use the loss.
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign calculated value to train_op after code implementation.

            #Tried with the given learning rate but it was not converging so used decay learning rate for good accuracy
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            newlearning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                          2, 0.95, staircase=True)

            if self.mode == 1:
                train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #for model1 learning rate is 0.1 and using SGD
            else:
                train_op = tf.train.AdamOptimizer(newlearning_rate).minimize(loss)

            # ======================================================================
            # Define accuracy op.
            # ----------------- YOUR CODE HERE ----------------------
            #
            correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(logits, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # ======================================================================
            # Allocate percentage of GPU memory to the session.
            # If you system does not have GPU, set has_GPU = False
            #
            has_GPU = True
            if has_GPU:
                gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
                config = tf.ConfigProto(gpu_options=gpu_option)
            else:
                config = tf.ConfigProto()

            # Create TensorFlow session with GPU setting.
            with tf.Session(config=config) as sess:
                tf.global_variables_initializer().run()

                for i in range(num_epochs):
                    print(20 * '*', 'epoch', i + 1, 20 * '*')
                    print(learning_rate)
                    start_time = time.time()
                    s = 0
                    while s < train_size:
                        e = min(s + batch_size, train_size)
                        batch_x = trainX[s: e]
                        batch_y = trainY[s: e]
                        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True, global_step: i})
                        s = e
                    end_time = time.time()
                    print('the training took: %d(s)' % (end_time - start_time))

                    count_example = len(testX)
                    # running in batches of 1000 as running all together was giving error
                    sumAccuracy = 0
                    for offset in range(0, count_example, 1000):
                        batch_x, batch_y = testX[offset:offset + 1000], testY[offset:offset + 1000]
                        execu = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
                        sumAccuracy = sumAccuracy + (execu * len(batch_x))
                    acc = sumAccuracy / count_example
                    print('Test accuracy : %g' % acc)

                return acc


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


