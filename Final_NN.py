import tensorflow as tf
from sklearn.model_selection import train_test_split

class FinalNetwork:

    def __init__(self):
        self.sess = tf.Session

    def train(self, input_num, output_num, hidden_num, input, test_size):

        # Get predictions and labels
        values = []
        labels = []

        for vector in input:
            pred_list = [vector[0], vector[1], vector[2]]
            label = [vector[3]]
            values.append(pred_list)
            labels.append(label)

        # Separate data
        inst_values, inst_labels, test_values, test_labels = train_test_split(values, labels, random_state=42, test_size=test_size)

        # Optmization variables
        learning_rate = 0.5
        epochs = 10
        batch_size = 100

        # Data placeholders
        x = tf.placeholder(tf.float32, [None, input_num])
        y = tf.placeholder(tf.float32, [None, output_num])

        # Weights and biases connecting input to hidden layer
        W_hidden = tf.Variable(tf.random_normal([input_num, hidden_num], stddev=0.01)) #stddev=0.03
        b_hidden = tf.Variable(tf.random_normal([hidden_num]), name='b1')

        # Weights and biases connecting hidden layer to output
        W_output = tf.Variable(tf.random_normal([hidden_num, output_num], stddev=0.01)) #stddev=0.03
        b_output = tf.Variable(tf.random_normal([output_num]), name='b2')

        # Calculate output of hidden layer
        hidden_out = tf.add(tf.matmul(x, W_hidden), b_hidden)
        hidden_out = tf.nn.relu(hidden_out)

        # Calculate hidden layer output
        y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W_output), b_output))

        # Cost function
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

        # Add optimizer
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Setup the initialisation operator
        init_op = tf.global_variables_initializer()

        # Define an accuracy assesment operation
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Start the session
        self.sess.run(init_op)

        # Iterate epochs times
        for epoch in range(epochs):
            #_, c = sess.run([train_step, cost], feed_dict={x: inst_values, y: inst_labels})
            train_step.run(feed_dict={x: inst_values, y: inst_labels})

        # Print accuracy of model
        print(self.sess.run(accuracy, feed_dict={x: test_values, y_: test_labels}))

    def test(self, test_input):

        # Divide values and labels
        values = []
        labels = []

        # Get predictions and labels
        for vector in test_input:
            pred_list = [vector[0], vector[1], vector[2]]
            label = [vector[3]]
            values.append(pred_list)
            labels.append(label)

        # Define an accuracy assesment operation
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(self.sess.run(accuracy, feed_dict={x: values, y_: labels}))
