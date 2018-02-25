#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf

MAX_DOCUMENT_LENGTH = 140
EMBEDDING_SIZE = 50
n_words = 0
MAX_LABEL = 15
WORDS_FEATURE = 'words'  # Name of the input words feature.


class LSTM:
  def __init__(self):
  	self.testing_data_for_cm = []
  	self.testing_labels_for_cm = []
  	print("passed init!")

  def estimator_spec_for_softmax_classification(
      self, logits, labels, mode):
    """Returns EstimatorSpec instance for softmax classification."""
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={
              'class': predicted_classes,
              'prob': tf.nn.softmax(logits)
          })

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


  def rnn_model(self, features, labels, mode):
    # Convert indexes of words into embeddings.
    # Creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(
        features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for softmax
    # classification over output classes.
    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
    return estimator_spec_for_softmax_classification(
        logits=logits, labels=labels, mode=mode)

  # @param list of training instances
  def train(train_list):  
    tf.logging.set_verbosity(tf.logging.INFO)

    # Get the labels.
    train_dict = dict()
    labels = set()

    for inst in train_list: 
        l = inst.getLabel()
        if l not in labels:
            labels.add(l)
            train_dict[l] = []
        train_dict[l].append(inst)

    # Define training data
    x_train = train_dict
    y_train = labels

    # Process the vocabulary.
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform(x_train)

    x_train = np.array(list(x_transform_train))

    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    # Build the model.
    model_fn = rnn_model
    classifier = tf.estimator.Estimator(model_fn=model_fn)

    # Train the model.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_train},
        y=y_train,
        batch_size=len(x_train),
        num_epochs=None,
        shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=100)

    # Save the model.
    saved = tf.train.Saver()

      
# @param list of testing instances
  def batchTest(test_list):  
    tf.logging.set_verbosity(tf.logging.INFO)

    # Get the labels.
    test_dict = dict()
    labels = set()

    for inst in test_list: 
        l = inst.getLabel()
        if l not in labels:
            labels.add(l)
            train_dict[l] = []
        test_dict[l].append(inst)

    # Define testing data
    x_test = test_dict
    y_test = labels

    self.testing_data_for_cm = x_test
    self.testing_labels_for_cm = y_test

    # Load trained model.
    with tf.Session() as sess:
      # Restore variables from disk.
      saver.restore(sess, "/tmp/model.ckpt")

      # Predict.
      test_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={WORDS_FEATURE: x_test},
          y=y_test,
          num_epochs=1,
          shuffle=False)
      predictions = classifier.predict(input_fn=test_input_fn)
      y_predicted = np.array(list(p['class'] for p in predictions))
      y_predicted = y_predicted.reshape(np.array(y_test).shape)

      # Score with sklearn.
      score = metrics.accuracy_score(y_test, y_predicted)
      print('Accuracy (sklearn): {0:f}'.format(score))

      # Score with tensorflow.
      scores = classifier.evaluate(input_fn=test_input_fn)
      print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))

      # Save the model.
      saved = tf.train.Saver()

  def getConfusionMatrix(self):  
	return tf.confusion_matrix(labels=self.testing_data_for_cm, predictions=self.testing_labels_for_cm, num_classes= None)

def main():
    test = LSTM()
    #test.train()
    #test.getConfusionMatrix()
    

if __name__ =="__main__":
  main()
