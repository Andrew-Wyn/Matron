from __future__ import print_function
from sklearn import metrics
from IPython import display
import tensorflow as tf

import collections
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nltk

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

from tensorflow.python.data import Dataset


# wrap into a class for future improvements
class DNN_Wrapper:
    def __init__(self, units):

        text_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key="text", vocabulary_list=vocabulary)

        my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
            my_optimizer, 5.0)

        self.classifier = tf.estimator.DNNClassifier(                                      
            feature_columns= [tf.feature_column.indicator_column(text_feature_column)],
            hidden_units= units,
            optimizer= my_optimizer                                                
        )


    def train(self, input_function, steps):
        try:
            self.classifier.train(
                input_fn=input_function,
                steps=100)
        except ValueError as err:
            print(err) 
        
    
    def evaluate(self, input_function, steps):
        try:
            return self.classifier.evaluate(
                    input_fn=input_function,
                    steps=1)
        except ValueError as err:
            print(err) 
    
    def predict(self):
        validation_predictions = list(self.classifier.predict(input_fn=lambda: _input_fn_not_tfr([tfrecord_test_path])))
        print("sdasd")
        return np.array([item['probabilities'] for item in validation_predictions]) # dovrei tornare il valore piu alto ??
        

def _parse_function(record):
    """Extracts features and labels.

    Args:
      record: File path to a TFRecord file    
    Returns:
      A `tuple` `(labels, features)`:
        features: A dict of tensors representing the features
        labels: A tensor with the corresponding labels.
    """
    features = {
        # terms are strings of varying lengths
        "text": tf.VarLenFeature(dtype=tf.string),
        # labels are 0 or 1
        "spam": tf.FixedLenFeature(shape=[1], dtype=tf.int64)
    }

    parsed_features = tf.parse_single_example(record, features)

    terms = parsed_features['text'].values
    labels = parsed_features['spam']

    return {'text': terms}, labels


# Create an input_fn that parses the tf.Examples from the given files,
# and split them into features and targets.
def _input_fn(input_filenames, num_epochs=None, shuffle=True):

    # Same code as above; create a dataset and map features and labels.
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    # Our feature data is variable-length, so we pad and batch
    # each field of the dataset structure to whatever size is necessary.
    ds = ds.padded_batch(1, ds.output_shapes)

    ds = ds.repeat(num_epochs)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()

    sess = tf.compat.v1.Session()
    print(sess.run(features))
    sess.close()

    return features, labels


# TODO
def _input_fn_not_tfr(input_filenames, num_epochs=1, shuffle=True):

    # Same code as above; create a dataset and map features and labels.
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    # Our feature data is variable-length, so we pad and batch
    # each field of the dataset structure to whatever size is necessary.
    ds = ds.padded_batch(1, ds.output_shapes)

    ds = ds.repeat(num_epochs)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()

    sess = tf.compat.v1.Session()
    print(sess.run(features))
    sess.close()

    return features


def _get_stemmed_from_dataframe(dataframe :pd.DataFrame) -> set:
    vocabulary = set()

    for r in dataframe.text.str.lower():
        r = tokenizer.tokenize(r)
        for w in r:
            if w not in excess_words:
                vocabulary.add(ps.stem(w))
    
    return vocabulary

def _write_tfrecord(path :str = "void.tfrecords", data_frame :pd.DataFrame = None):
    # write a tfrecord file
    with tf.io.TFRecordWriter(path) as writer:
        for row in data_frame.values:
            features, label = row[:-1], row[-1]
            example = tf.train.Example()
            example.features.feature["text"].bytes_list.value.extend(np.asarray(word_tokenize(features[0])).astype(bytes))
            example.features.feature["spam"].int64_list.value.append(label)
            writer.write(example.SerializeToString())

if __name__ == "__main__":

    nltk.download('stopwords')
    nltk.download('punkt')

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    ps = PorterStemmer() # PorterStemmer is necessary to stem the words (remove the infections (prefixes, suffixes and derivations))
    tokenizer = RegexpTokenizer(r'\w+') # remove automatically the punctualizations

    excess_words = set(stopwords.words('english'))
    excess_words.add("subject")
    excess_words.add("_")

    csv_dataframe = pd.read_csv("email.csv")
    
    vocabulary = _get_stemmed_from_dataframe(csv_dataframe)


    tfrecord_training_path = "training.tfrecords"
    tfrecord_test_path = "test.tfrecords"

    csv_dataframe = csv_dataframe.reindex(
        np.random.permutation(csv_dataframe.index)
        )

    _write_tfrecord(path = tfrecord_training_path, data_frame = csv_dataframe.head(1200))
    _write_tfrecord(path = tfrecord_test_path, data_frame = csv_dataframe.head(1))

    ##################### DNN ###############################
    dnn = DNN_Wrapper([20,20])                                                                           

    
    dnn.train(
        input_function=lambda: _input_fn([tfrecord_training_path]),
        steps = 100
        )

    evaluation_metrics = dnn.evaluate(
        input_function=lambda: _input_fn([tfrecord_training_path]),
        steps=1)

    print("Training set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")

    evaluation_metrics = dnn.evaluate(
        input_function=lambda: _input_fn([tfrecord_test_path]),
        steps=1)

    print("Test set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")
    
    # dont do nothing 
    #try:
    
    for i in dnn.predict():
        print(i)
    #except ValueError as err:
    #    print(err) 
    