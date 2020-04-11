from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# print(tf.__version__)

# import tensorflow as tf
# tf.test.gpu_device_name()


df = pd.read_excel('NLP_data_scientist_test/data/Entity_sentiment_trainV2.xlsx')


"""## **PREPROCESSING**"""

# creating input examples

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['Sentiment'])
df['entity_label'] = encoder.fit_transform(df['Entity'])
mask = np.random.rand(len(df)) < 0.7 #spliting 70% of data for training and 30% for testing 

train = df[mask]
test = df[~mask]

DATA_COLUMN = 'Sentence'
ENTITY_COLUMN = 'Entity'
LABEL_COLUMN = 'label'
# list of labels pos/ neg
label_list = [0,1]

# BERT's run_classifier code to create examples from the data
train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = x[ENTITY_COLUMN], 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = x[ENTITY_COLUMN], 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

"""## **Loading BERT**"""

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

"""**Tokenizing**"""

tokenizer = create_tokenizer_from_hub_module()

# Sequences set to be at most 32 tokens long.
MAX_SEQ_LENGTH = 32

# train and test features converted to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

"""**create_model**"""

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  """Creates a classification model."""

  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)


  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout to prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # one-hot encoding labels
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # for predicting, predicted labels and the probabiltiies are needed.
    if is_predicting:
      return (predicted_labels, log_probs)

    # for train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)

# Model function

# model_fn_builder creates our true model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predicted_labels)
        auc = tf.metrics.auc(
            label_ids,
            predicted_labels)
        recall = tf.metrics.recall(
            label_ids,
            predicted_labels)
        precision = tf.metrics.precision(
            label_ids,
            predicted_labels) 
        true_pos = tf.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.metrics.true_negatives(
            label_ids,
            predicted_labels)   
        false_pos = tf.metrics.false_positives(
            label_ids,
            predicted_labels)  
        false_neg = tf.metrics.false_negatives(
            label_ids,
            predicted_labels)
        return {
            "eval_accuracy": accuracy,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn

"""**OUTPUT_DIR**"""

# Assigning Output Directory

OUTPUT_DIR = 'OUTPUT_DIR'#@param {type:"string"}
DO_DELETE = False #@param {type:"boolean"}

if DO_DELETE:
  try:
    tf.gfile.DeleteRecursively(OUTPUT_DIR)
  except:
    # Doesn't matter if the directory didn't exist
    pass
tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

"""**CONFIG**"""

# Compute train and warmup steps from batch size
MAX_SEQ_LENGTH = 32
BATCH_SIZE = 50
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 6
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 252

# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify outpit directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir= OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE},
  model_dir = OUTPUT_DIR)


def getPrediction(in_sentences,entities):
  labels = ["Negative", "Positive"]
  input_examples = [run_classifier.InputExample(guid="", text_a = sent, text_b = ent, label = 0) for sent,ent in zip(in_sentences,entities)] # here, "" is just a dummy label
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn)
  return [({"Sentence":sentence,"Entity":entity, "pred_prob":prediction['probabilities'], "sentiment":labels[prediction['labels']]}) for sentence, entity, prediction in zip(in_sentences,entities, predictions)]

sent = ["the website was great", "the movie was good but I didn't like it"]
ent = ["website","movie"]
predictions = getPrediction(sent,ent)

predictions



# libs
from io import StringIO
import json
import flask
from flask import Flask, request
import time
from flask import jsonify

def __init__(self, sent, entity):

    self.sentences = sentences
    self.entities = entities

# Model Inference
def sentiment_scores(json):
    """
    calls getprediction function 
    """
    results = getpredictions(json['sentences'],json['entities'])
    return results


app = Flask(__name__)

@app.route('/ping', methods=['GET'])
@app.route('/', methods=['POST'])
def sentiment_analysis():

    if flask.request.content_type == 'application/json':
        input_json = flask.request.get_json()
        print("Input json")
        print(input_json)
    else:
        return flask.Response(response='Content type should be application/json', status=415, mimetype='application/json')

    # Get the response
    response = sentiment_scores(input_json)

    out = StringIO()
    json.dump(response, out)
    return flask.Response(response=out.getvalue(), status=200, mimetype='application/json')


if __name__ == '__main__':

    app.run(port=8000)