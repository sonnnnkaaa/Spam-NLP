import sys
import os
import re
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers, Sequential
import tensorflow as tf


MAX_WORDS = 10000 # Maximum number of words for the tokenizer
MAX_LEN = 50      # Maximum sequence length
RANDOM_STATE = 42 # Seed
TEST_SIZE = 0.2   # 20% for test
BATCH_SIZE = 32   # Batch size
CV_EPOCHS = 10    # Number of epochs for cross-valiation scores
EPOCHS = 20       # Number of epochs for the main model
N_CLASSES = 2     # Number of classes for classification


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.random.set_seed(RANDOM_STATE) # Setting seed


def text_preprocessing(text):
    """
      Lowers the case of text and removes unnecessary characters.

      Args:
          text (string): the string that needs to be processed

      Returns:
          string: processed string
    """
    text = text.lower()                   # Lower register
    text = re.sub("[^a-zA-Z]", " ", text) # Cleaning
    return text


def tokenizing(data, text_col, tar_col=None, max_words=MAX_WORDS, max_len=MAX_LEN, tokenizer_path=None):
    """
      Tokenizes text and breaks it into sequences.

      Args:
          data (pandas DataFrame): data containing text
          text_col (string): name of column containing text
          tar_col (string): name of column containing labels (optional, default=None)
          max_words (int): number of words for the tokenizer (default=MAX_WORDS)
          max_len (int): sequences length
          tokenizer_path (string): path to the tokenizer if this already exists (optional, default=None)

      Returns:
          ndarray: array with shape (, max_len)
          ndarray: array with shape (,) or None if tar_col is not specified
    """
    data[text_col] = data[text_col].apply(text_preprocessing)                       # Applying text_preprocessing for text column of data
    if tokenizer_path:
        # If tokenizer already exists
        tokenizer = joblib.load(tokenizer_path)                                     # Loading tokenizer
    else:
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")               # Creating tokenizer with OOV token
        tokenizer.fit_on_texts(data[text_col])                                      # Fitting tokenizer on text data
        joblib.dump(tokenizer, "../models/tokenizer.pk1")                           # Saving tokenizer
    sequences = tokenizer.texts_to_sequences(data[text_col])                        # Breaking text into sequences
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post") # Reducing sequences to the same length
    if tar_col:
        # If tar_col specified
        y = data[tar_col].values                                                    # Labels
        return X, y
    return X, None


def create_model():
    """
      Creates and compiles NLP-model.

      Returns:
          Sequential: compiled Keras model
    """

    # Architechture of NLP-model
    nlp_model = Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(input_dim=MAX_WORDS, output_dim=128),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.25),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    # Compiling model
    nlp_model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return nlp_model