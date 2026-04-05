import sys
import os
import re
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers, Sequential
import tensorflow as tf


MAX_WORDS = 10000
MAX_LEN = 50
RANDOM_STATE = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
CV_EPOCHS = 10
EPOCHS = 20
N_CLASSES = 2


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.random.set_seed(RANDOM_STATE)


def text_preprocessing(text):
    text = text.lower()                   # lower register
    text = re.sub("[^a-zA-Z]", " ", text) # cleaning
    return text


def tokenizing(data, text_col, tar_col=None, max_words=MAX_WORDS, max_len=MAX_LEN, tokenizer_path=None):
    data[text_col] = data[text_col].apply(text_preprocessing)
    if tokenizer_path:
        tokenizer = joblib.load(tokenizer_path)
    else:
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>") # tokenizer
        tokenizer.fit_on_texts(data[text_col])                 # fitting tokenizer
        joblib.dump(tokenizer, "../models/tokenizer.pk1")
    sequences = tokenizer.texts_to_sequences(data[text_col])                # sequences
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    if tar_col:
        y = data[tar_col].values
        return X, y
    return X, None


def create_model():
    nlp_model = Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(input_dim=MAX_WORDS, output_dim=128),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.25),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    nlp_model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return nlp_model