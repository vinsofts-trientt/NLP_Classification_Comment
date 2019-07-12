import numpy as np
import pandas as pd
from keras.models import Model
from keras.models import Sequential
from keras import regularizers
from keras import optimizers
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D,GRU,Conv1D, MaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
from keras.losses import binary_crossentropy
from keras import backend as K
from pyvi import ViTokenizer
import re
from keras.models import load_model
from sklearn.model_selection import train_test_split
from statistics import *
from word_embedding.word2vec_gensim import Word2Vec
import keras
word2vec_model = Word2Vec.load('pretrained_word2vec.bin')
print("word2vec_model",word2vec_model['kiểm_tra'])