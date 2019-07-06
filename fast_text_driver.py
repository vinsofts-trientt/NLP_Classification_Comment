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
EMBEDDING_FILES = [
    'cc.vi.300.vec/cc.vi.300.vec'
    # 'glove6b300dtxt/glove.6B.300d.txt',
    # 'crawl-300d-2M.vec/crawl-300d-2M.vec'

]
num_filters = 64
BATCH_SIZE = 256
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
# EPOCHS = 4
MAX_LEN = 100
weight_decay = 1e-4
# embedding---------------------------
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path,encoding='utf-8') as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix

 # embedding--------------------------- 

 #buil model----------------------------   
# def custom_loss(y_true, y_pred):
#     return binary_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_pred) * y_true[:,1]


# def build_model(embedding_matrix):
#     words = Input(shape=(MAX_LEN,))
#     x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
#     x = SpatialDropout1D(0.3)(x)
#     x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x) #lsrm 2 chiều
#     x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)

#     hidden = concatenate([
#         GlobalMaxPooling1D()(x),
#         GlobalAveragePooling1D()(x),
#     ])
#     hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
#     hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
#     result = Dense(1, activation='sigmoid')(hidden)
#     # aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
#     model = Model(inputs=words, outputs=[result])
#     model.compile(loss=['binary_crossentropy'], optimizer='adam',metrics=['accuracy'])
#     return model

# def build_model(embedding_matrix):
#     words = Input(shape=(MAX_LEN,))
#     x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
#     x = SpatialDropout1D(0.2)(x)
#     x = Bidirectional(GRU(80, return_sequences=True))(x)
#     avg_pool = GlobalAveragePooling1D()(x)
#     max_pool = GlobalMaxPooling1D()(x)
#     conc = concatenate([avg_pool, max_pool])
#     outp = Dense(1, activation="sigmoid")(conc)
    
#     model = Model(inputs=words, outputs=[outp])
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])

#     return model



def build_model(embedding_matrix):
    words = Input(shape=(MAX_LEN,))
    model = Sequential()
    model.add(Embedding(*embedding_matrix.shape,weights=[embedding_matrix], trainable=False))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(1, activation='sigmoid'))  #multi-label (k-hot encoding)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model 

 #buil model---------------------------- 


# preprocess data----------- 
def preprocess(text):
    # text = re.sub('[\n!,.?@#]', '', text)
    text = text.lower()
    list_happen = ["😊","❤️","😁","😄","😆","😍","🤣","😂","🤩","😚","😋",'😜',"😝","🤗",":)",":}","^^",";)",
    "👌","=))","😅","👍","👍🏻","💕","❤","👏","💟","<3",":D",":P","^_^","😉","✌️"]
    list_sad = ["😡","🤔","🤨","😐","😏","😒","😶","🙄","😌","😔","🤕","🤒","👿","🤬","😤",'😫',"😩","😭",":(","😈","-_-","👎"]
    for happen in list_happen:          
        text = text.replace(happen, "vui")
    for sad in list_sad:          
        text = text.replace(sad, "tệ")
    # text = ViTokenizer.tokenize(text)
    return text
# preprocess data----------- 

#read data----------
# train = pd.read_csv('D:/NLP/Toxic_Comment_Vie/data/train.txt',sep='\t', header=None)
# test = pd.read_csv('D:/NLP/Toxic_Comment_Vie/data/test.txt',sep='\t', header=None)
train = pd.read_csv('data/train.txt',sep='\t', header=None)
test = pd.read_csv('data/test.txt',sep='\t', header=None)
xtrain1 = train
xtrain2 = test

xtrain = []
ytrain = []
xtest = []
for index,row in xtrain1.iterrows():
    if row[0] == "0" or  row[0] == "1": 
        preprocess(row[0])
        ytrain.append(row[0])
    elif row[0][0:6] != "train_":
        xtrain.append(preprocess(row[0]))


for index,row in xtrain2.iterrows():
    # print(row[0])
    if row[0][0:5] != "test_":
        xtest.append(preprocess(row[0]))

# print(xtrain)
# print(ytrain)
#read data----------


#tính avg  số từ trong 1 câu
# listAvg = []
# for i in xtrain:
#     listAvg.append(len(i.split(" ")))

# avg = statistics.mean(listAvg)
# print("avg",avg)

#tính avg  số từ trong 1 câu


X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.1, random_state=42)

# print("X_test",X_test) 
# print("X_test",type(X_test))
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(xtrain + xtest)  #tạo ra 1 từ điển#tạo ra 1 từ điển
xtrain = tokenizer.texts_to_sequences(X_train)
xtest = tokenizer.texts_to_sequences(X_test)
xtrain = sequence.pad_sequences(xtrain, maxlen=MAX_LEN)  #chuẩn hóa các câu có cùng 1 chiều dài
xtest = sequence.pad_sequences(xtest, maxlen=MAX_LEN)
# print("xtrain",xtrain.shape)

embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)

print("embedding_matrix",embedding_matrix)
print("embedding_matrix",embedding_matrix.shape)
model = build_model(embedding_matrix)
model.fit(
    np.array(xtrain),
    np.array(y_train),
    batch_size=BATCH_SIZE,
    epochs=50,
    validation_data=(np.array(xtest), np.array(y_test)), 
    verbose=1
    # tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)  # hiển thị tensorBoard
    # callbacks=[LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** 1))]
)
model.save("model.h5")