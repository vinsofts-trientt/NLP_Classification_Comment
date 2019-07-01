import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
from keras.losses import binary_crossentropy
from keras import backend as K
from pyvi import ViTokenizer
import re

EMBEDDING_FILES = [
    # '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    'D:/NLP/crawl-300d-2M.vec/crawl-300d-2M.vec',
    # '../input/fasttext/glove.840B.300d.txt'
]

BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 220

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


def build_model(embedding_matrix):
    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    # aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=[result])
    model.compile(loss=['binary_crossentropy'], optimizer='adam',metrics=['accuracy'])
    return model
 #buil model---------------------------- 


# preprocess data----------- 
def preprocess(text):
    text = re.sub('[\n!,.?@#]', '', text)
    text = text.lower()
    text = ViTokenizer.tokenize(text)
    # emoticons = re.findall(r"(?:|;|=)(?:-)?(?:\)\(|D|P)", text)
    # text = re.sub(r"[\W]+", " ", text.lower()) + " ".join(emoticons).replace('-', '')
    # text = re.sub("\n", ' ', text)
    return text
# preprocess data----------- 

#read data----------
train = pd.read_csv('D:/NLP/Toxic_Comment_Vie/data/train.txt',sep='\t', header=None)
test = pd.read_csv('D:/NLP/Toxic_Comment_Vie/data/test.txt',sep='\t', header=None)
xtrain1 = train.head(20)
xtrain2 = test.head(20)

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
    print(row[0])
    if row[0][0:5] != "test_":
        xtest.append(preprocess(row[0]))

print(xtrain)
print(ytrain)
#read data----------


tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(xtrain + xtest)  #tạo ra 1 từ điển#tạo ra 1 từ điển
xtrain = tokenizer.texts_to_sequences(xtrain)
xtest = tokenizer.texts_to_sequences(xtest)
xtrain = sequence.pad_sequences(xtrain, maxlen=MAX_LEN)  #chuẩn hóa các câu có cùng 1 chiều dài
xtest = sequence.pad_sequences(xtest, maxlen=MAX_LEN)


embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)


model = build_model(embedding_matrix)
model.fit(
    xtrain,
    ytrain,
    batch_size=BATCH_SIZE,
    epochs=6,
    verbose=1,
    callbacks=[
        LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** 1))
    ]
)
model.save("model.h5")

