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
EMBEDDING_FILES = [
    # 'D:/NLP/Toxic_Comment_Vie/pretrained_word2vec.bin'
    'pretrained_word2vec.bin'
]
# path_train= 'D:/NLP/Toxic_Comment_Vie/data/train.txt'
# path_test = 'D:/NLP/Toxic_Comment_Vie/data/test.txt'
# path_synonym = 'C:/Users/anlan/OneDrive/Desktop/core_nlp-master1/data/sentiment/synonym.txt'
path_train= 'data/train.txt'
path_test = 'data/test.txt'
path_synonym = 'synonym.txt'
word2vec_model = Word2Vec.load(EMBEDDING_FILES[0])
BATCH_SIZE = 256
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
def build_model(input_dim):

        # model = Sequential()

        # model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=input_dim))
        # model.add(Dropout(0.1))
        # model.add(Bidirectional(LSTM(16)))
        # model.add(Dense(1, activation="softmax"))

        # model.compile(loss=['binary_crossentropy'], optimizer='adam',metrics=['accuracy'])
        # return model


        words = Input(shape=(100,100))
        # x = Embedding(input_dim=100,output_dim=100, trainable=False)(words)
        x = SpatialDropout1D(0.3)(words)
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
    
    text = text.lower()
    list_happen = ["😊","❤️","😁","😄","😆","😍","🤣","😂","🤩","😚","😋",'😜',"😝","🤗",":)",":}","^^",";)",
    "👌","=))","😅","👍","👍🏻","💕","❤","👏","💟","<3",":D",":P","^_^","😉","✌️"]
    list_sad = ["😡","🤔","🤨","😐","😏","😒","😶","🙄","😌","😔","🤕","🤒","👿","🤬","😤",'😫',"😩","😭",":(","😈","-_-","👎"]
    for happen in list_happen:          
        text = text.replace(happen, " vui")
    for sad in list_sad:          
        text = text.replace(sad, " buồn")

    text = re.sub('[\n!,.?@#?!.,#$%\()*+-/:;<=>@[\\]^_`{|}~`"""“”’∞θ÷α•−β∅³π‘₹´°£€\×™√²—–&]', '', text)
#     text = preprocess1(text)
    text = ViTokenizer.tokenize(text)
    # emoticons = re.findall(r"(?:|;|=)(?:-)?(?:\)\(|D|P)", text)
    # text = re.sub(r"[\W]+", " ", text.lower()) + " ".join(emoticons).replace('-', '')
    # text = re.sub("\n", ' ', text)
    return text
# preprocess data----------- 

#read data----------
train = pd.read_csv(path_train,sep='\t', header=None)
test = pd.read_csv(path_test,sep='\t', header=None)
# xtrain1 = train.head(21)
# xtrain2 = test.head(21)
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

def load_synonym_dict(file_path):
    sym_dict = dict()
    with open(file_path, 'r',encoding="utf-8") as fr:
        lines = fr.readlines()
    lines = [ln.strip() for ln in lines if len(ln.strip()) > 0]
    for ln in lines:
        words = ln.split(",")
        words = [w.strip() for w in words]
        for word in words[1:]:
            sym_dict.update({word: words[0]})
    return sym_dict

def word_embed_sentences(sentences, max_length=100):
        # sym_dict = load_synonym_dict('C:/Users/anlan/OneDrive/Desktop/core_nlp-master1/data/sentiment/synonym.txt')
        sym_dict = load_synonym_dict(path_synonym)
        print("sym_dict",sym_dict)
        embed_sentences = []
        word_dim = word2vec_model["chán"].shape[0]  #(100,)
        for sent in sentences:
            # print("sent",sent)
            sent = sent.split()
            if sent == []:
                sent = ["xấu"]
            # print(sent)
            embed_sent = []
            for word in sent:  # lặp từng từ trong 1 câu
                # print(" sym_dict", sym_dict)
                # print(word)
                if (sym_dict is not None) and (word.lower() in  sym_dict):
                    replace_word =  sym_dict[word.lower()] #replace các từ đồng nghĩa
                    embed_sent.append( word2vec_model[replace_word])
                elif word.lower() in  word2vec_model:
                    embed_sent.append( word2vec_model[word.lower()])
                else:
                    embed_sent.append(np.zeros(shape=(word_dim), dtype=float))
            if len(embed_sent) > max_length:
                embed_sent = embed_sent[:max_length]
            elif len(embed_sent) < max_length:
                # print("sent",sent)
                embed_sent = np.concatenate((embed_sent, np.zeros(shape=(max_length - len(embed_sent),word_dim), dtype=float)),axis=0)

            # print(embed_sent.shape)
            embed_sentences.append(embed_sent)
        return embed_sentences


embedding_matrix = np.array(word_embed_sentences(xtrain))
# print("embedding_matrix",embedding_matrix.shape)
# print("embedding_matrix.shape[0]",embedding_matrix.shape[0])
# print("np.array(y_train)",np.array(ytrain))
X_train, X_test, y_train, y_test = train_test_split(embedding_matrix, ytrain, test_size=0.1, random_state=42)

model = build_model(input_dim=(embedding_matrix.shape[1], embedding_matrix.shape[2]))
model.fit(X_train, np.array(y_train), batch_size=BATCH_SIZE, epochs=50,validation_data=(X_test,np.array(y_test)),verbose=1)
model.save("model.h5")