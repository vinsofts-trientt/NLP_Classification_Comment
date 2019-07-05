import numpy as np
import pandas as pd
from keras.models import Model
# from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
# from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
# from keras.callbacks import LearningRateScheduler
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.models import load_model
import re
from pyvi import ViTokenizer
# from keras.losses import mean_squared_abs_error
MAX_LEN = 220
# x_test = ["Chất lượng sản phẩm tuyệt vời"]
x_test = ["shop Chính quá ok, rất hài lòng sản phẩm :))"]
def preprocess(text):
    text = re.sub('[\n!,.?@#]', '', text)
    text = text.lower()
    text = ViTokenizer.tokenize(text)
    # emoticons = re.findall(r"(?:|;|=)(?:-)?(?:\)\(|D|P)", text)
    # text = re.sub(r"[\W]+", " ", text.lower()) + " ".join(emoticons).replace('-', '')
    # text = re.sub("\n", ' ', text)
    return text

train = pd.read_csv('D:/NLP/Toxic_Comment_Vie/data/train.txt',sep='\t', header=None)
test = pd.read_csv('D:/NLP/Toxic_Comment_Vie/data/test.txt',sep='\t', header=None)
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

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(xtrain + xtest)  #tạo ra 1 từ điển#tạo ra 1 từ điển
x_test = tokenizer.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
print("x_test",x_test)
model = load_model("model.h5")
predict = model.predict(np.array(x_test))
if predict[0][0] > 0.5:
    print("thô tục")
elif predict[0][0] < 0.5:
    print("ok")
print(predict)