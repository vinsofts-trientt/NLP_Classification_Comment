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
    # '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    # 'D:/NLP/crawl-300d-2M.vec/crawl-300d-2M.vec',
    # 'D:/NLP/Toxic_Comment_Vie/cc.vi.300.vec/cc.vi.300.vec',
    # 'D:/NLP/Toxic_Comment_Vie/cc.vi.300.bin/cc.vi.300.bin',
    # 'D:/NLP/glove6b300dtxt/glove.6B.300d.txt',
    'cc.vi.300.vec/cc.vi.300.vec',
]

# path_train= 'D:/NLP/Toxic_Comment_Vie/data/train.txt'
# path_test = 'D:/NLP/Toxic_Comment_Vie/data/test.txt'
path_train= 'data/train.txt'
path_test = 'data/test.txt'

num_filters = 64
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 100
weight_decay = 1e-4

# embedding---------------------------
word2vec_model = Word2Vec.load('pretrained_word2vec.bin')
#load pre-train
def get_coefs(word, *arr):
    # print("word",word)
    # print("arr",arr)
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):  
    with open(path,encoding='utf-8') as f:
        # print("sss",dict(get_coefs(*line.strip().split(' ')) for line in f))
        return dict(get_coefs(*line.strip().split(' ')) for line in f)
#load result của pre-train
#     example {'-3.690e-02 -1.966e-01 -7.410e-02  1.753e-01  1.750e-02 -9.620e-02
#   1.112e-01  1.287e-01 -1.403e-01  4.460e-02  1.364e-01 -2.050e-01
#   6.780e-02 -1.056e-01  3.300e-03 -3.805e-01 -1.637e-01 -4.780e-02
#  -1.000e-04 -3.730e-02  1.114e-01 -7.360e-02  1.599e-01  6.370e-02
#   8.420e-02 -5.160e-02  7.180e-02  2.443e-01  1.418e-01 -1.560e-01
#  -6.380e-02  2.805e-01 -2.453e-01 -1.450e-02  6.710e-02  1.037e-01','xa'}

def build_matrix(word_index, path):  #embedding các từ trong từ điển
    embedding_index = load_embeddings(path)
    # print("embedding_index",embedding_index.shape)
    # print("vọngthực",embedding_index['vui_vẻ'])
    embedding_matrix = np.zeros((len(word_index) + 1, 100))  #Tạo ra 1 mảng matrix bằng 0 vs shape = (len(word_index) + 1, 300)
    # print("embedding_matrix",embedding_matrix.shape)
    for word, i in word_index.items():
        # print(word)
        print(i)
        try:
            # print("embedding_index[word]",embedding_index[word].shape)
            embedding_matrix[i] = word2vec_model[word]
        # except KeyError: #nếu từ ko có trong từ điển pre train  sẽ bỏ qua
            # print("pass")
            # embedding_matrix[i] = embedding_index[word]  
        except KeyError: #nếu từ ko có trong từ điển pre train  sẽ bỏ qua
            # print("pass")
            pass
    return embedding_matrix

 # embedding--------------------------- 

 #buil model----------------------------   


# def custom_loss(y_true, y_pred):
#     return binary_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_pred) * y_true[:,1]

# LSTM
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

#   GRU
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

# cnn
# def build_model(embedding_matrix):
#     words = Input(shape=(MAX_LEN,))
#     model = Sequential()
#     model.add(Embedding(*embedding_matrix.shape,weights=[embedding_matrix], trainable=False))
#     model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
#     model.add(MaxPooling1D(2))
#     model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
#     model.add(GlobalMaxPooling1D())
#     model.add(Dropout(0.3))
#     model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(Dense(1, activation='sigmoid'))  #multi-label (k-hot encoding)

#     adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#     model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

#     return model 


 #buil model---------------------------- 


# preprocess data----------- 
def preprocess(text):
#     text = re.sub('[\n!,.?@#]', '', text)
    text = text.lower()
    list_happen = ["😊","❤️","😁","😄","😆","😍","🤣","😂","🤩","😚","😋",'😜',"😝","🤗",":)",":}","^^",";)",
    "👌","=))","😅","👍","👍🏻","💕","❤","👏","💟","<3",":D",":P","^_^","😉","✌️"]
    list_sad = ["😡","🤔","🤨","😐","😏","😒","😶","🙄","😌","😔","🤕","🤒","👿","🤬","😤",'😫',"😩","😭",":(","😈","-_-","👎"]
    for happen in list_happen:          
        text = text.replace(happen, " vui")
    for sad in list_sad:          
        text = text.replace(sad, " buồn")

    text = re.sub('[\n!,.?@#?!.,#$%\()*+-/:;<=>@[\\]^_`{|}~`"""“”’∞θ÷α•−β∅³π‘₹´°£€\×™√²—–&0123456789]', '', text)
    text = ViTokenizer.tokenize(text)
    # emoticons = re.findall(r"(?:|;|=)(?:-)?(?:\)\(|D|P)", text)
    # text = re.sub(r"[\W]+", " ", text.lower()) + " ".join(emoticons).replace('-', '')
    # text = re.sub("\n", ' ', text)
    return text
# preprocess data----------- 

#read data----------
train = pd.read_csv(path_train,sep='\t', header=None)
test = pd.read_csv(path_test,sep='\t', header=None)
xtrain1 = train
xtrain2 = test
# xtrain1 = train.head(21)
# xtrain2 = test.head(21)

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

# print("xtrain",xtrain)
# print("xtest",xtest)
# print(ytrain)
# read data----------

# listAvg = []
# for i in xtrain:
#     listAvg.append(len(i.split(" ")))

# avg = mean(listAvg)
# print("avg",avg)


X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)
# print("X_test",X_test) 
# X_train = xtrain

tokenizer = text.Tokenizer(lower=True, filters='')
tokenizer.fit_on_texts(xtrain + xtest)  #tạo ra 1 từ điển tokenizer {'rất': 1, 'phẩm': 2, 'shop': 3, 'sản': 4,...}
# tokenizer.fit_on_texts(xtrain)  #mã hóa 1 data train theo từ điển
print("tokenizer",tokenizer.word_index)
xtrain = tokenizer.texts_to_sequences(X_train)
# print("xtrain",xtrain)
xtest = tokenizer.texts_to_sequences(X_test)
xtrain = sequence.pad_sequences(xtrain, maxlen=MAX_LEN)  #chuẩn hóa các câu có cùng 1 chiều dài
xtest = sequence.pad_sequences(xtest, maxlen=MAX_LEN)
# print("xtrain",xtrain)


embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)

print("*embedding_matrix.shape",*embedding_matrix.shape)

# print("embedding_matrix",embedding_matrix)
# print("embedding_matrix",embedding_matrix.shape)
model = build_model(embedding_matrix)
model.fit(
    np.array(xtrain),
    np.array(y_train),
    batch_size=BATCH_SIZE,
    epochs=10,
    validation_data=(np.array(xtest), np.array(y_test)), 
    verbose=1
    # tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)  # hiển thị tensorBoard
    # callbacks=[LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** 1))]
)
model.save("model.h5")
# model = load_model("model.h5")
# predict = model.predict(np.array(xtest))
# print(predict)


# tokenizer {'rất': 1, 'phẩm': 2, 'shop': 3, 'sản': 4, 'như': 5, 'có': 6, 'mình': 7, 'đẹp': 8, 'và': 9, 'chất': 10, 'hàng': 11, 'đóng': 12, 'gói': 13, 'lượng': 14,
#  'nhưng': 15, 'không': 16, 'đã': 17, 'của': 18, 'ra': 19, 'là': 20, 'chỉ': 21, 'tiền': 22, 'chưa': 23, 'đáng': 24, 'ko': 25, '�': 26, 'chắc': 27, 'chắn': 28,
# 'tuyệt': 29, 'vời': 30, 'khi': 31, 'dây': 32, 'cuốn': 33, 'sách': 34, 'cách': 35, 'sau': 36, 'sự': 37, 'cảm': 38, 'nhận': 39, 'mà': 40, 'cũng': 41, 
# 'lần': 42, 'mua': 43, 'phục': 44, 'vụ': 45, 'nên': 46, 'đọc': 47, 'điện': 48, 'sp': 49, 'k': 50, 'vọng': 51, 'sao': 52, 'nào': 53, 'lại': 54, 'trong': 55,
# 'người': 56, 'đi': 57, 'này': 58, 'đc': 59, 'thì': 60, 'đổi': 61, 'kg': 62, 'truyện': 63, 'theo': 64, 'mô': 65, 'chị': 66, 'vật': 67, 'quá': 68, 'đánh': 69,
# 'màu': 70, 'thất': 71, 'vì': 72, 'nó': 73, 'sẽ': 74, 'nói': 75, 'về': 76, 'những': 77, 'họ': 78, 'phải': 79, 'vào': 80, 'thời': 81, 'đó': 82, 'câu': 83,
# 'chuyện': 84, 'cái': 85, 'thực': 86, 'sâu': 87, 'hơn': 88, 'còn': 89, 'áo': 90, 'gió': 91, 'đợt': 92, 'giao': 93, 'khác': 94, 'vải': 95, 'định': 96, 
# 'mới': 97, 'rồi': 98, 'tốt': 99, 'biết': 100, 'qua': 101, 'quyết': 102, 'được': 103, 'típ': 104, 'quen': 105, 'tả': 106, 'nhàm': 107, 'thích': 108,
# 'kawi': 109, 'bỏ': 110, 'càng': 111, 'các': 112, 'gần': 113, 'ứng': 114, 'tg': 115, 'âm': 116, 'tệ': 117, 'kém': 118, 'dung': 119, 'dc': 120, 'tot': 121,
# 'cam': 122, 'on': 123, '�': 124, 'son': 125, 'mịn': 126, 'lên': 127, 'trên': 128, 'ảnh': 129, 'hộp': 130, 'giày': 131, 'đen': 132, 'tất': 133,
# 'hơi': 134, '1': 135, 'chút': 136, 'kỳ': 137, 'khá': 138, 'nhiều': 139,
# 'hi': 140, 'việc': 141, 'học': 142, 'tập': 143, 'sinh': 144, 'viên': 145, 'trường': 146, 'harvard': 147, 'nỗ': 148, 'lực': 149, 'thế': 150, '4h': 151,
# 'sáng': 152, 'tại': 153, 'thức': 154, 'dậy': 155, 'khắc': 156, 'đấy': 157, 'cả': 158, 'một': 159, 'cần': 160, 'ở': 161, 'đây': 162, 'ẩn': 163, 'dấu': 164, 
# 'để': 165, 'tự': 166, 'bản': 167, 'thân': 168, 'mỗi': 169, 'lòng': 170, 'đơn': 171, 'thuần': 172, 'dạy': 173, 'kĩ': 174, 'năng': 175, 'hầu': 176, 
# 'buồn': 177, 'trước': 178, 'hồng': 179, 'ok': 180, '2': 181, 'mưa': 182, 'ý': 183, 'điều': 184, 'cứng': 185, 'cáp': 186, 'với': 187, 'cố': 188, 
# 'dáng': 189, 'chung': 190, 'hợp': 191, 'túi': 192, 'nhanh': 193, 'đặt': 194, 'buổi': 195, 'tối': 196, 'trưa': 197, 'mai': 198, 'dùng': 199, 
# 'thử': 200, 'tiềnvì': 201, 'ngay': 202, 'sale': 203, 'sử': 204,
# 'dụng': 205, 'nt': 206, 'trao': 207, 'vs': 208, 'đưa': 209, 'giải': 210, 'hoàn': 211, 'time': 212, 'muốn': 213, 'giá': 214, 'gì': 215,
# 'cám': 216, 'ơn': 217, 'đẹpphom': 218, 'oki': 219, 'luônquá': 220, 'ưng': 221, 'chuẩn': 222, 'it': 223, 'so': 224, 'good': 225, 'xong': 226,
# 'xúc': 227, 'vụt': 228, 'trôi': 229, 'mấttruyện': 230, 'cũ': 231, 'thuộcmiêu': 232, 'đặc': 233, 'sắcmột': 234, 'vài': 235, 'tình': 236, 'tiết': 237,
# 'chánvô': 238, 'límình': 239, 'vốn': 240, 'vọngthực': 241, 'muanhân': 242, 'nam': 243, 'kiểu': 244, 'trainhà': 245, 'giàubad': 246, 'boythưng': 247,
# 'chánmình': 248, 'shock':249,
# 'thậtnữ': 250, 'chính': 251, 'tầm': 252, 'thườngkiểu': 253, 'ngoài': 254, 'mạnh': 255, 'mềm': 256, 'bánh': 257, 'bèolúc': 258, 'đầu': 259, 'chứ': 260,
#  'thấy': 261, 'nảnhức': 262, 'hứcmình': 263,
# 'trót': 264, 'hết': 265, 'thôinói': 266, 'đồng': 267, 'lắmmong': 268, 'tác': 269, 'bớt': 270, 'chán': 271, 'bởi': 272, 'lúc': 273, 'thuộc': 274, 'đột': 275,
#  'phá': 276, 'cốt': 277, 'miêu': 278, 'ổ': 279, 'cắm': 280, 'cảncó': 281, 'cản': 282, 'khoảng': 283, 'gỉam': 284, 'kể': 285, '80': 286, 'oko': 287, 
#  'xuyên': 288, 'tảhoàn': 289, 'toàn': 290, 'tường': 291, 'test': 292, '15': 293, '2cm': 294, 'lênshop': 295, 'tư': 296, 'vấn': 297, 'chat': 298, 'yêu': 299, 
#  'cầu': 300, 'khách': 301, 'gọi': 302, 'đứt': 303, 'tùm': 304, 'lum': 305, 'rách': 306, 'trời': 307, 'chỗ': 308, 'lũng': 309, 'cuối': 310, 'gian': 311, 
#  'chậm': 312, 'trẩu': 313, 'nha': 314, 'bạn': 315, 'làm': 316, 'phiền': 317}