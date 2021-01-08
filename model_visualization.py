import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

from pooling import MaskGlobalMaxPooling1D
from pooling import MaskGlobalAveragePooling1D
from dataset import SimpleTokenizer, find_best_maxlen
from dataset import load_THUCNews_title_label
from dataset import load_weibo_senti_100k
from dataset import load_simplifyweibo_4_moods
from dataset import load_hotel_comment
from dataset import get_class_weight
from heatmap import get_grad_cam_weights, convert_to_heatmap

X, y, classes = load_hotel_comment()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.9, random_state=7788)

class_weight = get_class_weight(y_train)
num_classes = len(classes)
tokenizer = SimpleTokenizer()
tokenizer.fit(X_train)
X_train = tokenizer.transform(X_train)

maxlen = find_best_maxlen(X_train)
maxlen = 128

X_train = sequence.pad_sequences(
    X_train, 
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0.0
)
y_train = tf.keras.utils.to_categorical(y_train)

num_words = len(tokenizer)
embedding_dims = 128

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
x = Embedding(num_words, embedding_dims,
    embeddings_initializer="normal",
    input_length=maxlen,
    mask_zero=True)(inputs)
x = Dropout(0.1)(x)
x = Conv1D(filters=128,
           kernel_size=2,
           padding="same",
           activation="relu",
           strides=1)(x)
x = Conv1D(filters=128,
           kernel_size=3,
           padding="same",
           activation="relu",
           strides=1)(x)

# 提取特征模型
features_model = Model(inputs, x)

x = MaskGlobalMaxPooling1D()(x, mask=mask)
x = Dense(128)(x)
x = Dropout(0.1)(x)
x = Activation("relu")(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
model.summary()

# 特征到类别模型
filters = 128
pred_inputs = tf.keras.Input(shape=(maxlen, filters))
x = pred_inputs
layer_names = ["mask_global_max_pooling1d", "dense", 
               "dropout_1", "activation", "dense_1"]
for name in layer_names:
    x = model.get_layer(name)(x)
outputs = x
pred_model = Model(pred_inputs, outputs)

batch_size = 32
epochs = 10
callbacks = []
model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    class_weight=class_weight
)

markdown_template = '<font color="{}">{}</font>'
def render_color_markdown(text, hexcolors):
    ss = []
    for c, h in zip(text, hexcolors):
        ss.append(markdown_template.format(h, c))
    return "".join(ss)

id_to_classes = {j:i for i,j in classes.items()}
def visualization():
    fd = open("results.md", "w")
    for sample, label in zip(X_test, y_test):
        sample_len = len(sample)
        if sample_len > maxlen:
            sample_len = maxlen

        x = np.array(tokenizer.transform([sample]))
        x = sequence.pad_sequences(
            x, 
            maxlen=maxlen,
            dtype="int32",
            padding="post",
            truncating="post",
            value=0
        )

        y_pred = model.predict(x)[0]
        y_pred_id = np.argmax(y_pred)
        # 预测错误的样本跳过
        if y_pred_id != label:
            print(label, sample)
            continue
            
        # 预测权重
        weights = get_grad_cam_weights(
            x,
            features_model,
            pred_model
        )
        heatmap = convert_to_heatmap(weights)[:sample_len]
        rstring = render_color_markdown(sample, heatmap)
        print(
            rstring, " =>", id_to_classes[y_pred_id],
            end="\n\n\n",
            file=fd,
            flush=True
        ) # 命令行 typora results.md 可观察结果
        # time.sleep(1)
        input() # 按回车预测下一个样本

if __name__ == "__main__":
    visualization()

