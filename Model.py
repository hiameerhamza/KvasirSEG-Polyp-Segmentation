# Kvasir-SEG Polyp Segmentation - Cleaned Script
# Author: Ameer Hamza
# Description: U-Net with Attention & SE Blocks for Polyp Segmentation using TensorFlow and Kvasir-SEG Dataset


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal


input_dir = "C:/Users/Ameer/Downloads/KvasirSEG/images/"
target_dir = "C:/Users/Ameer/Downloads/KvasirSEG/masks/"
#img_size = (128, 128)
num_classes = 2
print("Train set:  ", len(os.listdir(input_dir)))
print("Train masks:", len(os.listdir(target_dir)))


image_ids = []
paths = []
for dirname, _, filenames in os.walk(input_dir):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        paths.append(path)
        image_id = filename.split(".")[0]
        image_ids.append(image_id)

d = {"id": image_ids, "Image_Path": paths}
df = pd.DataFrame(data = d)
df = df.set_index('id')

mask_ids = []
mask_path = []
for dirname, _, filenames in os.walk(target_dir):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        mask_path.append(path)

        mask_id = filename.split(".")[0]
        mask_id = mask_id.split("_mask")[0]
        mask_ids.append(mask_id)
d = {"id": mask_ids,"mask_path": mask_path}
mask_df = pd.DataFrame(data = d)
mask_df = mask_df.set_index('id')
df["mask_path"] = mask_df["mask_path"]
img_size = [256,256]

def preprocessing(car_path, mask_path):
    car_img = tf.io.read_file(car_path)
    car_img = tf.image.decode_jpeg(car_img, channels=3)
    car_img = tf.image.resize(car_img, img_size)
    car_img = tf.cast(car_img, tf.float32) / 255.0

    mask_img = tf.io.read_file(mask_path)
    mask_img = tf.image.decode_jpeg(mask_img, channels=3)
    mask_img = tf.image.resize(mask_img, img_size)
    mask_img = mask_img[:,:,:1]
    mask_img = tf.math.sign(mask_img)
    return car_img, mask_img

def create_dataset(df):
        ds = tf.data.Dataset.from_tensor_slices((df["Image_Path"].values, df["mask_path"].values))
        ds = ds.map(preprocessing, tf.data.AUTOTUNE)
        return ds
train_df, valid_df = train_test_split(df, random_state=42, test_size=.3)
train = create_dataset(train_df)
valid = create_dataset(valid_df)



TRAIN_LENGTH = len(train)
BATCH_SIZE = 8
BUFFER_SIZE = 500
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
valid_dataset = valid.batch(BATCH_SIZE)

def conv_blockX(x, growth_rate, name, drop_rate=None, activation='elu'):
      x1 = tf.keras.layers.Conv2D(4 * growth_rate, 1, use_bias=False, activation=None, name=name + '_1_conv')(x)
      x1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, renorm=True,name=name + '_1_bn')(x1) #
      x1 = tf.keras.layers.Activation(activation, name=name + '_1_actv')(x1)
      x1 = tf.keras.layers.Conv2D(growth_rate, 3, padding='same', use_bias=False,activation=None, name=name + '_2_conv')(x1)
      x1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, renorm=True,name=name + '_2_bn')(x1)
      x1 = tf.keras.layers.Activation(activation, name=name + '_2_actv')(x1)
      if drop_rate is not None:
          x1 = tf.keras.layers.Dropout(rate=drop_rate, name=name + '_drop')(x1)
          x =tf.keras.layers.Concatenate(name=name + '_concat')([x, x1])
      return x

def transition_block(x, out_channels, name, activation='sigmoid'):
      x = tf.keras.layers.Conv2D(out_channels, 1, activation=None, use_bias=False, name=name + '_conv')(x)
      x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, renorm=True,name=name + '_bn')(x) #
      x = tf.keras.layers.Activation(activation, name=name + '_actv')(x)
      return x

def resblock(x,filters,stride=(1,1)):
    residual = x
    out = Conv2D(filters,kernel_size=(3,3),strides=stride,padding='same')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters,kernel_size=(3,3),strides=stride,padding='same')(x)
    out = BatchNormalization()(out)
    out = Add()([out,residual])
    out = Activation('relu')(out)
    return out

def Attention_B(X, G, k):
    FL = int(X.shape[-1])
    init = RandomNormal(stddev=0.02)
    theta = Conv2D(k,(2,2), strides = (2,2), padding='same')(X)
    Phi = Conv2D(k, (1,1), strides =(1,1), padding='same', use_bias=True)(G)
    ADD = Add()([theta, Phi])
    ADD = Activation('relu')(ADD)
    Psi = Conv2D(1,(1,1), strides = (1,1), padding="same",kernel_initializer=init)(ADD)
    Psi = Activation('sigmoid')(Psi)
    Up = Conv2DTranspose(1, (2,2), strides=(2, 2), padding='valid')(Psi)
    Final = Multiply()([X, Up])
    Final = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-5)(Final)
    print(Final.shape)
    return Final
def Up3(input1,input2,kernel=(3,3),stride=(1,1), pad='same'):
    up = Conv2DTranspose(int(input1.shape[-1]),(1, 1), strides=(2, 2), padding='same')(input2)
    up = Concatenate()([up,input1])
    return up

def dual_att_blocks(skip,prev,out_channels):
    up = Conv2DTranspose(out_channels,4, strides=(2, 2), padding='same')(prev)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)
    inp_layer = Concatenate()([skip,up])
    inp_layer = Conv2D(out_channels,3,strides=(1,1),padding='same')(inp_layer)
    inp_layer = BatchNormalization()(inp_layer)
    inp_layer = Activation('relu')(inp_layer)
    se_out = se_block(inp_layer,out_channels)
    sab = spatial_att_block(inp_layer,out_channels//4)
    #sab = Add()([sab,1])
    sab = Lambda(lambda y : y+1)(sab)
    final = Multiply()([sab,se_out])
    return final
def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    return Multiply()([in_block, x])
def spatial_att_block(x,intermediate_channels):
    out = Conv2D(intermediate_channels,kernel_size=(1,1),strides=(1,1),padding='same')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(1,kernel_size=(1,1),strides=(1,1),padding='same')(out)
    out = Activation('sigmoid')(out)
    return out
def encoding_block(inputs, filters):
    C = Conv2D(filters,3,padding="same",kernel_initializer="he_normal")(inputs)
    C = BatchNormalization()(C)
    C = Activation("relu")(C)

    C = Conv2D(filters,3,padding="same",kernel_initializer="he_normal")(C)
    C = BatchNormalization()(C)
    C = Activation("relu")(C)
    return C


def mixBlock(a1,b1,filter):
    x=Attention_B(a1,b1,filter) #n43=D6   16 8
    x = Up3(x,b1)
    y = dual_att_blocks(a1,b1,filter)
    z = Concatenate()([x,y])
    z = Conv2D(filter,kernel_size=(1,1),strides=(1,1),padding='same')(z)
    x = BatchNormalization()(z)
    x = Activation('relu')(x)
    x = Conv2D(filter,kernel_size=(3,3),strides=(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter,kernel_size=(3,3),strides=(1,1),padding='same')(x)
    x = Add()([x,z])
    pred4 = Conv2D(filter,kernel_size=(1,1),strides=(1,1),padding='same',activation="relu")(x)
    x = UpSampling2D(size=(2,2),interpolation='bilinear')(pred4) #32
    return x

def unet_model(input_size, filters, n_classes):
    inputs = Input(input_size)
    # Contracting Path (encoding)
    C1 = encoding_block(inputs,filters) #128 256
    D1=MaxPooling2D(pool_size=(2, 2))(C1)
    C2 = encoding_block(D1,filters*2) #64 128
    D2=MaxPooling2D(pool_size=(2, 2))(C2)
    C3 = encoding_block(D2,filters*4)#32 64
    D3=MaxPooling2D(pool_size=(2, 2))(C3)
    C4 = encoding_block(D3,filters*8 )#16 32
    D4=MaxPooling2D(pool_size=(2, 2))(C4)
    C5 = encoding_block(D4,filters*16) #16 32
    D4=MaxPooling2D(pool_size=(2, 2))(C5)
    C6 = encoding_block(D4,filters*16)#8 16
    D6=MaxPooling2D(pool_size=(2, 2))(C6)
    C7 = encoding_block(C6,filters*64)#8 16
    D7=MaxPooling2D(pool_size=(2, 2))(C7)
    D7=resblock(D7,filters*64)
    x=layers.Dropout(rate=0.5)(D7)
    #########################
    x=mixBlock(C7,D7,256)
    x=mixBlock(x,C6,256)
    x=mixBlock(x,C5,64)
    x=layers.Dropout(rate=0.9)(x)
    x=mixBlock(x,C4,32)
    x=mixBlock(x,C3,16)
    x=layers.Dropout(rate=0.5)(x)
    x = Conv2D(1,(1,1), strides=(1,1), padding="same",activation='sigmoid',name='x')(x)
    model = Model(inputs=inputs, outputs=x)
    return model

img_height = 256
img_width = 256
num_channels = 3
model = unet_model((img_height, img_width, num_channels), filters=16, n_classes=1)



# !pip install segmentation-models
from segmentation_models.metrics import IOUScore,FScore
from segmentation_models.losses import JaccardLoss,DiceLoss
import tensorflow as tf

import keras.backend as K


auc=tf.keras.metrics.AUC(from_logits=True)
miou=tf.keras.metrics.MeanIoU(num_classes=2)
precision=tf.keras.metrics.Precision(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
)
recall=tf.keras.metrics.Recall(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
)

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_loss(in_gt, in_pred):
    return 1-dice_coef(in_gt, in_pred)



from tensorflow.keras.optimizers import Adam,SGD
import time
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau,TensorBoard
weight_path="C:/Users/Ameer/Downloads/KvasirSEG/thesisAfter.best.hdf5".format('cxr_reg')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=5,verbose=1, mode='min', min_delta=0.0002, cooldown=4, min_lr=1e-6)
early = EarlyStopping(monitor="val_loss", mode="min",patience=22)
NAME = "kvasir-seg {}".format(int(time.time()))
tfBoard=TensorBoard(log_dir='./graphs/{}'.format(NAME), histogram_freq=0,write_graph=True, write_images=True)
callbacks_list = [checkpoint, early, reduceLROnPlat,tfBoard]
#model.summary()
from tensorflow.keras.utils import plot_model
#plot_model(model, )

newOptimizer=tf.keras.optimizers.Nadam(
    learning_rate=0.0015,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    name='Nadam'
)
model.compile(optimizer= newOptimizer, loss= DiceLoss(), metrics=[miou,dice_coef,precision,recall, 'accuracy'])


EPOCHS = 150
STEPS_PER_EPOCH = len(df) // BATCH_SIZE


model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=valid_dataset,
                          callbacks=callbacks_list,verbose=1)

import keras
from matplotlib import pyplot as plt

plt.plot(model_history.history['mean_io_u_1'])
plt.plot(model_history.history['val_mean_io_u_1'])
plt.title('Mean IoU Value')
plt.ylabel('Mean IoU')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()


plt.plot(model_history.history['dice_coef'])
plt.plot(model_history.history['val_dice_coef'])
plt.title('Dice Value')
plt.ylabel('Dice Coefficient')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()






import keras
from matplotlib import pyplot as plt

plt.plot(model_history.history['mean_io_u_3'])
plt.plot(model_history.history['val_mean_io_u_3'])
plt.title('Mean IoU Value')
plt.ylabel('Mean IoU')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

plt.plot(model_history.history['dice_coef'])
plt.plot(model_history.history['val_dice_coef'])
plt.title('Dice Value')
plt.ylabel('Dice Coefficient')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(model_history.history['precision_3'])
plt.plot(model_history.history['val_precision_3'])
plt.title('Model Precision')
plt.ylabel('Precision')
plt.xlabel('epoch')
plt.legend(['Train Precision', 'Val Precision'], loc='upper left')
plt.show()

plt.plot(model_history.history['recall_3'])
plt.plot(model_history.history['val_recall_3'])
plt.title('Model Recall')
plt.ylabel('Recall')
plt.xlabel('epoch')
plt.legend(['Train Recall', 'Val Recall'], loc='upper left')
plt.show()
