import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam,Adam,SGD
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import *
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
import numpy as np
from glob import glob

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


TRAIN_IMAGES = sorted(glob("/home/anish/anish/GCPAnet/DUT_BENCH/train/image/*"))
TRAIN_MASKS  = sorted(glob("/home/anish/anish/GCPAnet/DUT_BENCH/train/mask/*"))
VAL_IMAGES  = sorted(glob("/home/anish/anish/GCPAnet/DUT_BENCH/validation/image/*"))
VAL_MASKS   = sorted(glob("/home/anish/anish/GCPAnet/DUT_BENCH/validation/mask/*"))


H,W = 416,416

def read_images_mask(image,mask):
	im = tf.io.read_file(image)
	im = tf.io.decode_image(im,channels=3)
	ma = tf.io.read_file(mask)
	ma = tf.io.decode_image(ma,channels=1)
	#ma = tf.image.rgb_to_grayscale(ma)
	im = tf.cast(im,tf.float32)/255.
	ma = tf.cast(ma,tf.float32)/255.
	return im,ma



def random_rotate(image,mask):
    prob = np.random.randint(0,10,1)[0]
    if prob > 6:
        rotates = np.random.randint(0,15,1)[0]
        if rotates > 10:
            im = tf.image.rot90(image,k=1)
            ma = tf.image.rot90(mask,k=1)
        else:
            im = tf.image.rot90(image,k=2)
            ma = tf.image.rot90(mask,k=2)
    else:
        im = image
        ma = mask
        
    return im,ma      



def change_il(image):
    if np.random.randint(0,16,1)[0] > 13:
        im_proc = tf.image.random_contrast(image,lower=0.3,upper=0.6)
    if np.random.randint(0,16,1)[0] > 13:
        im_proc = tf.image.random_saturation(image,lower=0.3,upper=0.6)
    else:
        im_proc = image
    return im_proc 


def resize_pad(image,mask,H,W):
    m = 26
    n = 52
    o = 104
    p = 416
    im = tf.image.resize_with_pad(image,H,W)
    ma1 = tf.image.resize_with_pad(mask,m,m)
    ma2 = tf.image.resize_with_pad(mask,n,n)
    ma3 = tf.image.resize_with_pad(mask,o,o)
    ma4 = tf.image.resize_with_pad(mask,p,p)
    return im,(ma1,ma2,ma3,ma4)



def central_crop(image,mask):
    if tf.random.uniform((),minval=1,maxval=14) > 10:
        im = tf.image.central_crop(image,0.9)
        ma = tf.image.central_crop(mask,0.9)
    if tf.random.uniform((),minval=1,maxval=14) > 13:
        im = tf.image.central_crop(image,0.6)
        ma = tf.image.central_crop(mask,0.6)
    else:
        im = image
        ma = mask
    return im,ma
	
@tf.function
def traingen(ima,mas):
    i,m = read_images_mask(ima,mas)
    i   = change_il(i)
    i,m = random_rotate(i,m)
    #i,m = central_crop(i,m)
    i,(m1,m2,m3,m4) = resize_pad(i,m,H,W)
    
    return i,(m1,m2,m3,m4)

@tf.function
def valgen(ima,mas):
    i,m = read_images_mask(ima,mas)
    i,(m1,m2,m3,m4) = resize_pad(i,m,H,W)
    return i,(m1,m2,m3,m4)

BUFFER = len(TRAIN_IMAGES)
val_list = len(VAL_IMAGES)
BATCH_SIZE=3
TRAIN = tf.data.Dataset.from_tensor_slices((TRAIN_IMAGES,TRAIN_MASKS))
VAL = tf.data.Dataset.from_tensor_slices((VAL_IMAGES,VAL_MASKS))

TRAIN = TRAIN.map(traingen,num_parallel_calls=tf.data.experimental.AUTOTUNE)
VAL = VAL.map(valgen,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ = TRAIN.batch(BATCH_SIZE,drop_remainder=True).shuffle(10).repeat()
val_ = VAL.batch(BATCH_SIZE,drop_remainder=True).shuffle(10).repeat()


def self_refinement(inps,d):
    x1 = Conv2D(256//d,3,padding="same")(inps)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Activation("relu")(x1)
    x2 = Conv2D(512//d,3,padding="same")(x1)
    w,b = tf.split(x2,2,axis=-1)
    out = Multiply()([x1,w])
    out = Add()([out,b])
    out = Activation("relu")(out)
    return out


def head_attention(inps):
    x1 = Conv2D(256,3,padding="same")(inps)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Activation("relu")(x1)
    x2 = Conv2D(512,3,padding="same")(x1)
    w,b = tf.split(x2,2,axis=-1)
    out = Multiply()([x1,w])
    out = Add()([out,b])
    F   = Activation("relu")(out)
    ave = tf.math.reduce_mean(F,axis=-1)
    ave = Dense(256,activation="relu")(ave)
    Fn  = Dense(256,activation="sigmoid")(ave)
    out = Multiply()([F,Fn])
    return out


def GCF(inps):
    x1 = Conv2D(256,3,padding="same")(inps)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Activation("relu")(x1)
    
    gap = tf.math.reduce_mean(x1,axis=-1)
    gap = Dense(256,activation="relu")(gap)
    gap = Dense(256,activation="sigmoid")(gap)
    out = Multiply()([gap,x1])
    return out


def FIA(hin,gin,tin,ft):
    fl = Conv2D(ft,1,padding="same")(tin)
    fl = BatchNormalization(axis=-1)(fl)
    fl = Activation("relu")(fl)
    
    fh = Conv2D(ft,3,padding="same")(hin)# left block
    fh = Activation("relu")(fh)
    fh = UpSampling2D(interpolation="bilinear")(fh)
    fhl  = Multiply()([fl,fh]) #--->
    
    fdh = UpSampling2D()(hin)  #central block
    flh = Conv2D(ft,3,padding="same")(fl)
    flh = Activation("relu")(flh)
    flh = Multiply()([fdh,flh]) #--->
    
    fg = Conv2D(ft,3,padding="same")(gin)
    fg = Activation("relu")(fg)
    fg = UpSampling2D(interpolation="bilinear")(fg)
    fgl = Multiply()([fg,fl])
    
    conc = Concatenate()([fhl,flh,fgl])
    conc = Conv2D(ft,3,padding="same")(conc)
    conc = BatchNormalization(axis=-1)(conc)
    conc = Activation("relu")(conc)

    return conc


def GCPA():
    ENCODER_BASE = tf.keras.applications.ResNet50(include_top=False,input_shape=(416,416,3))
    for layer in ENCODER_BASE.layers:
        layer.traiable = True
    f0 = ENCODER_BASE.get_layer("activation").output
    
    f1 = ENCODER_BASE.get_layer("activation_9").output 
    f2 = ENCODER_BASE.get_layer("activation_21").output 
    f3 = ENCODER_BASE.get_layer("activation_36").output 
    f4 = ENCODER_BASE.get_layer("activation_48").output 
    
    aux1 = head_attention(f4)
    aux1 = self_refinement(aux1,1) 
    gux1 = GCF(f4)
    aux1 = FIA(aux1,gux1,f3,256)
    aux1_out = Conv2D(1,3,padding="same")(aux1)
    aux1_out = Activation("sigmoid",name="aux1")(aux1_out)
    
    aux2 = self_refinement(aux1,1)
    gux2 = GCF(f4)
    gux2 = UpSampling2D(interpolation="bilinear")(gux2)
    aux2 = FIA(aux2,gux2,f2,256)
    aux2_out = Conv2D(1,3,padding="same")(aux2)
    aux2_out = Activation("sigmoid",name="aux2")(aux2_out)
    
    aux3 = self_refinement(aux2,1)
    gux3 = GCF(f4)
    gux3 = UpSampling2D(interpolation="bilinear")(gux3)
    gux3 = UpSampling2D(interpolation="bilinear")(gux3)
    aux3 = FIA(aux3,gux3,f1,256)
    aux3_out = Conv2D(1,3,padding="same")(aux3)
    aux3_out = Activation("sigmoid",name="aux3")(aux3_out)

    dom0 = self_refinement(aux3,1)
    dom0 = UpSampling2D(interpolation="bilinear")(dom0)
    dom0 = Conv2D(64,3,padding="same")(dom0)
    dom0 = BatchNormalization(axis=-1)(dom0)
    dom0 = Activation("relu")(dom0)
    dom0 = UpSampling2D(interpolation="bilinear")(dom0)

    out  = Conv2D(1,3,padding="same")(dom0)
    out  = Activation("sigmoid",name="dom")(out)
    
    MODEL  = Model(ENCODER_BASE.input,[aux1_out,aux2_out,aux3_out,out])
    return MODEL
    
    
    
    
def schedule(epoch):
    cos_inner = np.pi * (epoch % (epochs // snapshots))
    cos_inner /= epochs // snapshots
    cos_out = np.cos(cos_inner) + 1
    lr = float(1e-5 / 2 * cos_out)
    tf.summary.scalar('learning rate', data=lr, step=epoch)
    return lr




snapshots = 5
epochs = 50
logdir = "GCPAkeras_tb"
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()


def schedule(epoch):
    cos_inner = np.pi * (epoch % (epochs // snapshots))
    cos_inner /= epochs // snapshots
    cos_out = np.cos(cos_inner) + 1
    lr = float(1e-5 / 2 * cos_out)
    tf.summary.scalar('learning rate', data=lr, step=epoch)
    return lr

losses  = {"aux1": "binary_crossentropy","aux2": "binary_crossentropy",
           "aux3": "binary_crossentropy","dom" : "binary_crossentropy"
          }
weights = {"aux1": 1.0, "aux2": 1.0,"aux3": 1.0,"dom": 1.0}
sched = LearningRateScheduler(schedule)

MOD = GCPA()




MOD.compile(optimizer="Nadam",loss=losses,loss_weights=weights,metrics=["acc"])

call_list = [TensorBoard(logdir),ModelCheckpoint("DUCTbenchc1.h5",monitor="val_loss",save_weights_only=True,verbose=1,mode="min"),
             EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=9),sched]
MOD.fit(train_,steps_per_epoch=BUFFER//BATCH_SIZE,epochs=epochs,callbacks=call_list,validation_data=val_,validation_steps=val_list//BATCH_SIZE)
