import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import random
import warnings

import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage.util.shape import view_as_windows

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
#from tensorflow.keras.layers.core import Dropout, Lambda
#from tensorflow.keras.layers.convolutional import Conv2D, Conv2DTranspose
#from tensorflow.keras.layers.pooling import MaxPooling2D
#from tensorflow.keras.layers.merge import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
#from keras.utils import plot_model
#K.tensorflow_backend._get_available_gpus()

import tensorflow as tf

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#######################################################################################################


# import file
# img = nib.load('/home/student2/Desktop/Working_Kunal/coupled_all_imgs_Guys.nii.gz')
img_input_1 = nib.load('/home/student2/Desktop/Working_Kunal/coupled_all_imgs_Guys_T1.nii.gz')
# img_input_2 = nib.load('/home/student2/Desktop/Working_Kunal/coupled_all_imgs_Guys_T2.nii.gz')
# img_input_3 = nib.load('/home/student2/Desktop/Working_Kunal/coupled_all_imgs_Guys_PD.nii.gz')
img_output = nib.load('/home/student2/Desktop/Working_Kunal/coupled_all_imgs_Guys_TOF.nii.gz')


## get the arrays
img_input_1 = img_input_1.get_fdata()
#img_input_2 = img_input_3.get_fdata()
#img_input_3 = img_input_3.get_fdata()
img_output = img_output.get_fdata()

print('shape of input data:', img_input_1.shape)
print('shape of output data:', img_output.shape)

############################ mask #############################################################
mask_ = (img_input_1 > 0)*1
img_output = img_output * mask_

## create and save masked output
#img_tmp_ = nib.Nifti1Image(img_output, np.eye(4))
#nib.save(img_tmp_,'/home/student2/Desktop/Working_Kunal/u-net-2/TOF_masked_T2.nii.gz')
################################################################################################

#img_input = np.stack([img_input_1, img_input_2], -1)
#print(img_input.shape)


#img_tmp_ = nib.Nifti1Image(img_input, np.eye(4))
#nib.save(img_tmp_,'/home/student2/Desktop/T1_T2.nii.gz')


## reshaping data
#img_input_1 = img_input_1.transpose([2,0,1])
#img_input_2 = img_input_2.transpose([2,0,1])
#img_input_3 = img_input_3.transpose([2,0,1])
img_output = img_output.transpose([2,0,1])

img_input = img_input_1.transpose([2,0,1])
print(img_input.shape)

#img_tmp_ = nib.Nifti1Image(img_input[0].squeeze(), np.eye(4))
#nib.save(img_tmp_,'/home/student2/Desktop/T1T2PD.nii.gz')

#print('shape of reshaped input data:', img_input_1.shape)
#print('shape of reshaped output data:', img_output.shape)

####################



##input reshape to add channel dim
img_input = np.expand_dims(img_input,-1)
img_output = np.expand_dims(img_output,-1)

print('shape of 4D input data:', img_input.shape)
print('shape of 4D output data:', img_output.shape)



## cropping of image for U-net
c_x = np.int(img_input.shape[1]/2)
c_y = np.int(img_input.shape[2]/2)
img_input = img_input[:,c_x-184:c_x+184,c_y-248:c_y+248,:]

c_x = np.int(img_output.shape[1]/2)
c_y = np.int(img_output.shape[2]/2)
img_output = img_output[:,c_x-184:c_x+184,c_y-248:c_y+248,:]

print('shape of input patch Image:', img_input.shape)
print('shape of output patch Image:', img_output.shape)



## cut out input and output for training and testing data
input_train = img_input[0:np.int(img_input.shape[0]*0.9),...]
output_train = img_output[0:np.int(img_output.shape[0]*0.9),...]

input_test = img_input[np.int(img_input.shape[0]*0.9):,...]
output_test = img_output[np.int(img_output.shape[0]*0.9):,...]

print('shape of input training data:', input_train.shape)
print('shape of output training data:', output_train.shape)
print('shape of input testing data:', input_test.shape)
print('shape of output testing data:', output_test.shape)



## initializing parameters for network
IMG_HEIGHT = input_train.shape[2]
IMG_WIDTH = input_train.shape[1]
IMG_CHANNELS = input_train.shape[3]

print('IMG_HEIGHT:', IMG_HEIGHT)
print('IMG_WIDTH:', IMG_WIDTH)
print('IMG_CHANNELS:', IMG_CHANNELS)


## customizing loss function
def custom_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred) * K.exp(2*K.abs(y_true)))


## Build U-Net model
inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
#s = Lambda(lambda x: x / 255) (inputs)

print('input layer:', inputs.shape)
c1 = Conv2D(16, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (inputs)
print('convolutional layer:', c1.shape)
c1 = Dropout(0.1) (c1)
print('dropout layer:', c1.shape)
c1 = Conv2D(16, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (c1)
print('convolutional layer:', c1.shape)
p1 = MaxPooling2D((2, 2)) (c1)
print('maxpool layer:', p1.shape)


c2 = Conv2D(32, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (5, 5), activation='elu', kernel_initializer='glorot_uniform', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
#model.compile(optimizer='adam', loss=custom_loss, metrics=['binary_accuracy'])
model.compile(optimizer='adam', loss=custom_loss, metrics=['binary_accuracy'])
model.summary()


# Fit model
#earlystopper = EarlyStopping(patience=5, verbose=1)
#checkpointer = ModelCheckpoint('TOF_prediction_using_T1.h5', verbose=1, save_best_only=True)
checkpointer = ModelCheckpoint('TOF_masked_T1_custom_loss_xe^2x_only_groundtruth.h5', verbose=1, save_best_only=True, period=10)
tensorBoard = TensorBoard(log_dir='/home/student2/TBLogs/TOF_masked_T1_custom_loss_xe^2x_only_groundtruth')
#history = model.fit(input_train, output_train, validation_split=0.1, batch_size=16, epochs=50, callbacks=[earlystopper, checkpointer])
history = model.fit(input_train, output_train, validation_split=0.1, batch_size=8, epochs=500, callbacks=[tensorBoard, checkpointer])


# Predict on train, val and test
model = load_model(('TOF_masked_T1_custom_loss_xe^2x_only_groundtruth.h5'), custom_objects={'custom_loss': custom_loss})
preds_train = model.predict(input_train[:int(input_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(input_train[int(input_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(input_test, verbose=1)



### save training input, output and predicted images
ix = random.randint(0, len(preds_train))
img_tmp_ = nib.Nifti1Image(np.squeeze(input_train[ix]), np.eye(4))
nib.save(img_tmp_,'/home/student2/Desktop/Working_Kunal/u-net-2/input_train_T1_custom_loss_xe^2x_only_groundtruth.nii.gz')
img_tmp_ = nib.Nifti1Image(np.squeeze(output_train[ix]), np.eye(4))
nib.save(img_tmp_,'/home/student2/Desktop/Working_Kunal/u-net-2/output_train_T1_custom_loss_xe^2x_only_groundtruth.nii.gz')
img_tmp_ = nib.Nifti1Image(np.squeeze(preds_train[ix]), np.eye(4))
nib.save(img_tmp_,'/home/student2/Desktop/Working_Kunal/u-net-2/preds_train_T1_custom_loss_xe^2x_only_groundtruth.nii.gz')



### save test input, output and predicted images
ix = random.randint(0, len(preds_test))
img_tmp_ = nib.Nifti1Image(np.squeeze(img_input[np.int(img_input.shape[0]*0.9)+ix]), np.eye(4))
nib.save(img_tmp_,'/home/student2/Desktop/Working_Kunal/u-net-2/input_test_T1_custom_loss_xe^2x_only_groundtruth.nii.gz')
img_tmp_ = nib.Nifti1Image(np.squeeze(img_output[np.int(img_output.shape[0]*0.9)+ix]), np.eye(4))
nib.save(img_tmp_,'/home/student2/Desktop/Working_Kunal/u-net-2/output_test_T1_custom_loss_xe^2x_only_groundtruth.nii.gz')
img_tmp_ = nib.Nifti1Image(np.squeeze(preds_test[ix]), np.eye(4))
nib.save(img_tmp_,'/home/student2/Desktop/Working_Kunal/u-net-2/preds_test_T1_custom_loss_xe^2x_only_groundtruth.nii.gz')

"""

metric = []

for i in range(0, len(preds_test)):
    input_tmp = img_input[np.int(img_input.shape[0]*0.9)+i]
    output_tmp = preds_test[i]
    sub_tmp = output_tmp - input_tmp
    metric = np.append(sub_tmp)

plt.figure(1)
plt.plot(metric)
plt.show()
"""


###########################################
############################################
###########################################

from numpy import inf
import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
"""
###################################################################
# image import IXI MRA
img = nib.load('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/IXI012-HH/IXI012-HH-1211-T2toMRA.nii.gz')

# image shape
img_shape = img.shape
print(img.shape)

# convert the image into data
img_data = img.get_fdata()

# plotting images
plt.figure(1)
plt.imshow(img_data[:,:,50])
plt.show()

# plotting patches
plt.figure(2)
plt.subplot(221)
plt.imshow(img_data[0:np.int(img_shape[0]/2),0:np.int(img_shape[1]/2),50])
plt.subplot(222)
plt.imshow(img_data[0:np.int(img_shape[0]/2),np.int(img_shape[1]/2):,50])
plt.subplot(223)
plt.imshow(img_data[np.int(img_shape[0]/2):,0:np.int(img_shape[1]/2),50])
plt.subplot(224)
plt.imshow(img_data[np.int(img_shape[0]/2):,np.int(img_shape[1]/2):,50])
plt.show()

# saving the patches in variables
p1 = img_data[0:np.int(img_shape[0]/2),0:np.int(img_shape[1]/2),:]
p2 = img_data[0:np.int(img_shape[0]/2),np.int(img_shape[1]/2):,:]
p3 = img_data[np.int(img_shape[0]/2):,0:np.int(img_shape[1]/2),:]
p4 = img_data[np.int(img_shape[0]/2):,np.int(img_shape[1]/2):,:]

# saving the image as nifti files
p1 = nib.Nifti1Image(p1, np.eye(4))
nib.save(p1, os.path.join('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/IXI012-HH/IXI012-HH-1211-T2toMRA-patch-1.nii.gz'))

p2 = nib.Nifti1Image(p2, np.eye(4))
nib.save(p2, os.path.join('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/IXI012-HH/IXI012-HH-1211-T2toMRA-patch-2.nii.gz'))

p3 = nib.Nifti1Image(p3, np.eye(4))
nib.save(p3, os.path.join('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/IXI012-HH/IXI012-HH-1211-T2toMRA-patch-3.nii.gz'))

p4 = nib.Nifti1Image(p4, np.eye(4))
nib.save(p4, os.path.join('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/IXI012-HH/IXI012-HH-1211-T2toMRA-patch-4.nii.gz'))
#########################################################
"""
np.seterr(divide='ignore', invalid='ignore')

# import MRA image
img_MRA = nib.load('/home/student2/Desktop/Working_Kunal/IXI-MRA/Guys/IXI002-Guys/IXI002-Guys-0828-MRA.nii.gz')

# import T1 image
img_T1 = nib.load('/home/student2/Desktop/Working_Kunal/IXI-MRA/Guys/IXI002-Guys/IXI002-Guys-0828-T1toMRA.nii.gz')

# import ref test image
img_reftest = nib.load('/home/student2/Desktop/Working_Kunal/IXI-MRA/Guys/IXI002-Guys/ref_test.nii.gz')

# convert MRA into data
img_MRA_data = np.double(img_MRA.get_fdata())

# convert T1 into data
img_T1_data = np.double(img_T1.get_fdata())

# convert ref test into data
img_reftest_data = np.double(img_reftest.get_fdata())

# subtract MRA by T1
div_MRA_T1 = (img_MRA_data-img_T1_data)

# subtract ref test by T1
div_reftest_T1 = img_reftest_data-img_T1_data

# remove Nan values
div_MRA_T1_nan = div_MRA_T1[~np.isnan(div_MRA_T1)]
div_MRA_T1_nan[div_MRA_T1_nan >= inf] = 0

div_reftest_T1_nan = div_reftest_T1[~np.isnan(div_reftest_T1)]
div_reftest_T1_nan[div_reftest_T1_nan >= inf] = 0

# print max and min value
print("max value in MRA-T1:", div_MRA_T1_nan.max(), "..........min value in MRA-T1:", div_MRA_T1_nan.min())
print("max value in reftest-T1:", div_reftest_T1_nan.max(), ".........min value in reftest-T1:",  div_reftest_T1_nan.min())

# view divided image
#plt.figure(1)
#plt.imshow(div_MRA_T1_nan[:,:,50])
#plt.show()

# mean
mean_MRA_T1 = np.mean(div_MRA_T1_nan)
print("difference bw MRA and T1:", mean_MRA_T1)

mean_reftest_T1 = np.mean(div_reftest_T1_nan)
print("difference bw reftest and T1:", mean_reftest_T1)

#########################################
#########################################
########################################
