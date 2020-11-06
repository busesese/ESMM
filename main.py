# -*- coding: utf-8 -*-
# @Time    : 2020-11-05 17:49
# @Author  : WenYi
# @Contact : wenyi@cvte.com
# @Description :  script description

import numpy as np
import pandas as pd
from esmm import CTCVRNet
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model_train import train_model

# data include ctr data and cvr data, ctr data include ctr user data and ctr item data,
# user data include numerical data and categorical data
# item data include numerical data and categorical data
# we generate sample data include user feature data and item feature data
# user feature data include 5 numerical data and 5 categorical data
# item feature data include 5 numerical data and 5 categorical data
ctr_user_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['user_numerical_{}'.format(i) for i in range(5)])
ctr_user_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),
                                           columns=['user_cate_{}'.format(i) for i in range(5)])
ctr_item_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['item_numerical_{}'.format(i) for i in range(5)])
ctr_item_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                           columns=['item_cate_{}'.format(i) for i in range(3)])

cvr_user_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['user_numerical_{}'.format(i) for i in range(5)])
cvr_user_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),
                                           columns=['user_cate_{}'.format(i) for i in range(5)])
cvr_item_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_item_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                           columns=['item_cate_{}'.format(i) for i in range(3)])


ctr_user_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['user_numerical_{}'.format(i) for i in range(5)])
ctr_user_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),
                                         columns=['user_cate_{}'.format(i) for i in range(5)])
ctr_item_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['item_numerical_{}'.format(i) for i in range(5)])
ctr_item_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)), columns=['item_cate_{}'.format(i) for i in range(3)])

cvr_user_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['user_numerical_{}'.format(i) for i in range(5)])
cvr_user_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),
                                         columns=['user_cate_{}'.format(i) for i in range(5)])
cvr_item_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_item_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                         columns=['item_cate_{}'.format(i) for i in range(3)])

ctr_target_train = pd.DataFrame(np.random.randint(0, 2, size=10000))
cvr_target_train = pd.DataFrame(np.random.randint(0, 2, size=10000))

ctr_target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))
cvr_target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))

train_data = [ctr_user_numerical_feature_train, ctr_user_cate_feature_train, ctr_item_numerical_feature_train,
              ctr_item_cate_feature_train, cvr_user_numerical_feature_train, cvr_user_cate_feature_train,
              cvr_item_numerical_feature_train, cvr_item_cate_feature_train, ctr_target_train, cvr_target_train]
val_data = [ctr_user_numerical_feature_val, ctr_user_cate_feature_val, ctr_item_numerical_feature_val,
            ctr_item_cate_feature_val, cvr_user_numerical_feature_val, cvr_user_cate_feature_val,
            cvr_item_numerical_feature_val, cvr_item_cate_feature_val, ctr_target_val, cvr_target_val]


cate_feature_dict = {}
user_cate_feature_dict = {}
item_cate_feature_dict = {}
for idx, col in enumerate(ctr_user_cate_feature_train.columns):
    cate_feature_dict[col] = ctr_user_cate_feature_train[col].max() + 1
    user_cate_feature_dict[col] = (idx, ctr_user_cate_feature_train[col].max() + 1)
for idx, col in enumerate(ctr_item_cate_feature_train.columns):
    cate_feature_dict[col] = ctr_item_cate_feature_train[col].max() + 1
    item_cate_feature_dict[col] = (idx, ctr_item_cate_feature_train[col].max() + 1)

ctcvr = CTCVRNet(cate_feature_dict)
ctcvr_model = ctcvr.build(user_cate_feature_dict, item_cate_feature_dict)
opt = optimizers.Adam(lr=0.003, decay=0.0001)
ctcvr_model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"], loss_weights=[1.0, 1.0],
                    metrics=[tf.keras.metrics.AUC()])

# keras model save path
filepath = "esmm_best.h5"

# call back function
checkpoint = ModelCheckpoint(
    filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1)
earlystopping = EarlyStopping(
    monitor='val_loss', min_delta=0.0001, patience=8, verbose=1, mode='auto')
callbacks = [checkpoint, reduce_lr, earlystopping]

# trian model
train_model(cate_feature_dict, user_cate_feature_dict, item_cate_feature_dict, train_data, val_data)
