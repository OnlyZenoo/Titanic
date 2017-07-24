#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:50:36 2017

@author: Just
"""

#coidng:utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

ntrain=train.shape[0] ## 891
ntest=test.shape[0]   ## 418
kf=KFold(n_splits=5,random_state=2017)

def get_oof(clf,x_train,y_train,x_test):
    oof_train=np.zeros((ntrain,))  ##shape为(ntrain,)表示只有一维 891*1
    oof_test=np.zeros((ntest,))    ## 418*1
    oof_test_skf=np.empty((5,ntest))  ## 5*418
    for i,(train_index,test_index) in enumerate(kf.split(x_train)):
        kf_x_train=x_train[train_index] ## (891/5 *4)*7 故shape：(712*7)
        kf_y_train=y_train[train_index] ## 712*1
        kf_x_test=x_train[test_index]   ## 179*7

        clf.train(kf_x_train,kf_y_train)

        oof_train[test_index]=clf.predict(kf_x_test) #819*1 新train 和 y_train 训练新模型
        oof_test_skf[i,:]=clf.predict(x_test)

    oof_test[:]=oof_test_skf.mean(axis=0) # 用新模型预测418 test
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

get_oof()