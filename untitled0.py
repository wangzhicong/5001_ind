# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:58:49 2018

@author: wangz
"""

from auto_ml import Predictor
#from auto_ml.utils import get_boston_dataset
import pandas as pd
df_train = pd.read_csv('train.csv')


df_test = pd.read_csv('test.csv')
del df_train['id']
del df_test['id']

cat = df_test.columns

cat1 = cat[0:6]
cat2 = cat[6:14]

d1 = df_train[cat1].copy().apply(lambda x: 2*x)