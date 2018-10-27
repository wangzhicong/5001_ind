# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:14:03 2018

@author: wangz
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,OneHotEncoder,scale
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# class for loading data
class data_loader:
    def __init__(self):
        self.x = None
        self.y = None
        self.x_test = None
        
    # load data from file
    def load_data(self,train,test):
        self.x = pd.read_csv(train)
        self.y = self.x.loc[:,'time'].apply(lambda x:np.log(x)).values
        del self.x['time']
        del self.x['id']
        self.x_test = pd.read_csv(test)
        del self.x_test['id']
        
    
    # pre-process data with different method
    def preprocess(self,type=0):
        x_len = self.x.shape[0]
        x = self.x.append(self.x_test,ignore_index = True)
        
        x = pre_wrapper().get(type,x)
        self.x = x[:x_len]
        self.x_test = x[x_len:]
        
    
    #balance the dataset with drop, but no improvement
    def balance(self):
        for i in range(self.y['time'].count()):
            import random
            if self.y['time'].loc[i] == 1:
                rand_num = random.random()
                if rand_num < 0.2:
                    
                    self.x.drop(i,inplace=True)
                    self.y.drop(i,inplace=True)
    
    def select(self,feature_ratio = 0.9):
        feature_num = self.x.shape[1]
        selector = SelectKBest(chi2, k=int(feature_num*feature_ratio))
        self.x = selector.fit_transform(self.x,self.y)
        self.x_test = selector.transform(self.x_test)
    #return data for model
    def get_data(self):
        y = self.y
        x = self.x
        x_test = self.x_test
        if type(self.x) == pd.core.frame.DataFrame:
            x = x.values
        if type(self.x_test) == pd.core.frame.DataFrame:
            x_test = x_test.values
        return x,y,x_test

#age	workclass	fnlwgt	education	education-num	Marital-status	occupation	relationship	race	sex	capital-gain	capital-loss	hours-per-week	native-country

#two different kinds of attributes
encoder_cate = ['penalty']
encoder_num = ['l1_ratio','alpha','max_iter','random_state','n_jobs','n_samples','n_features','n_classes','n_clusters_per_class','n_informative','flip_y','scale']




import os




class pre_wrapper:
    def get(self,type,X):
        if type == 'onehot':
            return onehot(X)
        elif type == 'encode':
            return baseline(X)
        elif type == 'pred':
            return prediction(X)
        elif type == 'regu':
            return regular(X)
        else:
            print('No pre-processer, do nothing')
            return X


            

#encode the attribute into one hot, will not used
def onehot(x):
    output = 0
    x = x.replace(' ?',np.nan)
    for i in x.columns:
        x = x.fillna(str(-1))     
    for feat in encoder_cate:
        a = LabelEncoder().fit_transform(x[feat])
        a = OneHotEncoder( sparse=False ).fit_transform(a.reshape(-1,1))
        
        try:
            output = np.hstack((output,a))
        except:
            output = a
            
    for feat in encoder_num:
        #x[feat] = x[[feat]].apply(scale)
        a = x[feat].values.reshape(-1,1)
        output = np.hstack((output,a))
    return output

#no preprocess, just encode categorical into nums, will not use
def baseline(x):
    x = x.replace(' ?',np.nan)
    for i in x.columns:
        x = x.fillna(str(-1))  
    for feat in encoder_cate:
        x[feat] = LabelEncoder().fit_transform(x[feat])
    
    
    return x
        
def regular(x):
    for feat in encoder_num:
       x[feat] = x[[feat]].apply(scale)  
      
    return x

#use models to predict the missing values 
#pca, not use      
def preprocesser_3(x):
    x = x.replace(' ?',np.nan)
    for i in x.columns:
        x = x.fillna(str(-1))  
    for feat in encoder_cate:
        x[feat] = LabelEncoder().fit_transform(x[feat])
    from sklearn.decomposition import PCA
    model = PCA(n_components = 'mle',svd_solver = 'full')
    x = model.fit_transform(x.values)
    return x

# let the gain and loss into one attribute, which will be present in positive and negetave values
# not use
def preprocesser_4(x):
    x = x.replace(' ?',np.nan)
    for i in x.columns:
        x = x.fillna(str(-1))
    for feat in encoder_cate:
        x[feat] = LabelEncoder().fit_transform(x[feat])
    del x['education']
    del x['fnlwgt']
    for i in range(len(x['capital-gain'])):
        x['capital-gain'][i] = x['capital-gain'][i] - x['capital-loss'][i]
    del x['capital-loss']
    
    for feat in ['capital-gain']:
        x[feat] = x[[feat]].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    return x 

