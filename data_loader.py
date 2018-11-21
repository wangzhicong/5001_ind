# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:14:03 2018

@author: wangz
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,OneHotEncoder
from sklearn.preprocessing import scale,MinMaxScaler,minmax_scale,normalize
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import PolynomialFeatures as pl
# class for loading data

#del_cat =['random_state','flip_y','scale','n_features','n_classes']
class data_loader:
    def __init__(self,max=-1):
        self.x = None
        self.y = None
        self.x_test = None
        self.index = None
        self.max = max
        self.test_index =None
        self.sc = StandardScaler()
        
    # load data from file
    def load_data(self,train,test):
        self.x = pd.read_csv(train)
        self.y = self.x.loc[:,'time']
        self.y = self.y
        self.x_test = pd.read_csv(test)
         
        del self.x['time']
        del self.x['id']
        del self.x_test['id']
        #add feature from self generated data
        y0 = pd.read_csv('model_params/train_data0.csv')
        self.x['additional'] = y0['time']
        y1 = pd.read_csv('model_params/test_data0.csv')
        self.x_test['additional'] = y1['time']
        
        # fill the n_jobs
        self.test_index = self.x_test['n_jobs'].apply(lambda x: x if x<=self.max and x >0 else self.max).values
        self.x['n_jobs'] = self.x['n_jobs'].apply(lambda x: x if x<=self.max and x >0 else self.max)

    def preprocess(self,type=0):
        x_len = self.x.shape[0]
        if type != 'aug':
            x = self.x.append(self.x_test,ignore_index = True)
            x,self.y = pre_wrapper().get(type,x,self.y)
            try:
                del x['time']
            except:
                pass
            self.x = x[:x_len]
            self.x_test = x[x_len:]
                
        else:
            self.x,self.y = pre_wrapper().get(type,self.x,self.y)
        
    #return data for model
    def get_data(self):
        y = self.y.values
        x = self.x.copy()
        x_test = self.x_test.copy()
        if type(self.x) == pd.core.frame.DataFrame:
            x = x.values
            
        if type(self.x_test) == pd.core.frame.DataFrame:
            x_test = x_test.values
        
        return x,y,x_test,self.test_index


encoder_cate = ['penalty']
encoder_onehot = ['n_jobs']
encoder_num = ['l1_ratio','alpha','max_iter','n_samples','n_features','n_classes','n_clusters_per_class','n_informative']
double_cat = ['n_samples']

times = ['max_iter','n_samples','n_classes','0']
divides = ['l1_ratio','alpha','max_iter','random_state','n_samples']


# different pre-processing method      
class pre_wrapper:
    def get(self,type,X,y):
        if type == 'encode':
            return baseline(X),y
        elif type == 'regu':
            return regular(X),y
        elif type == 'pca':
            return pca(X),y
        elif type == 'add':
            return add(X),y
        elif type == 'poly':
            return poly(X),y
        else:
            print('No pre-processer, do nothing')
            return X,y

def regular(x):
    for i in x.columns:
        x[i] = x[[i]].apply(scale)  
 
    return x


# use poly to generate new features
def poly(x):
    t = x.values
    t = pl(degree=3,interaction_only=True).fit_transform(t)
    
    return t

# add features
def add(x):
    
    #x['feature_1'] = x['max_iter'] * x['n_samples'] * x['n_features'] * x['n_classes'] / x['n_jobs']
    del x['random_state']
    del x['flip_y']
    del x['scale']
    #del x['penalty']
    #del x['n_samples']
    #del x['alpha']
    #del x['l1_ratio']
    #del x['n_informative']
    #del x['penalty']
    #x['-1'] = x['n_features'] - x['n_informative']
    #x['3'] = x['n_samples'] * x['0']
    #x['4'] = x['max_iter'] * x['0']
    #x['5'] = x['n_classes'] * x['0']
    #x['6'] = x['n_samples'] * x['n_features']
    #x['7'] =  x['n_samples'] /x['n_classes'] 
    #x['8'] = x['n_classes'] * x['max_iter']
    x['9'] = x['n_classes'] * x['n_clusters_per_class']
    #del x['n_classes']
    #del x['n_clusters_per_class']
    #x['1'] = x['n_samples'] * x['max_iter'] / x['n_jobs']
    #x['0'] = x['max_iter'] / x['n_jobs']
    return x
    
# encode             
def baseline(x):
    '''
    shape = len(x)
    x['0'] = x['n_informative']
    for i in range(shape):
        if x['penalty'].loc[i] in code:
            x['0'].loc[i] = x['n_informative'].loc[i]  
        else:
            x['0'].loc[i] = x['n_features'].loc[i]
    '''
    for feat in encoder_cate:
        x[feat] = LabelEncoder().fit_transform(x[feat])

    return x
       

#pca, not use      
def pca(x):
    from sklearn.decomposition import PCA
    model = PCA(n_components = 1,svd_solver = 'auto')
    x = model.fit_transform(x.values)
    x = pd.DataFrame(x)
    return x


