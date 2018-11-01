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
import random
# class for loading data

#del_cat =['random_state','flip_y','scale','n_features','n_classes']
class data_loader:
    def __init__(self,max=-1,dele = False):
        self.x = None
        self.y = None
        self.x_test = None
        self.index = None
        self.max = max
        self.test_index =None
        self.dele = dele
        #self.sc = MinMaxScaler(feature_range=(0,50))
        self.sc = StandardScaler()
        
    # load data from file
    def load_data(self,train,test):
        self.x = pd.read_csv(train)
        #self.y = self.x.loc[:,'time'].values
        self.y = self.x.loc[:,'time']#.apply(lambda x: x*x)
        self.y = self.y
         
        #y2 = self.sc.transform(np.array(y))
        #print(y)
        del self.x['time']
        del self.x['id']
        '''
        x1 = self.x.copy()
        x1[double_cat] = x1[double_cat]#.apply(lambda x:x*(1-random.uniform(-0.05,0.05)) )
        self.x = self.x.append(x1,ignore_index = True)
        y0 = pd.read_csv('model_params/train_data0.csv')
        y0 = y0.loc[:,'time']
        self.y = self.y.append(y0,ignore_index = True)
        '''
        
        
        
        self.x_test = pd.read_csv(test)
        del self.x_test['id']
        self.test_index = self.x_test['n_jobs'].apply(lambda x: x if x<=self.max and x >0 else self.max).values
        
        
        
        
        self.x = self.x.append(self.x_test,ignore_index = True)
        self.x['n_jobs'] = self.x['n_jobs'].apply(lambda x: x if x<=self.max and x >0 else self.max)
        y1 = pd.read_csv('model_params/test_data0.csv')
        y1 = y1.loc[:,'time']#.apply(lambda x:*x)
        self.y = self.y.append(y1,ignore_index = True)#.apply(lambda x: np.exp(x))
        
            
        self.index = self.x['n_jobs'].values
        if self.max > 0:
            for i in range(len(self.y)):
              
                if self.dele:
                    self.y[i] *= self.index[i]
            
    
            
        #y = self.y
        #y = self.sc.fit_transform(y.reshape(500,1)).reshape(1,500).tolist()
        #self.y = y
        
        
    # pre-process data with different method
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
        self.x = selector.fit_transform(self.x.values,list(self.y.values))
        self.x_test = selector.transform(self.x_test.values)
    #return data for model
    def get_data(self):
        y = self.y.values
        
       #y = np.append(y,y1,axis =0)
        x = self.x.copy()
        x_test = self.x_test.copy()
        if type(self.x) == pd.core.frame.DataFrame:
            if self.dele:
                del x['n_jobs']
            x = x.values
            
        if type(self.x_test) == pd.core.frame.DataFrame:
            if self.dele:
                #pass
                del x_test['n_jobs']
                #index = list(index)
            x_test = x_test.values
        return x,y,x_test,self.test_index

#age	workclass	fnlwgt	education	education-num	Marital-status	occupation	relationship	race	sex	capital-gain	capital-loss	hours-per-week	native-country

#two different kinds of attributes
encoder_cate = ['penalty']
encoder_onehot = ['n_jobs']
encoder_num = ['l1_ratio','alpha','max_iter','random_state','n_samples','n_features','n_classes','n_clusters_per_class','n_informative','flip_y','scale']
double_cat = ['n_samples']

times = ['n_samples','n_features','n_classes','n_clusters_per_class','n_informative']
divides = []


from sklearn.ensemble import RandomForestClassifier as rf
from xgboost.sklearn import XGBClassifier as xgb
import os




class pre_wrapper:
    def get(self,type,X,y):
        if type == 'onehot':
            return onehot(X),y
        elif type == 'encode':
            return baseline(X),y
        elif type == 'pred':
            return prediction(X),y
        elif type == 'regu':
            return regular(X),y
        elif type == 'pca':
            return pca(X),y
        elif type == 'aug':
            return aug(X,y)
        elif type == 'add':
            return add(X),y
        
        else:
            print('No pre-processer, do nothing')
            return X,y
code= ['l1','elasticnet']
def add(x):
    #x['eff'] = x['penalty']
    #for i in range(len(x['penalty'])):
    #    x['eff'].loc[i] = x['n_features'].loc[i] if x['penalty'].loc[i] not in code else x['n_informative'].loc[i]
    count = 2
    '''
    for i in x.columns:
        x[str(count)] = x[i]*x[i]
        count+=1
        #x[str(count)] = x[i]
        #count+=1
        #x[str(count)] = x[i]*x[i]*x[i]
        #count+=1
    '''
    for i in times:
        for j in times:
            x[str(count)] = x[i]*x[j]
            count +=1
            for k in times:
                x[str(count)] = x[i]*x[j]*x[k]
                count +=1
    
                #for m in times:
                #    x[str(count)] = x[i]*x[j]*x[k]*x[m]
                #    count +=1
    #= x ['n_informative'] / x['n_features'] 
    #x['dum'] = x['penalty'] * x['eff']
    #x['1'] = x['n_features']*x['n_samples']
    #x['1'] = x['n_samples'] * x['max_iter'] / x['n_jobs']
    x['0'] = x['max_iter'] / x['n_jobs']
    #for i in times:
    #    x[str(count)] = x[i] / x['n_jobs']
    #    count +=1
    #x['3'] = x['n_features'] *x['n_informative']
    
    return x

def aug(x,y):
    for i in range(2,3):
        rnd = (1+random.random())
        tmp = x.copy()
        tmp[double_cat] = tmp[double_cat].apply(lambda x : rnd*x)
        x = pd.concat((x,tmp))
        y1 = y.copy()
        y1 = y1.apply(lambda x : rnd*x)
        y = pd.concat((y,y1))
    return x,y

    
                

#encode the attribute into one hot, will not used
def onehot(x):
    output = 0
    x = x.replace(' ?',np.nan)
    for i in x.columns:
        x = x.fillna(str(-1))     
    for feat in encoder_onehot:
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
    
    for feat in encoder_cate:
        #x[feat] = LabelEncoder().fit_transform(x[feat])
        #x[feat] = x[feat].apply(lambda x:'el' if x in code else x  )
        x[feat] = LabelEncoder().fit_transform(x[feat])
        
    
    
    
    return x
        
def regular(x):
    
    x[x.columns] = x[x.columns].apply(scale)  
      
    return x

#use models to predict the missing values
def preprocesser_2x(x):
    x = x.replace(' ?',str(-1))
    #
    encoders = {}
    for feat in encoder_cate:
        encoders[feat] = LabelEncoder()
        x[feat] = encoders[feat].fit_transform(x[feat])
        if '-1' in list(encoders[feat].classes_):
            x[feat] = x[feat].replace(encoders[feat].transform([str(-1)])[0],np.nan)  
    
    x = x.dropna()
    return x,encoders

def prediction(x):
    save_name = 'preprocessed_data/predicting.csv'
    if os.path.exists(save_name):
        print('loading %s data' % save_name)
        x = pd.read_csv(save_name)
        del x['Unnamed: 0']
        return x
    x_drop,encoders = preprocesser_2x(x)
    x = x.replace(' ?',str(-1))
    for feat in encoder_cate:
        x[feat] = encoders[feat].transform(x[feat])
    
    for feat in encoder_cate:
        if '-1' not in list(encoders[feat].classes_):
            continue
        print('dealing with %s' % feat)     
        train = x_drop.copy()
        f = open('model_params/rf.txt','r')
        a = f.read()
        param = eval(a)
        f.close()
        model = rf()
        model.set_params(**param)
        y_feat = train[feat].values
        del train[feat]
        x_feat = train.values
        model.fit(x_feat,y_feat)
        test = x.copy()
        del test[feat]
        for i in range(len(x[feat])):
            if x[feat].loc[i] == encoders[feat].transform([str(-1)])[0]:
                x[feat].loc[i] = model.predict(test.loc[i].values.reshape(1,-1))
            else:
                continue
    
    x.to_csv(save_name) 
        
    return x


 
#pca, not use      
def pca(x):
    from sklearn.decomposition import PCA
    model = PCA(n_components = 150,svd_solver = 'full')
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

