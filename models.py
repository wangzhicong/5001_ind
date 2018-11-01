# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:23:52 2018

@author: wangz
"""

from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVR as svm
from sklearn.svm import LinearSVR as lsvm
import numpy as np

# object to do train and validation
class model_trainer:
    def __init__(self,fold_number = 5):
        self.model = None
        self.fold_number=fold_number
    
    # k fold validation and show the avg accuracy
    def validation(self,x_train,y_train,model,index,sc):       
        kf=KFold(self.fold_number, shuffle=True, random_state=0)
        loss = 0       
        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            #print("Fold",i+1)
            eclf = model.fit(np.array(x_train)[train_index], np.array(y_train)[train_index])
            pred = eclf.predict(np.array(x_train)[test_index])
            #loss += mse(np.true_divide(pred,np.array(index[test_index])),np.true_divide(np.array(y_train)[test_index],np.array(index[test_index])))
            loss += mse(sc.inverse_transform(pred.reshape(int(400/self.fold_number),1)),sc.inverse_transform(np.array(y_train)[test_index].reshape(400//self.fold_number,1)))
            #print(pred)
        print('mse: ', loss/self.fold_number)
    
    # train model for prediction
    def train_model(self,x_train,y_train,model):
        self.model = model
        self.model.fit(x_train,y_train)        
        


from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import RidgeCV as rc
from sklearn.linear_model import Lasso as la
from sklearn.neural_network import MLPRegressor as mlp

from sklearn.model_selection import GridSearchCV
import time
import os
from xgboost.sklearn import XGBClassifier as xgb

# object to select models


class model_factory: 
    def __init__(self):
        self.models=[]
        self.param_grid = {}
       
    # add model and grid search params   
    def add_model(self,model_type):
        if model_type == 'lr':
            self.models.append((model_type,lr(normalize= True)))
        elif model_type == 'ridge':
            self.models.append((model_type,rc(normalize= True,cv=None)))
        elif model_type == 'lasso':
            self.models.append((model_type,la(normalize= True)))
        elif model_type == 'svm':
            self.models.append((model_type,svm()))
            self.param_grid['svm']={
                    'kernel':['linear','poly','rbf','sigmoid'],
                    'C':range(10,100,10),
                    'epsilon':[0.01]
                    
                    }
        elif model_type == 'mlp':
            self.models.append((model_type,mlp()))
            self.param_grid['mlp']={
                    'hidden_layer_sizes':[(32,32)],
                    'activation':['identity', 'logistic', 'tanh', 'relu'],
                    'solver':['lbfgs','adam'],
                    'alpha':[0.001,0.01],
                    'learning_rate':['constant', 'invscaling', 'adaptive'],
                    'learning_rate_init':[0.001,0.01,0.1],
                    #'early_stopping':[True,False],
                    #'validation_fraction':[0.1,0.05,0.2],
                    #'max_iter':[200,1000,2000]
                    }
        elif model_type == 'xgb':
            self.models.append((model_type,xgb()))
            self.param_grid[model_type]={
                    'max_depth':range(3,10,2),
                    'min_child_weight':range(1,6,2),
                    'n_estimators':range(100,1101,200),
                    'learning_rate':[0.01,0.05,0.1],
                    'n_jobs': [4],
                    'reg_alpha': [0,0.005,0.01],
                    'subsample':[0.8,1],
                    'colsample_bytree':[0.8,1]
                    }


    #set the params for different models after grid search
    def create_model(self,model_type,parameters):

        if model_type == 'lr':
            model = lr()
        elif model_type == 'svm':
            model = svm()
        elif model_type == 'mlp':
            model = mlp()
        
        return  model.set_params(**parameters)
    
    # grid search, if param file exist then directly set param    
    def set_parameters(self,x,y):
        model = self.models.copy()
        self.models = []
        for name,model in model:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            #print('%s set parameters for model %s '% (time_str,name))
            if os.path.exists('model_params/' + name+'.txt'):
                #print('parameter already exists and loading %s model parameters' % name)
                f = open('model_params/'+name+'.txt','r')
                a = f.read()
                param = eval(a)
                f.close()
                self.models.append((name,self.create_model(name,param)))
                continue
            print('mannually find best parameters for model %s' % name)
            try:
                local_param_grid = self.param_grid[name]
                grid_search = GridSearchCV(model, param_grid=local_param_grid,verbose=2)
                start = time.time()
                grid_search.fit(x,y) 
                print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
                      % (time.time() - start, len(grid_search.cv_results_['params'])))
                f = open('model_params/'+name+'.txt','w')
                f.write(str(grid_search.best_estimator_.get_params()))
                f.close()
                self.models.append((name,self.create_model(name,grid_search.best_estimator_.get_params())))
            except:
                self.models.append((name,model))
            
            
    def get_models(self):
        output = []
        for i in self.models:
            #print(i)
            output.append(i[1])
            
        return output