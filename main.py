# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:13:31 2018

@author: wangz
"""
import warnings
warnings.filterwarnings('ignore')
import data_loader # preprocess 
import models # model
import numpy as np
import pandas as pd
train_filename = 'train.csv'
test_filename = 'test.csv'
from tqdm import tqdm
model=['mlp','svm','mlp']
log = False
m = 8
dele= False
dd = True
split = True
mini = 100
info = []
st = True


a = 0
for i in range(1):
    # data loader and pre-processer
    dataset = data_loader.data_loader(max=m,dele=dele)
    dataset.load_data(train_filename,test_filename)
        
    steps=['encode','add','regu']
    for i in steps:
        dataset.preprocess(type=i)
            
    if log:
        dataset.y.apply(lambda x: np.log(x+1))
    
    x,y,test,test_index=dataset.get_data()
        
    model_stack = models.model_factory()
    for i in model:
        model_stack.add_model(i)
    model_stack.set_parameters(x,y)
    
    model_trainer = models.model_trainer(fold_number=5)
    
    #model_trainer.validation(x,y,model_stack.models[0][1],dataset.index,dataset.sc)
    if st:
        from mlxtend.regressor import StackingRegressor,StackingCVRegressor
        mods = model_stack.get_models()
        sclf  = StackingRegressor(regressors=mods,use_features_in_secondary =split,
                                  meta_regressor=mods[0],verbose=0)
        
        
        sclf.fit(x,y)
        result = sclf.predict(test)
    else:
        

    
    
    
        model_trainer.train_model(x,y,model_stack.models[0][1])
        result = model_trainer.model.predict(test)
        
        result = list(result)
    
    tt = result
    if not log:
        result = [ -x if x < 0 else x for x in result]
        #
        if dele:
            result = [result[i]/test_index[i] for i in range(len(result))]
        #result = dataset.sc.inverse_transform(np.array(result).reshape(100,1)).reshape(100)
        
        
        result = list(result)
    else:
        result = [np.exp(result[i])-1 for i in range(len(result))]
    
    index = list(range(0,len(result)))
    output = {'id':index,'Time':result}
    output = pd.DataFrame(output)
    output = pd.DataFrame(output,columns=['id', 'Time'])
    
    from sklearn.metrics import mean_squared_error as mse
    baseline = pd.read_csv('x-1.85.csv')
    baseline_label=list(baseline['Time'].values)
    #print(mse(np.array(result),baseline_label))
    ore = mse(np.array(result),baseline_label)
    
    baseline = pd.read_csv('X.csv')
    baseline_label=list(baseline['Time'].values)
    #print(mse(np.array(result),baseline_label))
    ore1 = mse(np.array(result),baseline_label)
    
    print(ore,ore1)    
    output.to_csv(str(ore)+'.csv',index=False)