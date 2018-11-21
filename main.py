# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:13:31 2018

@author: wangz
"""
import data_loader # preprocess 
import models # model
import numpy as np
import pandas as pd
from mlxtend.regressor import StackingRegressor,StackingCVRegressor
from sklearn.metrics import mean_squared_error as mse

train_filename = 'train.csv'
test_filename = 'test.csv'



log = not False # log to map y into real space
m = 16 # guess maximum n_job 

#load data
dataset = data_loader.data_loader(max=m)
dataset.load_data(train_filename,test_filename)

#preprocessing step          
steps=['encode','add','regu']
for k in steps:
    dataset.preprocess(type=k)       
if log:
    dataset.y = dataset.y.apply(lambda x: np.log(x+1))

# get data    
x,y,test,test_index=dataset.get_data()

# set model parameters
# a little bit complicated because it is modified from my another asgn     
# parameters in model params 
#i f not exist, then grid search 
model=['mlp' for i in range(10)] # ten mlps
model_stack = models.model_factory()
for k in model:
    model_stack.add_model(k)
model_stack.set_parameters(x,y)
    
# model fusion part, use stacking

mods = model_stack.get_models()
sclf  = StackingRegressor(regressors=mods,use_features_in_secondary =True,meta_regressor=mods[0],verbose=0)
sclf.fit(x,y)
result = sclf.predict(test)

# map back the prediction
if not log:
    result= [ -x if x < 0 else x for x in result]
    result= list(result)
else:
    result = [np.exp(result[j])-1  for j in range(len(result))]
    
    
# cal the mse with the temp best result
# decide which to submit 

output = result

baseline = pd.read_csv('eval/19.csv')
baseline_label=list(baseline['Time'].values)
num = mse(np.array(output),baseline_label)
print(num)

#saving
index = list(range(0,100))
output = {'id':index,'Time':output}
output = pd.DataFrame(output)
output = pd.DataFrame(output,columns=['id', 'Time'])
output.to_csv(str(num)+'.csv',index=False)