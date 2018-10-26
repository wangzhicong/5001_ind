# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:13:31 2018

@author: wangz
"""

import data_loader # preprocess 
import models # model
import time
import pandas as pd
train_filename = 'train.csv'
test_filename = 'test.csv'




# data loader and pre-processer
dataset = data_loader.data_loader()
dataset.load_data(train_filename,test_filename)
steps=['encode']
for i in steps:
    dataset.preprocess(type=i)
x,y,test=dataset.get_data()

#model_stack = models.model_factory()
#model_stack.add_model('lr')
#model = model_stack.models[0][1]

model_trainer = models.model_trainer()
model = models.nn()
model_trainer.validation(x,y,model)
model_trainer.train_model(x,y,model)

'''
result = model_trainer.model.predict(test)
result = list(result)
result = [0 if x<0 else x for x in result]
index = list(range(0,len(result)))
output = {'id':index,'Time':result}
output = pd.DataFrame(output)
output = pd.DataFrame(output,columns=['id', 'Time'])
output.to_csv('rotk_never_give_up.csv',index=False)
'''