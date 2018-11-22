from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
from datetime import datetime
import tqdm

def timing(data,name):
    times = {'time':[]}
    for idx,attr in tqdm.tqdm(data.iterrows()):
        param = attr.to_dict()
        sgd = {}
        gene = {}
        
        for i in param:
            if i in class_cates:
                gene[i] = param[i]
            elif i in sgd_cates:
                sgd[i] = param[i]
            
        start = datetime.now()
        x,y = make_classification(**gene)
        model = SGDClassifier(**sgd)
        model.fit(x,y)
        time = datetime.now()-start
        times['time'].append(time.total_seconds())
        
        save = pd.DataFrame(times)
        save.to_csv(name)
        
    return 0

class_cates = ['n_samples','n_features','n_classes','n_clusters_per_class','n_informative','flip_y','scale' ]
sgd_cates = ['penalty','l1_ratio','alpha','max_iter','random_state','n_jobs']

train_params= pd.read_csv('../train.csv')
test_params = pd.read_csv('../test.csv')

test_time = timing(test_params,'test_data.csv')
train_time = timing(train_params,'train_data.csv')


