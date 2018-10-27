# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:23:52 2018

@author: wangz
"""

from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier
import tensorflow as tf
import numpy as np

# object to do train and validation
class model_trainer:
    def __init__(self,fold_number = 5):
        self.model = None
        self.fold_number=fold_number
    
    # k fold validation and show the avg accuracy
    def validation(self,x_train,y_train,model):       
        self.model=model
        kf=KFold(self.fold_number, shuffle=True, random_state=0)
        loss = 0       
        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            print("Fold",i+1)
            pred = self.model.fit(np.array(x_train)[train_index], np.array(y_train)[train_index],np.array(x_train)[test_index],np.array(y_train)[test_index])
            loss += pred
        print('accuracy: ', loss/self.fold_number)
    
    # train model for prediction
    def train_model(self,x_train,y_train,model):
        self.model = model
        self.model.train(x_train,y_train)        
        


from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import GridSearchCV
import time
import os



class nn:
    def neural_net_model(self,X,input_dim):
        X = tf.reshape(X,[-1,input_dim])
        dense = tf.layers.dense(inputs=X,units=64, activation=tf.nn.relu)
        dense =  tf.layers.dense(inputs=dense,units=32, activation=tf.nn.relu)
        #dense =  tf.layers.dense(inputs=dense,units=32, activation=tf.nn.relu)
        output = tf.layers.dense(inputs=dense,units=1, activation=tf.nn.relu)
        
        return output
        
    def fit(self,x,y,x_test,y_test):
        xs = tf.placeholder("float")
        ys = tf.placeholder("float")
        output = self.neural_net_model(xs,13)
        cost = tf.reduce_mean(tf.square(output-ys))
        cost2 = tf.reduce_mean(tf.square(tf.exp(output)-tf.exp(ys)))
        # our mean squared error cost function
        trainer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
        with tf.Session() as sess:
            # Initiate session and initialize all vaiables
            sess.run(tf.global_variables_initializer())
            #saver.restore(sess,'yahoo_dataset.ckpt')
            for i in range(10):
                #x = np.reshpe([-1,13])
                sess.run([cost,trainer],feed_dict= {xs:x, ys:y})
            score = sess.run(cost2, feed_dict={xs:x_test,ys:y_test})
            print('Cost :',score)
            
            return score
        
    def train_prediction(self,x,y,x_test):
        xs = tf.placeholder("float")
        ys = tf.placeholder("float")
        output = self.neural_net_model(xs,13)
        cost = tf.reduce_mean(tf.square(output-ys))
        # our mean squared error cost function
        trainer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
        
        with tf.Session() as sess:
            
            # Initiate session and initialize all vaiables
            sess.run(tf.global_variables_initializer())
            #saver.restore(sess,'yahoo_dataset.ckpt')
            for i in range(100):
                sess.run([cost,trainer],feed_dict= {xs:x, ys:y})
                    # Run cost and train with each sample
            score = sess.run(output, feed_dict={xs:x_test}) 
            return score
        



# object to select models
class model_factory: 
    def __init__(self):
        self.model = None
        self.param_grid = {}
        
    # add model and grid search params   
    def add_model(self,model_type):
        if model_type == 'lr':
            self.models.append((model_type,lr(normalize= True)))

    #set the params for different models after grid search
    def create_model(self,model_type,parameters):

        if model_type == 'lr':
            model = lr()

        return  model.set_params(**parameters)
    
    # grid search, if param file exist then directly set param    
    def set_parameters(self,x,y):
        model = self.models.copy()
        self.models = []
        for name,model in model:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print('%s set parameters for model %s '% (time_str,name))
            if os.path.exists('model_params/' + name+'.txt'):
                print('parameter already exists and loading %s model parameters' % name)
                f = open('model_params/'+name+'.txt','r')
                a = f.read()
                param = eval(a)
                f.close()
                self.models.append((name,self.create_model(name,param)))
                continue
            print('mannually find best parameters for model %s' % name)
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
            
            
