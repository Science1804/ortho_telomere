import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import pre_processF as ps
from sklearn.svm import SVC, LinearSVC;
from sklearn.linear_model import LogisticRegression;
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from DNN import DNN
from AED2 import AED2

class machine_learning:
    def __init__(self,DATA):
        'Data is the dictionary containing - Trainig , Validation and Test ';
        
        self.data = DATA ;
        self.models = {};
        self.models['SVC'] = SVC(); self.models['LinearSVC'] = LinearSVC();
        self.models['LogisticRegression'] = LogisticRegression();       
        self.models['LinearRegression'] = LinearRegression()
        
    def classifiers(self,name='SVC',get_plot=False,rdict=False,rmodel=True):
        self.name = name
        model = self.models[self.name];
        
        model.fit(self.data['train']['X'],self.data['train']['Y'])
        train = model.predict(self.data['train']['X'])
        predictions = model.predict(self.data['test']['X'])
        info1 = ps.get_metrics(self.data['test']['Y'],predictions)
        info2 = ps.get_metrics(self.data['train']['Y'],train)
        info = {};
        info['Train'] = info2 ;info['Test'] = info1 ;
        
        if get_plot :
            ps.Plot_1(info)
        
        if rdict:
            return info 
        
        if rmodel:
            return model
        
     
        
        
class deep_learning(DNN):
    def __init__(self,DATA,neurons,act='relu',cact='sigmoid',c=1,kreg=None):
        'Data is the dictionary containing - Trainig , Validation and Test ';
        
        self.data = DATA ;
        self.inp_shape = DATA['train']['X'].shape[1];
        
        super().__init__(neurons,act=act,cact=cact,c=c,input_shape=(self.inp_shape),kreg=kreg)
         
        
    def make_classifier(self,name='Classifier',epochs=100,verbose=0,vs=0.25,lr=0.0001,get_trC=False,get_tsP=False,rdict=False,rmodel=True):
        model = self.NN(name)
        # model.summary()
        opt1 = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='binary_crossentropy',optimizer=opt1,metrics=['Accuracy','AUC','Precision','Recall'])
        hist = model.fit(self.data['train']['X'],self.data['train']['Y'],validation_split=vs,epochs=epochs,verbose=verbose)
        
        if get_trC:
            ps.Plot_2(hist);

        Ts = model.evaluate(self.data['test']['X'],self.data['test']['Y'],verbose=verbose)
        info1 = ps.get_info(Ts)
        
        Tr = model.evaluate(self.data['train']['X'],self.data['train']['Y'],verbose=verbose)
        info2 = ps.get_info(Tr)
        info = {}
        info['Train'] = info2; info['Test'] = info1; 
        
        if get_tsP:
            ps.Plot_1(info)
            
        if rdict and rmodel:
            return hist , info , model 
            
        if rdict:
            return info
        if rmodel:
            return model
        
        
        
        
class AUTOENCODER(AED2):
    def __init__(self,DATA,neurons_in_layers2,lower_dim,acv='relu',eacv='relu',reg='l2',ddrps=False,**kwargs):
        self.DATA = DATA;
        if 'drps' in  kwargs.keys():
            self.drps = kwargs['drps']
        if 'initializer' in kwargs.keys():
            self.initializer = kwargs['initializer']
        else :
            self.initializer = 'glorot_uniform'
        
        input_shape = self.DATA['train']['X'].shape[1]
        nl1 = [input_shape];
        
        if len(neurons_in_layers2) > 0 :
            nl1.extend(neurons_in_layers2)
            
        super().__init__(nl1,lower_dim,input_shape,acv,eacv,reg,ddrps,*kwargs)
        
    
    def make_AE(self,epochs=100,vs=0.33,Loss='mse',optimizer='adam',lr=0.01,metrics=['Accuracy','Precision','Recall'],svloc_wname='models/29',get_trC=True,rmodel=True):
        
        self.encoder = self.Encoder();
        self.decoder = self.Decoder(self.encoder.layers[-2].output_shape[1:]);
        self.encoder.summary();
        self.decoder.summary();
        self.autoencoder = tf.keras.Model(inputs=self.encoder.inputs,outputs=self.decoder(self.encoder(self.encoder.inputs)),name='AutoEncoder_{}'.format(random.randint(0,100)))
        if  optimizer=='adam':
            self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr) ;
        else:
            self.optimizer1 = optimizer;
        
        
        self.autoencoder.compile(loss='mse', optimizer=self.optimizer1,metrics= metrics)
        
        hist = self.autoencoder.fit(self.DATA['train']['X'],self.DATA['train']['X'],epochs=epochs,validation_split=vs,shuffle=True)
        
        evals = self.autoencoder.fit(self.DATA['test']['X'],self.DATA['test']['X'])
        
        if get_trC:
            ps.Plot_2(hist);
            
        if rmodel:
            self.encoder.save(svloc_wname+'_enc.hdf5');
            self.decoder.save(svloc_wname+'_dec.hdf5');
            self.autoencoder.save(svloc_wname+'_auto.hdf5');
            # return self.encoder , self.decoder , self.autoencoder
        
        return hist