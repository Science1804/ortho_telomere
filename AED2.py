import tensorflow as tf
import numpy as np
#import pandas as pd

class AED2:

    def __init__(self,neurons_in_layers,lower_dim,input_shape=(2830),acv='relu',eacv='relu',reg='l2',ddrps=False,**kwargs):
        if 'drps' in  kwargs.keys():
            self.drps = kwargs['drps']
        if 'initializer' in kwargs.keys():
            self.initializer = kwargs['initializer'] #tf.keras.initializers.LecunNormal()
        else :
            self.initializer = 'glorot_uniform'
        
            
        self.neurons_in_layers = neurons_in_layers 
        self.input_shape = input_shape
        self.lower_dim = lower_dim
        self.acv = acv
        self.reg = reg
        self.eacv = eacv
        return

    def Encoder(self):
        "layers_filter should be a list: [32,64,64] "
        inputs = tf.keras.layers.Input(shape=self.input_shape,name='Input')
        x = inputs
        for neurons in self.neurons_in_layers:
            x = tf.keras.layers.Dense(units=neurons,activation=self.acv,kernel_regularizer=self.reg,kernel_initializer=self.initializer)(x)
            try:
                x = tf.keras.layers.Droput(self.drps[i])
            except:
                pass


        latent_space = tf.keras.layers.Dense(self.lower_dim,name='lower_dim_vector',activation=self.eacv)(x)

        encoder = tf.keras.Model(inputs=inputs,outputs=latent_space, name='Encoder')

        return encoder


    def Decoder(self,enc_shape):
        "layers_filter should be a list: [32,64,64] given as same to encoder "
        self.neurons_in_layers.reverse();
        try :
            self.drps.reverse();
        except:
            pass 
    
        lower_inputs = tf.keras.layers.Input(shape=self.lower_dim,name='Decoder_Input')
        x = tf.keras.layers.Dense(enc_shape[0],activation=self.acv,kernel_regularizer=self.reg,kernel_initializer=self.initializer)(lower_inputs)
        try :
            x = tf.keras.layers.Dropout(self.drps[0])(x)
        except:
            pass
    
        for i,neurons in enumerate(self.neurons_in_layers[1:]):
            x = tf.keras.layers.Dense(units=neurons,activation=self.acv,kernel_regularizer=self.reg,kernel_initializer=self.initializer)(x)
            try:
                x = tf.keras.layers.Dropout(self.drps[i+1])(x)
            except:
                pass

        outputs = x

        decoder = tf.keras.Model(inputs=lower_inputs,outputs=outputs, name='Decoder')

        return decoder   

        
    def Classifier1(self,encoder=None,layers=None,ln=1,cactv=None,actabv=None,etrainable=None):
        if actabv == None:
            actabv='relu'
        
        if cactv == None:
            cactv = 'sigmoid'
            
        if encoder == None:
            encoder = self.Encoder()
        
        if etrainable == False:
            for layer in encoder.layers:
                layer.Trainable = False ;
        
        
        if layers != None:
            x = tf.keras.layers.Dense(layers[0],activation=actabv,kernel_regularizer='l2')(encoder.get_layer('lower_dim_vector').output);
            for each in layers[1:]:
                x = tf.keras.layers.Dense(each,activation=actabv,kernel_regularizer='l2')(x);
                x = tf.keras.layers.Dropout(0.1)(x);
            x = tf.keras.layers.Dense(ln,activation=cactv)(x) ;
        else :
            x = tf.keras.layers.Dense(ln,activation=cactv)(encoder.get_layer('lower_dim_vector').output);
            
        classifier1 = tf.keras.Model(inputs=encoder.inputs,outputs=x,name='classifier2');
        
        return classifier1
