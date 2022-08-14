import tensorflow as tf
class MCC:
    
    def __init__(self,filters,act='relu',cact='sigmoid',c=1,input_shape=(256,256,3),strides=(2,2),kernel=(3,3),kreg=None):
    
        self.filters = filters ;
        self.act = act ;
        self.input_shape = input_shape ;
        self.strides = strides
        self.kernel = kernel ;
        self.cact = cact
        self.c = c ;
        
        if kreg==None:
            self.kreg = None;
        else:
            self.kreg = kreg

        return


    def Classifier(self):
        inp_layer = tf.keras.layers.Input(shape=self.input_shape,name='encoder input')
        x = inp_layer;

        for filter in self.filters:
            x = tf.keras.layers.Conv2D(filter,kernel_size=self.kernel,strides=self.strides,activation=self.act,kernel_regularizer=self.kreg)(x)

        latent = tf.keras.layers.Flatten(name='latent_flatten')(x)
        latent = tf.keras.layers.Dense(self.c,activation=self.cact)(latent)
        classifier = tf.keras.Model(inputs=inp_layer,outputs=latent,name='Classifier')

        return classifier


    def Classifier_plus(self,neurons):
        self.neurons = neurons; 
        inp_layer = tf.keras.layers.Input(shape=self.input_shape,name='encoder input')
        x = inp_layer;

        for filter in self.filters:
            x = tf.keras.layers.Conv2D(filter,kernel_size=self.kernel,strides=self.strides,activation=self.act,kernel_regularizer=self.kreg)(x)

        latent = tf.keras.layers.Flatten(name='latent_flatten')(x)
        x2 = latent ;
        
        for neuron in self.neurons:
            x2 = tf.keras.layers.Dense(neuron,activation=self.act,kernel_regularizer=self.kreg)(x2)
        
        x2 = tf.keras.layers.Dense(self.c,activation=self.cact)(x2)
        
        classifier = tf.keras.Model(inputs=inp_layer,outputs=x2,name='Classifier2')

        return classifier