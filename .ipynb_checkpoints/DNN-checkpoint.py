import tensorflow as tf
class DNN:
    
    def __init__(self,neurons,act='relu',cact='sigmoid',c=1,input_shape=(3000),kreg=None):
    
        self.neurons = neurons ;
        self.act = act ;
        self.input_shape = input_shape ;
        self.cact = cact
        self.c = c ;
        
        if kreg==None:
            self.kreg = None;
        else:
            self.kreg = kreg

        return


    def NN(self,name='Classifier'):
        inp_layer = tf.keras.layers.Input(shape=self.input_shape,name='Input')
        x = inp_layer;

        for neuron in self.neurons:
            x = tf.keras.layers.Dense(neuron,activation=self.act,kernel_regularizer=self.kreg)(x)

        x = tf.keras.layers.Dense(self.c,activation=self.cact)(x)
        classifier = tf.keras.Model(inputs=inp_layer,outputs=x,name=name)

        return classifier