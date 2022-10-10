import tensorflow as tf
" ready to import classif linear Variational Autoencoder"
" why linear ? - cause its not desined to take matrix based image input rather just an array of shape [ samples x features ]"


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class VAE(tf.keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
                name="reconstruction_loss"
            )
            self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.keras.losses.mse(data, reconstruction)
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss + kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

class VAED2:

    class Sampling(tf.keras.layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    
    def __init__(self,neurons_in_layers,lower_dim,input_shape=(2830),acv='relu',reg='l2'):
            
        self.neurons_in_layers = neurons_in_layers 
        self.input_shape = input_shape
        self.lower_dim = lower_dim
        self.acv = acv
        self.reg = reg
        return

    def Encoder(self):
        "layers_filter should be a list: [32,64,64] "
        inputs = tf.keras.layers.Input(shape=self.input_shape,name='Input')
        x = inputs
        for neurons in self.neurons_in_layers:
            x = tf.keras.layers.Dense(units=neurons,activation=self.acv,kernel_regularizer=self.reg)(x)
        z_mean = tf.keras.layers.Dense(self.lower_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(self.lower_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

        return encoder


    def Decoder(self,enc_shape):
        "layers_filter should be a list: [32,64,64] given as same to encoder "
        self.neurons_in_layers.reverse();
        lower_inputs = tf.keras.layers.Input(shape=self.lower_dim,name='Decoder_Input')
        x = tf.keras.layers.Dense(enc_shape[0],activation=self.acv,kernel_regularizer=self.reg)(lower_inputs)
        for i,neurons in enumerate(self.neurons_in_layers[:]):
            x = tf.keras.layers.Dense(units=neurons,activation=self.acv,kernel_regularizer=self.reg)(x)
            
        outputs = x

        decoder = tf.keras.Model(inputs=lower_inputs,outputs=outputs, name='Decoder')

        return decoder   
       
