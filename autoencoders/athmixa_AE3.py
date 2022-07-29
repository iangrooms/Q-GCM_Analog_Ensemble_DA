import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import concatenate
import numpy as np
from tensorflow.keras import regularizers
import sys
import string
import custom_layers
split = custom_layers.Splitc3     
relu_hmixa = custom_layers.Reluhmixa2

sys.path.append('/projects/luya7574/ml_qgcm/diagnosing_reconstructions/Scripts')
from define_grid import *

class gAE(keras.Model):
    def __init__(self, arch_params, lweights, **kwargs):
        super(gAE, self).__init__(**kwargs)
        kr, br, ar, at_dimx, at_dimy, beta1 = arch_params
        avepool_size=2
        at_last_hidden_dimx = np.int(at_dimx/(8*avepool_size)); at_last_hidden_dimy = np.int(at_dimy/(8*avepool_size)); at_channels=1

        # Atmosphere hmixa encoder
        atc1=1
        encoder_at_inputs_1 = keras.Input(shape=(at_dimx, at_dimy, atc1), name="hmixa_input");   
        hmixaE = layers.Conv2D(8*atc1, (9,9), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="hmixa_conv1a")(encoder_at_inputs_1); 
        hmixaE = layers.Conv2D(16*atc1, (5,5), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="hmixa_conv1b")(hmixaE);   
        hmixaE = layers.MaxPooling2D((2,2), padding='same',name="hmixa_pool1")(hmixaE);                                        
        hmixaE = layers.Conv2D(32*atc1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="hmixa_conv2a")(hmixaE);   
        hmixaE = layers.MaxPooling2D((2,2), padding='same',name="hmixa_pool2")(hmixaE);                                        
        hmixaE = layers.Conv2D(32*atc1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="hmixa_conv3a")(hmixaE);                                  
        hmixaE = layers.AveragePooling2D(pool_size=(avepool_size, avepool_size), padding="same", name="hmixa_avepool1")(hmixaE);  
        hmixaE = layers.Conv2D(32*atc1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="hmixa_conv4a")(hmixaE); 
        hmixaE = layers.Flatten(name="hmixa_flatten")(hmixaE);   
        hmixaE = keras.Model(inputs=encoder_at_inputs_1,outputs=hmixaE)
        
        c1 = hmixaE.output
        c1 = layers.Dense(beta1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar)(c1)
        
        encoder = keras.Model(inputs= hmixaE.input, outputs = c1)
        print(encoder.summary())
        
        self.encoder = keras.Model(inputs= hmixaE.input, outputs = c1)
        
        latent_inputs = keras.Input(shape=((beta1),), name="00_latent")
        c1 = split()(latent_inputs,beta1)
        hmixaD = c1

        hmixaD = layers.Dense(hmixaE.output.shape[1], kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar)(hmixaD)

        # Atmosphere hmixa Decoder
        hmixaD = layers.Reshape((at_last_hidden_dimx, at_last_hidden_dimy, 32*atc1),name="hmixa_reshape")(hmixaD);   
        hmixaD = layers.UpSampling2D(size=(avepool_size, avepool_size),name="hmixa_upsample")(hmixaD);   
        hmixaD = layers.Conv2DTranspose(32*atc1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="hmixa_transpose1a")(hmixaD);   
        hmixaD = layers.Conv2DTranspose(32*atc1, (3,3), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="hmixa_transpose2a")(hmixaD);   
        hmixaD = layers.Conv2DTranspose(32*atc1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="hmixa_transpose1b")(hmixaD);   
        hmixaD = layers.Conv2DTranspose(16*atc1, (3,3), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="hmixa_transpose2b")(hmixaD);
        hmixaD = layers.Conv2DTranspose(16*atc1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="hmixa_transpose3b")(hmixaD);
        hmixaD = layers.Conv2DTranspose(8*atc1, (5,5), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="hmixa_transpose2c")(hmixaD);   
        hmixaD = layers.Conv2DTranspose(atc1, (9,9), strides=1, padding="same", name="hmixa_transpose1d" )(hmixaD);   
        hmixaD = keras.Model(inputs=latent_inputs, outputs=hmixaD)        
        
        decoder = keras.Model(inputs=latent_inputs, outputs = hmixaD.output)
        print(decoder.summary())

        # Combine
        self.decoder = keras.Model(inputs=latent_inputs, outputs = hmixaD.output)
        self.lweights = lweights
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        #self.grad_loss_tracker = keras.metrics.Mean(name="grad_loss")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.mse_loss_tracker
            #self.grad_loss_tracker
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        # elif isinstance(data, list):
        #     at, oc = data
        with tf.GradientTape() as tape:
            # Encode.
            z = self.encoder(data)
            
            # Decode.
            r_hmixa1 = self.decoder(z)

            diff = data-r_hmixa1
            diff2 = tf.square(diff)

            # Calculate losses.
            mse_loss = self.lweights[0]*tf.reduce_mean(diff2)
                        
            #grad_loss = 2*mse_loss +
             #   (self.lweights[0]*self.get_gmm(diff[0],diff2[0]))
 
            total_loss = mse_loss #+ (self.lweights[2]*grad_loss)
        
        # Calculate gradient.
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        # Clip
        grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        grads = [(tf.clip_by_norm(grad, clip_norm=2.0)) for grad in grads]
        
        # Update weights.
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.mse_loss_tracker.update_state(mse_loss)
        #self.grad_loss_tracker.update_state(grad_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "mse_loss": self.mse_loss_tracker.result()
           # "grad_loss": self.grad_loss_tracker.result()
        }
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        # Encode.
        z = self.encoder(data)
        
        # Decode.
        r_hmixa1 = self.decoder(z)

        # Calculate differences and squared differences.    
        diff = data-r_hmixa1
        diff2 = tf.square(diff)

        # Calculate losses.
        mse_loss = self.lweights[0]*tf.reduce_mean(diff2)
        
        #grad_loss = 2*mse_loss + (self.lweights[0]*self.get_gmm(diff[0],diff2[0]))
    
        total_loss = mse_loss #+ (self.lweights[2]*grad_loss) 
        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss
         #   "grad_loss": grad_loss
        } 

    # Save weights
    def save_weights(self, path, **kwargs):
        self.encoder.save_weights(path+'.e.h5')
        self.decoder.save_weights(path+'.d.h5')
    # Load weights
    def load_weights(self, path, **kwargs):
        self.encoder.load_weights(path+'.e.h5')#, by_name=True)
        self.decoder.load_weights(path+'.d.h5')#, by_name=True)

    # Countparams
    def count_params(self, **kwargs):
        return self.encoder.count_params()+self.decoder.count_params()
    
    # Predict
    def predict(self, data):
        Z = self.encoder(data)
        Rat1 = self.decoder(Z)
        return Rat1
    
    def d_predict64(self,Z):
        Rat1 = self.decoder(Z)
        return Rat1.numpy().astype('float64')

    def get_gmm(self, diff,diff2):
        gmm = (tf.reduce_mean(diff2[:,0::diff2.shape[1]-1,1:-1,:])
        +tf.reduce_mean(diff2[:,1:-1,0::diff2.shape[2]-1,:])
        +2*tf.reduce_mean(diff2[:,1:-1,1:-1,:])
        -2*(tf.reduce_mean(tf.math.multiply(diff[:,1:,:,:],diff[:,:-1,:,:]))
            +tf.reduce_mean(tf.math.multiply(diff[:,:,1:,:],diff[:,:,:-1,:]))
            )
        )
        return gmm 