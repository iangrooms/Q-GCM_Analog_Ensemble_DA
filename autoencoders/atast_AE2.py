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

# this one just calculates the gradient loss for sst.

class gAE(keras.Model):
    def __init__(self, arch_params, lweights, **kwargs):
        super(gAE, self).__init__(**kwargs)
        kr, br, ar, at_dimx, at_dimy, beta1 = arch_params
        avepool_size=2
        at_last_hidden_dimx = np.int(at_dimx/(8*avepool_size)); at_last_hidden_dimy = np.int(at_dimy/(8*avepool_size)); at_channels=1

        # Atmosphere encoder for ast
        atc1=1
        encoder_at_inputs_1 = keras.Input(shape=(at_dimx, at_dimy, atc1), name="ast_input");   
        astE = layers.Conv2D(8*atc1, (9,9), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ast_conv1a")(encoder_at_inputs_1); 
        astE = layers.Conv2D(16*atc1, (5,5), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ast_conv1b")(astE);   
        astE = layers.MaxPooling2D((2,2), padding='same',name="ast_pool1")(astE);                                        
        astE = layers.Conv2D(32*atc1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ast_conv2a")(astE);   
        astE = layers.MaxPooling2D((2,2), padding='same',name="ast_pool2")(astE);                                        
        astE = layers.Conv2D(32*atc1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ast_conv3a")(astE);                                  
        astE = layers.AveragePooling2D(pool_size=(avepool_size, avepool_size), padding="same", name="ast_avepool1")(astE);  
        astE = layers.Conv2D(32*atc1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ast_conv4a")(astE); 
        astE = layers.Flatten(name="ast_flatten")(astE);   
        astE = keras.Model(inputs=encoder_at_inputs_1,outputs=astE)

        c1 = astE.output
        c1 = layers.Dense(beta1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar)(c1)
        
        encoder = keras.Model(inputs= astE.input, outputs = c1)
        print(encoder.summary())
        
        self.encoder = keras.Model(inputs= astE.input, outputs = c1)

        latent_inputs = keras.Input(shape=((beta1),), name="00_latent")
        c1=split()(latent_inputs,beta1)
        astD = c1

        astD = layers.Dense(astE.output.shape[1], kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar)(astD)


        # Atmosphere ast Decoder
        astD = layers.Reshape((at_last_hidden_dimx, at_last_hidden_dimy, 32*atc1),name="ast_reshape")(astD);   
        astD = layers.UpSampling2D(size=(avepool_size, avepool_size),name="ast_upsample")(astD);   
        astD = layers.Conv2DTranspose(32*atc1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="ast_transpose1a")(astD);   
        astD = layers.Conv2DTranspose(32*atc1, (3,3), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="ast_transpose2a")(astD);   
        astD = layers.Conv2DTranspose(32*atc1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="ast_transpose1b")(astD);   
        astD = layers.Conv2DTranspose(16*atc1, (3,3), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ast_transpose2b")(astD);   
        astD = layers.Conv2DTranspose(8*atc1, (5,5), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ast_transpose2c")(astD);   
        astD = layers.Conv2DTranspose(atc1, (9,9), strides=2, padding="same", name="ast_transpose1d" )(astD);   
        astD = keras.Model(inputs=latent_inputs, outputs=astD)
        
        decoder = keras.Model(inputs=latent_inputs, outputs=astD.output)
        print(decoder.summary())

        # Combine
        self.decoder = keras.Model(inputs=latent_inputs, outputs=astD.output)
        self.lweights = lweights
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        #self.grad_loss_tracker = keras.metrics.Mean(name="grad_loss")
        #self.dot_loss_tracker = keras.metrics.Mean(name="dot_loss")
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.mse_loss_tracker
            #self.grad_loss_tracker
           # self.dot_loss_tracker
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
            r_ast1 = self.decoder(z)

            diff = data-r_ast1
            diff2 = tf.square(diff)

            # Calculate losses.
            mse_loss = self.lweights[0]*tf.reduce_mean(diff2)
                        
            #grad_loss = 2*mse_loss + (self.lweights[0]*self.get_gmm(diff[0],diff2[0]))
 
            total_loss = mse_loss #+ (self.lweights[2]*grad_loss)
        
        # Calculate gradient.
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        # Clip
        grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        #grads = [(tf.clip_by_norm(grad, clip_norm=2.0)) for grad in grads]
        
        # Update weights.
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.mse_loss_tracker.update_state(mse_loss)
        #self.grad_loss_tracker.update_state(grad_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "mse_loss": self.mse_loss_tracker.result()
         #   "grad_loss": self.grad_loss_tracker.result()
        }
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        # Encode.
        z = self.encoder(data)
        
        # Decode.
        r_ast1 = self.decoder(z)

        # Calculate differences and squared differences.    
        diff = data - r_ast1
        diff2 = tf.square(diff)

        # Calculate losses.
        mse_loss = self.lweights[0] * tf.reduce_mean(diff2)
    
        #grad_loss = (2 * mse_loss) + (self.lweights[0] * self.get_gmm(diff[0],diff2[0]))
        
        total_loss = mse_loss #+ (self.lweights[2]*grad_loss) 
        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss
        #    "grad_loss": grad_loss
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
        -2*(
            tf.reduce_mean(tf.math.multiply(diff[:,1:,:,:],diff[:,:-1,:,:]))
            +tf.reduce_mean(tf.math.multiply(diff[:,:,1:,:],diff[:,:,:-1,:]))
            )
        )
        return gmm
