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
        kr, br, ar, oc_dimx, oc_dimy, beta3 = arch_params
        avepool_size=2
        oc_last_hidden_dimx = np.int(oc_dimx/(4*avepool_size)); oc_last_hidden_dimy = np.int(oc_dimy/(4*avepool_size)); oc_channels=1

    
        # Ocean encoder
        occ1=1
        encoder_oc_inputs_1 = keras.Input(shape=(oc_dimx, oc_dimy, occ1), name="ocsst_input"); 
        Oesst = layers.Conv2D(8*occ1, (9,9), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ocsst_conv1a" )(encoder_oc_inputs_1);  
        Oesst = layers.Conv2D(16*occ1, (5,5), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ocsst_conv1b" )(Oesst);  
        Oesst = layers.MaxPooling2D((2,2), padding='same',name="ocsst_pool1" )(Oesst);                                 
        Oesst = layers.Conv2D(16*occ1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ocsst_conv2a" )(Oesst);                        
        Oesst = layers.Conv2D(32*occ1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ocsst_conv3a" )(Oesst); 
        Oesst = layers.MaxPooling2D((2,2), padding='same',name="ocsst_pool3" )(Oesst);                                 
        Oesst = layers.Conv2D(32*occ1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ocsst_conv4a" )(Oesst); 
        Oesst = layers.Conv2D(32*occ1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ocsst_conv4b" )(Oesst); 
        Oesst = layers.MaxPooling2D((2,2), padding='same',name="ocsst_pool4" )(Oesst);                                 
        Oesst = layers.Conv2D(32*occ1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ocsst_conv5a" )(Oesst); 
        Oesst = layers.AveragePooling2D(pool_size=(avepool_size, avepool_size), padding="same", name="ocsst_avepool1" )(Oesst);        
        Oesst = layers.Conv2D(32*occ1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ocsst_conv5b" )(Oesst); 
        Oesst = layers.Flatten(name="ocsst_flatten" )(Oesst); 
        Oesst = keras.Model(inputs=encoder_oc_inputs_1,outputs=Oesst)

        c3 = Oesst.output

        c3 = layers.Dense(beta3, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar)(c3)

        Z = c3
        
        encoder = keras.Model(inputs=Oesst.input, outputs = Z)
        print(encoder.summary())
        
        self.encoder = keras.Model(inputs=Oesst.input, outputs = Z)


        latent_inputs = keras.Input(shape=(beta3,), name="00_latent")
        c3=split()(latent_inputs,beta3)
        Odsst = c3

        Odsst = layers.Dense(Oesst.output.shape[1], kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar)(Odsst)


        # Ocean decoder
        Odsst = layers.Reshape((16, 16, 32*occ1),name="ocsst_reshape" )(Odsst); 
        Odsst = layers.UpSampling2D(size=(avepool_size, avepool_size),name="ocsst_upsample" )(Odsst); 
        Odsst = layers.Conv2DTranspose(32*occ1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="ocsst_transpose1a" )(Odsst); 
        Odsst = layers.Conv2DTranspose(32*occ1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="ocsst_transpose2a" )(Odsst); 
        Odsst = layers.Conv2DTranspose(32*occ1, (3,3), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="ocsst_transpose1b" )(Odsst); 
        Odsst = layers.Conv2DTranspose(32*occ1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="ocsst_transpose2b" )(Odsst); 
        Odsst = layers.Conv2DTranspose(32*occ1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="ocsst_transpose2bb" )(Odsst); 
        Odsst = layers.Conv2DTranspose(16*occ1, (3,3), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="ocsst_transpose2cc" )(Odsst); 
        Odsst = layers.Conv2DTranspose(16*occ1, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="ocsst_transpose2c" )(Odsst); 
        Odsst = layers.Conv2DTranspose(16*occ1, (3,3), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ocsst_transpose1d" )(Odsst); 
        Odsst = layers.Conv2DTranspose(8*occ1, (5,5), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="ocsst_transpose2d" )(Odsst); 
        Odsst = layers.Conv2DTranspose(occ1, (9,9), strides=1, padding="same", name="ocsst_transpose1e" )(Odsst); 
        Odsst = keras.Model(inputs=latent_inputs, outputs=Odsst)

        #define decoder
        decoder = keras.Model(inputs=latent_inputs, outputs=[Odsst.output])
        print(decoder.summary())
        
        self.decoder = keras.Model(inputs=latent_inputs, outputs=[Odsst.output])
        self.lweights = lweights
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
       # self.grad_loss_tracker = keras.metrics.Mean(name="grad_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.mse_loss_tracker
          #  self.grad_loss_tracker
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            # Encode.
            z = self.encoder(data)
            
            # Decode.
            r_oc1 = self.decoder(z)
            
            diff = data - r_oc1
            diff2 = tf.square(diff)
            
            # Calculate losses.
            mse_loss = self.lweights[0] * tf.reduce_mean(diff2)
           
            #grad_loss = 2 * mse_loss + (
            #    self.lweights[0]*self.get_gmm(diff[0],diff2))
 #               + #self.lweights[1]*self.get_gmm(diff[1],diff2[1]))
            
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
           # "grad_loss": self.grad_loss_tracker.result()
        }
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0] 
        # Encode.
        z = self.encoder(data)
        
        # Decode.
        r_oc1= self.decoder(z)

        # Calculate differences and squared differences.    
        diff = data-r_oc1
        diff2 = tf.square(diff)

        # Calculate losses.
        mse_loss = self.lweights[0]*tf.reduce_mean(diff2)
                
       # grad_loss = 2*mse_loss + (self.lweights[0]*self.get_gmm(diff[0],diff2[0])) 
        total_loss = mse_loss #+ self.lweights[2]*grad_loss
        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss
          #  "grad_loss": grad_loss
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
        Roc1 = self.decoder(Z)
        return Roc1
    
    def d_predict64(self,Z):
        Roc1 = self.decoder(Z)
        return Roc1.numpy().astype('float64')

    def get_gmm(self, diff,diff2):
        gmm = (tf.reduce_mean(diff2[:,0::diff2.shape[1]-1,1:-1,:])
        +tf.reduce_mean(diff2[:,1:-1,0::diff2.shape[2]-1,:])
        +2*tf.reduce_mean(diff2[:,1:-1,1:-1,:])
        -2*(  tf.reduce_mean(tf.math.multiply(diff[:,1:,:,:],diff[:,:-1,:,:]))  +tf.reduce_mean(tf.math.multiply(diff[:,:,1:,:],diff[:,:,:-1,:]))
            )
        )
