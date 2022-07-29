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
        oc_last_hidden_dimx = np.int(oc_dimx/(8*avepool_size)); oc_last_hidden_dimy = np.int(oc_dimy/(8*avepool_size)); oc_channels=3
        
         # po Encoder:
        cc=0;occ2=3
        encoder_oc_inputs_2 = keras.Input(shape=(oc_dimx, oc_dimy, occ2), name="oc%02d_input" % cc); cc+=1
        ocpoE = layers.Conv2D(4*occ2, (9,9), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="oc%02d_conv1a" % cc)(encoder_oc_inputs_2); cc+=1 
        ocpoE = layers.Conv2D(8*occ2, (5,5), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="oc%02d_conv1c" % cc)(ocpoE); cc+=1
        ocpoE = layers.MaxPooling2D((2,2), padding='same',name="oc%02d_pool1" % cc)(ocpoE); cc+=1                            
        ocpoE = layers.Conv2D(16*occ2, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="oc%02d_conv2a" % cc)(ocpoE); cc+=1
        ocpoE = layers.Conv2D(16*occ2, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="oc%02d_conv2de" % cc)(ocpoE); cc+=1
        ocpoE = layers.MaxPooling2D((2,2), padding='same',name="oc%02d_pool1x" % cc)(ocpoE); cc+=1                            
        ocpoE = layers.Conv2D(16*occ2, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="oc%02d_conv2ax" % cc)(ocpoE); cc+=1               
        ocpoE = layers.Conv2D(16*occ2, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="oc%02d_conv2azz" % cc)(ocpoE); cc+=1
        ocpoE = layers.MaxPooling2D((2,2), padding='same',name="oc%02d_pool2" % cc)(ocpoE); cc+=1                            
        ocpoE = layers.Conv2D(16*occ2, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="oc%02d_conv3a" % cc)(ocpoE); cc+=1
        ocpoE = layers.AveragePooling2D(pool_size=(avepool_size, avepool_size), padding="same", name="oc%02d_avepool1" % cc)(ocpoE); cc+=1
        ocpoE = layers.Conv2D(16*occ2, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same", name="oc%02d_conv3a" % cc)(ocpoE); cc+=1
        ocpoE = layers.Flatten(name="oc%02d_flatten" % cc)(ocpoE); cc+=1
        ocpoE = keras.Model(inputs=encoder_oc_inputs_2,outputs=ocpoE)

        c3 = ocpoE.output

        c3 = layers.Dense(beta3, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar)(c3)

        Z = c3
        
        encoder = keras.Model(inputs = ocpoE.input, outputs = Z)
        print(encoder.summary())
        
        self.encoder = keras.Model(inputs = ocpoE.input, outputs = Z)

        latent_inputs = keras.Input(shape=(beta3,), name="00_latent")
        c3=split()(latent_inputs,beta3)
        Pod = c3

        Pod = layers.Dense(ocpoE.output.shape[1], kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar)(Pod)


         # po Decoder
        cc = 1
        Pod = layers.Reshape((8, 8, 16*occ2),name="oc%02d_reshape" % cc)(Pod); cc+=1
        Pod = layers.Conv2DTranspose(16*occ2, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="oc%02d_transposeq" % cc)(Pod); cc+=1
        Pod = layers.UpSampling2D(size=(avepool_size, avepool_size),name="oc%02d_upsample" % cc)(Pod); cc+=1
        Pod = layers.Conv2DTranspose(16*occ2, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="oc%02d_transposew" % cc)(Pod); cc+=1
        Pod = layers.Conv2DTranspose(16*occ2, (3,3), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="oc%02d_transpose1y" % cc)(Pod); cc+=1
        Pod = layers.Conv2DTranspose(16*occ2, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="oc%02d_transpose1e" % cc)(Pod); cc+=1
        Pod = layers.Conv2DTranspose(16*occ2, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="oc%02d_transpose1r" % cc)(Pod); cc+=1
        Pod = layers.Conv2DTranspose(16*occ2, (3,3), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="oc%02d_transpose1zz" % cc)(Pod); cc+=1
        Pod = layers.Conv2DTranspose(16*occ2, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="oc%02d_transpose2a" % cc)(Pod); cc+=1
        Pod = layers.Conv2DTranspose(8*occ2, (3,3), activation="elu", strides=2, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="oc%02d_transpose1t" % cc)(Pod); cc+=1
        Pod = layers.Conv2DTranspose(4*occ2, (3,3), activation="elu", strides=1, kernel_regularizer=kr, bias_regularizer=br,activity_regularizer=ar, padding="same",name="oc%02d_transpose1tt" % cc)(Pod); cc+=1
        Pod = layers.Conv2DTranspose(occ2, (9,9), strides=2, padding="same", name="oc%02d_transpose2c" % cc)(Pod); cc+=1
        Pod = keras.Model(inputs=latent_inputs, outputs=Pod)
        
        decoder = keras.Model(inputs=latent_inputs, outputs=Pod.output)
        print(decoder.summary())
        

        # Combine
        self.decoder = keras.Model(inputs=latent_inputs, outputs=Pod.output)
        self.lweights = lweights
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
     #   self.grad_loss_tracker = keras.metrics.Mean(name="grad_loss")

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
        # elif isinstance(data, list):
        #     at, oc = data
        with tf.GradientTape() as tape:
            # Encode.
            z = self.encoder(data)
            
            # Decode.
            r_oc1 = self.decoder(z)

            diff = data - r_oc1
            diff2 = tf.square(diff)

            # Calculate losses.
            mse_loss = self.lweights[0]*tf.reduce_mean(diff2)
            
            #mse_loss = [x for x in mse_loss if x != None]
            #grad_loss = 2*mse_loss + (
 #               self.lweights[0]*self.get_gmm(diff[0],diff2[0])
 #               + #self.lweights[1]*self.get_gmm(diff[1],diff2[1]))
            
            total_loss = mse_loss # + (self.lweights[2]*grad_loss)
        
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
        r_oc1 = self.decoder(z)

        # Calculate differences and squared differences.    
        diff = data - r_oc1
        diff2 = tf.square(diff)

        # Calculate losses.
        mse_loss = self.lweights[0]*tf.reduce_mean(diff2)

       # grad_loss = 2*mse_loss + (self.lweights[0]*self.get_gmm(diff[0],diff2[0]) + self.lweights[1]*self.get_gmm(diff[1],diff2[1]))
        total_loss = mse_loss #+self.lweights[2]*grad_loss
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
