
import tensorflow as tf
from tensorflow.keras.layers import  Dense, Input, Masking, TimeDistributed
import numpy as np

'''
*__________________________________________*
|                                          |
| Version 3 (03/09/2020)                   |
| Angle Alphabet Cluster neurons into the  |
| akeras Layer. For mor information, please|
| please, refer to the github and the      |
| helper below.                            |
| Author: Fernando Limoeiro                |
| Inspired on the works of m3h0w and       |
| AlQuraish                                |
| Refs here                                |
|                                          |
|                                          |
|__________________________________________|
*                                          *
'''

class Alphabet(tf.keras.layers.Layer):
    def __init__(self,n_alphas=10, n_angles=2, alpha_range = [-np.pi,np.pi],
                 force_alpha_range=True, task='dih', name='alphabet_layer', **kwargs):
        
        '''
        Alphabet Layer:
        This layer consists of a Densely connected network followed by a linear/angular regression operation.
        The main purpose is to convert the information gathered from the core models into a probability distribution
        via softmax function. By doing this we then perform the "angular regression" and achieve angle prediction.
        
        Angular prediction comes in two flavors: angular regression (as of now) and vectorized (later on).
        Layer parameters:
        n_alphas: (int) number of angle alphabet units to consider
        n_angles: (int) number of angles to predict (2: [phi, psi], 3: [phi, psi, omega])
        alpha_range: (list,tuple) the initialization range for the alphabet.
        force_alpha_range: (bool) wether or not to force the alphabet to stay on the initial range.
        mode: (str) the alphabet mode, can be 'dih' for [phi,psi] angles or 'zmat' for [theta,phi].
        Just a disclaimer:
        It doest support """" masking """" to some extent. It means that the layer receives a 'mask' tensor
        propagated from previous layers (originally from Masking) and casts it to float32. Followed by a 
        tensor product between float_mask and output from Alphabet. The idea is that, by passing the mask and
        performing the product, BP algorithm will gradually propagatethe importance of sequence length.
        I have no clues whatsoever if this will work as intended, but  it is a try. 
        '''
        
        super(Alphabet, self).__init__(**kwargs)
        self.n_angles = n_angles
        self.n_alphas = n_alphas
        self.alpha_range = alpha_range
        self.task = task
        self.force_alpha_range = force_alpha_range
        
    def init_alphabet(self):
        '''
        Initializes the Angles Alphabet Matrix
        returns:
            Alphabet: matrix of floats of shape (n_alphas, n_angles)
        '''
        if self.task == 'dih':
            # return the initialized alphabet
            return tf.random.uniform(shape=(self.n_alphas, self.n_angles),
                                     minval=self.alpha_range[0], maxval=self.alpha_range[1],
                                     dtype='float32')
        elif self.task == 'zmat':
            # for now, will be initializing in the same manner as dih
            # but in future versions there will be 2 angle alphabets
            return tf.random.uniform(shape=(self.n_alphas, self.n_angles),
                                     minval=self.alpha_range[0], maxval=self.alpha_range[1],
                                     dtype='float32')
            
        
    def compute_mask(self, inputs, mask):
        # Time step masks must be the same for each input.
        # This is because the mask for an RNN is of size [batch, time_steps, 1],
        # and specifies which time steps should be skipped, and a time step
        # must be skipped for all inputs.
        # TODO(scottzhu): Should we accept multiple different masks?
        '''
        mask = nest.flatten(mask)[0]
        output_mask = mask if self.return_sequences else None
        if self.return_state:
            state_mask = [None for _ in self.states]
            return [output_mask] + state_mask
        else:
            return output_mask
        '''
        if mask is not None:
            return mask
    
    def get_config(self):
        # overiding get_config function
        # to enable layer serialization

        config = super().get_config().copy()
        config.update({
                    'n_alphas': self.n_alphas, 
                    'n_angles': self.n_angles,
                    'alpha_range': self.alpha_range,
                    'force_alpha_range': self.force_alpha_range,
                    'mode':self.task
                    })
        
        return config
    
    def build(self, input_shape):
        self.dense_unit = TimeDistributed(tf.keras.layers.Dense(self.n_alphas,
                                                activation='relu',
                                                use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                bias_initializer="zeros"))
        
        self.alphabet = tf.Variable(self.init_alphabet(),
                                    trainable = True, name='alphabet')
        
    def call(self, inputs, mask=None):
        '''
        This method is called everytime the layer is invoked
        by the model. Actually what it does is passing the input by a 
        TimeDistributed Dense and then softmax it along the last axis and
        perform the weighted average ("angular regression") over the
        alphabet array. Which, in turn, is also a learnable parameter.4
        '''
        outputs = self.dense_unit(inputs)
        outputs = tf.keras.activations.softmax(outputs)
        outputs = tf.einsum('ij,kbi->kbj', self.alphabet, outputs)
        '''
        if self.force_alpha_range:
            self.alphabet = tf.clip_by_value(self.alphabet, -np.pi,np.pi)
        ''' 
        if mask is not None:
            mask = tf.cast(mask, "float32")
            outputs = outputs * mask[..., tf.newaxis]
        return outputs
