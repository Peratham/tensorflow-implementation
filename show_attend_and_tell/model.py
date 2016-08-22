import tensorflow as tf
import numpy as np
from layers import rnn_forward, rnn_step_forward_with_attention, lstm_forward, lstm_step_forward_with_attention
from layers import word_embedding_forward, affine_forward, affine_tanh_forward 
from layers import temporal_affine_forward, temporal_softmax_loss
from utils import init_weight, init_bias
"""
This is a implementation for attention based image captioning model.
There are some notations. 
N is batch size.
L is spacial size of feature vector (196)
D is dimension of image feature vector (512)
T is the number of time step which is equal to length of each caption.
V is vocabulary size. 
M is dimension of word vector which is embedding size.
H is dimension of hidden state.
"""

class CaptionGenerator(object):
    """
    CaptionGenerator produces functions:
        - build_model: receives features and captions then build graph where root nodes are loss and logits.  
        - build_sampler: receives features and build graph where root nodes are captions and alpha weights.
    """
    def __init__(self, word_to_idx, batch_size= 100, dim_feature=[196, 512], dim_embed=128, 
                    dim_hidden=128, n_time_step=None, cell_type='rnn', dtype=tf.float32):

        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        # Initialize some hyper parameters
        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.V = len(word_to_idx)
        self.N = batch_size
        self.H = dim_hidden
        self.M = dim_embed
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.T = n_time_step
        self.dtype = dtype
        self.params = {}

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize parameters for generating initial hidden and cell states
        self.params['W_init_h'] = init_weight('W_init_h', [self.D, self.H])
        self.params['b_init_h'] = init_bias('b_init_h', [self.H])
        self.params['W_init_c'] = init_weight('W_init_c', [self.D, self.H])
        self.params['b_init_c'] = init_bias('b_init_c', [self.H])

        # Initialize word vectors
        self.params['W_embed'] = init_weight('W_embed', [self.V, self.M])

        # Initialize parametres for attention layer 
        self.params['W_proj_x'] = init_weight('W_proj_x', [self.D, self.D])
        self.params['W_proj_h'] = init_weight('W_proj_h', [self.H, self.D])
        self.params['b_proj'] = init_bias('b_proj', [self.D])
        self.params['W_att'] = init_weight('W_att', [self.D, 1])


        # Initialize parameters for the RNN/LSTM
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = init_weight('Wx', [self.M, self.H * dim_mul])
        self.params['Wh'] = init_weight('Wh', [self.H, self.H * dim_mul])
        self.params['Wz'] = init_weight('Wz', [self.D, self.H * dim_mul])
        self.params['b'] = init_bias('b', [self.H * dim_mul])

        # Initialize parameters for output-to-vocab
        self.params['W_vocab'] = init_weight('W_vocab', [self.H, self.V])
        self.params['b_vocab'] = init_bias('b_vocab', [self.V])

        # Cast parameters to correct dtype
        for k, v in self.params.iteritems():
            self.params[k] = tf.cast(v, self.dtype)
            
        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [self.N, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [self.N, self.T + 1])


    def build_model(self):
        """
        Place Holder:
        - features: input image features of shape (N, L, D)
        - captions: ground-truth captions; an integer array of shape (N, T+1) where
          each element is in the range [0, V)
        Returns
        - logits: score of shape (N, T, V)
        - loss: scalar loss
        """
        # place holder features and captions
        features = self.features
        captions = self.captions

        # parameters
        params = self.params

        # hyper parameters
        hyper_params = {'batch_size': self.N, 'spacial_size': self.L, 'dim_feature': self.D,
                            'n_time_step': self.T, 'dim_hidden': self.H, 'vocab_size': self.V}

        # captions for input/output and mask matrix
        captions_in = captions[:, :self.T]      # same as captions[:, :-1], tensorflow doesn't provide negative stop slice yet.
        captions_out = captions[:, 1:]  
        mask = tf.not_equal(captions_out, self._null)
        
        
        # generate initial hidden state using cnn features 
        mean_features = tf.reduce_mean(features, 1)
        h0 = affine_tanh_forward(mean_features, params['W_init_h'], params['b_init_h'])  # (N, H)
        c0 = affine_tanh_forward(mean_features, params['W_init_c'], params['b_init_c'])  # (N, h)

        # generate input x (word vector)
        x = word_embedding_forward(captions_in, params['W_embed'])  # (N, T, M)

        # lstm forward
        if self.cell_type == 'rnn':
            h = rnn_forward(x, features, h0, params, hyper_params)
        else: 
            h = lstm_forward(x, features, h0, c0, params, hyper_params)   # (N, T, H)

        # hidden-to-vocab
        logits = temporal_affine_forward(h, params['W_vocab'], params['b_vocab'], hyper_params)  # (N, T, V)
    
        # softmax loss
        loss = temporal_softmax_loss(logits, captions_out, mask, hyper_params)

        # generated word indices
        generated_captions = tf.argmax(logits, 2)   # (N, T)

        return loss, generated_captions


    def build_sampler(self, max_len=20):
        """
        Input:
        - max_len: max length for generating cations
        Place Holder:
        - features: input image features of shape (N, L, D)
        
        Returns
        - sampled_words: sampled word indices
        - alphas: sampled alpha weights
        """

        # place holder
        features = self.features

        #parameters
        params = self.params

        # hyper parameters 
        hyper_params = {'batch_size': self.N, 'spacial_size': self.L, 'dim_feature': self.D,
                            'n_time_step': self.T, 'dim_hidden': self.H, 'vocab_size': self.V}

        # generate initial hidden state using cnn features 
        mean_features = tf.reduce_mean(features, 1)
        prev_h = affine_tanh_forward(mean_features, params['W_init_h'], params['b_init_h'])  # (N, H)
        prev_c = affine_tanh_forward(mean_features, params['W_init_c'], params['b_init_c'])  # (N, h)

        sampled_word_list = []
        alpha_list = []

        for t in range(max_len):
            # embed the previous generated word
            if t == 0:
                x = tf.zeros([self.N, self.M])            # what about assign word vector for '<START>' token ?
            else:
                x = word_embedding_forward(sampled_word, params['W_embed'])    # (N, M)

            # lstm forward
            if self.cell_type == 'rnn':
                h, alpha = rnn_step_forward_with_attention(x, features, prev_h, params, hyper_params)    #  (N, H), (N, L)
            else: 
                h, c, alpha = lstm_step_forward_with_attention(x, features, prev_h, prev_c, params, hyper_params)    # (N, H), (N, H), (N, L)
                prev_c = c
                
            # prepare for next time step
            prev_h = h
            
            # save alpha weights
            alpha_list.append(alpha)

            # generate scores(logits) from current hidden state
            logits = affine_forward(h, params['W_vocab'], params['b_vocab'])       # (N, V)

            # sample word indices with logits
            sampled_word = tf.argmax(logits, 1)        # (N, ) where value is in the range of [0, V) 
            sampled_word_list.append(sampled_word)        # tensor flow doesn't provide item assignment 

        alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))     #  (N, T, L)
        sampled_captions = tf.transpose(tf.pack(sampled_word_list), (1,0))     # (N, max_len)

        return alphas, sampled_captions