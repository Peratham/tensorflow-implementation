import tensorflow as tf
import numpy as np
from rnn_layers import rnn_forward, lstm_forward, lstm_step_forward_with_attention
from rnn_layers import word_embedding_forward, affine_forward, affine_tanh_forward 
from rnn_layers import temporal_affine_forward, temporal_softmax_loss

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
        self.params['W_init_h'] = tf.Variable(tf.truncated_normal([self.D, self.H], stddev=1.0/np.sqrt(self.D)), name= 'W_init_h')
        self.params['b_init_h'] = tf.Variable(tf.zeros([self.H]), name='b_init_h')
        self.params['W_init_c'] = tf.Variable(tf.truncated_normal([self.D, self.H], stddev=1.0/np.sqrt(self.D)), name= 'W_init_c')
        self.params['b_init_c'] = tf.Variable(tf.zeros([self.H]), name='b_init_c')

        # Initialize word vectors
        self.params['W_embed'] = tf.Variable(tf.truncated_normal([self.V, self.M], stddev=1.0/np.sqrt(self.V)), name= 'W_embed')

        # Initialize parametres for attention layer 
        self.params['W_proj_x'] = tf.Variable(tf.truncated_normal([self.D, self.D], stddev=1.0/np.sqrt(self.D)), name= 'W_proj_x')
        self.params['W_proj_h'] = tf.Variable(tf.truncated_normal([self.H, self.D], stddev=1.0/np.sqrt(self.H)), name= 'W_proj_h')
        self.params['b_proj'] = tf.Variable(tf.zeros([self.D]), name='b_proj')
        self.params['W_att'] = tf.Variable(tf.truncated_normal([self.D, 1], 1.0/np.sqrt(self.D)), name= 'W_att')


        # Initialize parameters for the RNN/LSTM
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = tf.Variable(tf.truncated_normal([self.M, self.H * dim_mul], stddev=1.0/np.sqrt(self.M)), name= 'Wx')
        self.params['Wh'] = tf.Variable(tf.truncated_normal([self.H, self.H * dim_mul], stddev=1.0/np.sqrt(self.H)), name= 'Wh')
        self.params['Wz'] = tf.Variable(tf.truncated_normal([self.D, self.H * dim_mul], stddev=1.0/np.sqrt(self.D)), name= 'Wz')
        self.params['b'] = tf.Variable(tf.zeros([self.H * dim_mul]), name='b')

        # Initialize parameters for output-to-vocab
        self.params['W_vocab'] = tf.Variable(tf.truncated_normal([self.H, self.V], stddev=1.0/np.sqrt(self.H)), name= 'W_vocab')
        self.params['b_vocab'] = tf.Variable(tf.zeros([self.V]), name='b_vocab')

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

        # some hyper-parameters
        T = self.T
        N = self.N
        V = self.V
        H = self.H
        M = self.M
        L = self.L
        D = self.D

        # hyper_params used in some function calls
        hyper_params = {'batch_size': N, 'spacial_size': L, 'dim_feature': D,
                            'n_time_step': T, 'dim_hidden': H, 'vocab_size': V}

        # captions for input/output and mask matrix
        captions_in = captions[:, :T]      # same as captions[:, :-1], tensorflow doesn't provide negative stop slice yet.
        captions_out = captions[:, 1:]  
        mask = tf.not_equal(captions_out, self._null)
        
        # word embedding matrix
        W_embed = self.params['W_embed'] 

        # parameters for generating initial hidden/cell state
        W_init_h = self.params['W_init_h'] 
        b_init_h = self.params['b_init_h'] 
        W_init_c = self.params['W_init_c'] 
        b_init_c = self.params['b_init_c'] 

        # parameters for input-to-hidden, hidden-to-hidden
        Wx = self.params['Wx'] 
        Wh = self.params['Wh']
        b = self.params['b'] 

        # parameters for hidden-to-vocab 
        W_vocab = self.params['W_vocab'] 
        b_vocab = self.params['b_vocab'] 

        # generate initial hidden state using cnn features 
        mean_features = tf.reduce_mean(features, 1)
        h0 = affine_tanh_forward(mean_features, W_init_h, b_init_h)  # (N, H)
        c0 = affine_tanh_forward(mean_features, W_init_c, b_init_c)  # (N, h)

        # generate input x (word vector)
        x = word_embedding_forward(captions_in, W_embed)  # (N, T, M)

        # lstm forward
        if self.cell_type == 'rnn':
            h = rnn_forward(x, h0, Wx, Wh, b, hyper_params)
        else: 
            h = lstm_forward(x, features, h0, c0, self.params, hyper_params)   # (N, T, H)

        # hidden-to-vocab
        logits = temporal_affine_forward(h, W_vocab, b_vocab, hyper_params)  # (N, T, V)
    
        # softmax loss
        loss = temporal_softmax_loss(logits, captions_out, mask, hyper_params)

        # generated word indices
        generated_captions = tf.argmax(logits, 2)   # (N, T)

        return loss, generated_captions


    def build_sampler(self, max_len=20):
        """
        Input:
        - max_len: max length of generating cations

        Place Holder:
        - features: input image features of shape (N, L, D)
        
        Returns
        - sampled_words: sampled word indices
        - alphas: sampled alpha weights
        """

        # place holder
        features = self.features

        # some hyper-parameters
        T = self.T
        N = self.N
        V = self.V
        H = self.H
        M = self.M
        L = self.L
        D = self.D

        # hyper_params used in some function calls
        hyper_params = {'batch_size': N, 'spacial_size': L, 'dim_feature': D,
                            'n_time_step': T, 'dim_hidden': H, 'vocab_size': V}

        # word embedding matrix
        W_embed = self.params['W_embed'] 

        # parameters for generating initial hidden/cell state
        W_init_h = self.params['W_init_h'] 
        b_init_h = self.params['b_init_h'] 
        W_init_c = self.params['W_init_c'] 
        b_init_c = self.params['b_init_c'] 

        # parameters for input-to-hidden, hidden-to-hidden
        Wx = self.params['Wx'] 
        Wh = self.params['Wh']
        b = self.params['b'] 

        # parameters for hidden-to-vocab 
        W_vocab = self.params['W_vocab'] 
        b_vocab = self.params['b_vocab'] 

        # generate initial hidden state using cnn features 
        mean_features = tf.reduce_mean(features, 1)
        prev_h = affine_tanh_forward(mean_features, W_init_h, b_init_h)  # (N, H)
        prev_c = affine_tanh_forward(mean_features, W_init_c, b_init_c)  # (N, h)

        sampled_word_list = []
        alpha_list = []

        for t in range(max_len):
            # embed the previous generated word
            if t == 0:
                x = tf.zeros([N, M])            # what about assign word vector for '<START>' token ?
            else:
                x = word_embedding_forward(sampled_word, W_embed)    # (N, M)

            # lstm forward
            if self.cell_type == 'rnn':
                h = rnn_forward(x, h0, Wx, Wh, b, hyper_params)
            else: 
                h, c, alpha = lstm_step_forward_with_attention(x, features, prev_h, prev_c, self.params, hyper_params) # (N, H), (N, H), (N, L)

            # save alpha weights
            alpha_list.append(alpha)

            # generate scores from current hidden state and sample word indices
            logits = affine_forward(h, W_vocab, b_vocab)     # (N, V)
            sampled_word = tf.argmax(logits, 1)      # (N, ) where value is in the range of [0, V) 
            sampled_word_list.append(sampled_word)  # tensor flow doesn't provide item assignment 

        alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2)) #  (N, T, L)
        sampled_words = tf.transpose(tf.pack(sampled_word_list), (1,0)) # (N, max_len)

        return alphas, sampled_words
