import tensorflow as tf
from rnn_layers import word_embedding_forward, rnn_forward, affine_forward
from rnn_layers import temporal_affine_forward, temporal_softmax_loss

"""
There are some notation. 
N is batch size.
T is the number of time step which is equal to length of each caption.
V is vocabulary size. 
F is dimension of image feature vector (4096 or 512)
M is dimension of word vector which is embedding size.
H is dimension of hidden state.
"""

class ImageCaptioning(object):


    def __init__(self, word_to_idx, batch_size= 100, dim_feature=512, dim_embed=128, dim_hidden=128, n_time_step=None, dtype=tf.float32):
        # initialize ..
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.V = len(word_to_idx)
        self.N = batch_size
        self.H = dim_hidden
        self.M = dim_embed
        self.F = dim_feature
        self.T = n_time_step
        self.dtype = dtype
        self.params = {}

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = tf.Variable(tf.truncated_normal([self.V, self.M], stddev=0.1), name= 'W_embed')

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = tf.Variable(tf.truncated_normal([self.F, self.H], stddev=0.1), name= 'W_proj')
        self.params['b_proj'] = tf.Variable(tf.zeros([self.H]), name='b_proj')

        # Initialize parameters for the RNN
        #dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = tf.Variable(tf.truncated_normal([self.M, self.H], stddev=0.1), name= 'Wx')
        self.params['Wh'] = tf.Variable(tf.truncated_normal([self.H, self.H], stddev=0.1), name= 'Wh')
        self.params['b'] = tf.Variable(tf.zeros([self.H]), name='b')

        # Initialize output to vocab weights
        self.params['W_vocab'] = tf.Variable(tf.truncated_normal([self.H, self.V], stddev=0.1), name= 'W_vocab')
        self.params['b_vocab'] = tf.Variable(tf.zeros([self.V]), name='b_vocab')

        # Cast parameters to correct dtype
        for k, v in self.params.iteritems():
            self.params[k] = tf.cast(v, self.dtype)

    def build_model(self, features, captions):
        """
        Inputs:
        - features: input image features of shape (N, F)
        - captions: ground-truth captions; an integer array of shape (N, T) where
          each element is in the range [0, V)

        Returns
        - logits: score of shape (N, T, V)
        - loss: Scalar loss
        """

        # caption in, out and mask matrix
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]	    
        mask = (captions_out != self._null)

        # word embedding matrix
        W_embed = self.params['W_embed'] 

        # parameters for (cnn_features)-to-(initial_hidden)
        W_proj = self.params['W_proj'] 
        b_proj = self.params['b_proj'] 

        # parameters for input-to-hidden, hidden-to-hidden
        Wx = self.params['Wx'] 
        Wh = self.params['Wh']
        b = self.params['b'] 

        # parameters for hidden-to-vocab 
        W_vocab = self.params['W_vocab'] 
        b_vocab = self.params['b_vocab'] 

        # some hyper-parameters
        T = self.T
        N = self.N
        V = self.V
        H = self.H
        M = self.M
        F = self.F

        # params used in some function call
        rnn_param = {'n_time_step': T}
        param = {'batch_size': N, 'n_time_step': T, 'dim_hidden': H, 'vocab_size': V}

        # generate initial hidden state using cnn features 
        h0 = affine_forward(features, W_proj, b_proj)  # (N, H)

        # generate input x (word vector)
        x = word_embedding_forward(captions_in, W_embed)  # (N, T, M)

        # rnn forward
        h = rnn_forward(x, h0, Wx, Wh, b, rnn_param)  # (N, T, H)

        # hidden-to-vocab
        logits = temporal_affine_forward(h, W_vocab, b_vocab, param)
    
        # softmax loss
        loss = temporal_softmax_loss(logits, captions_out, mask, param)

        return logits, loss